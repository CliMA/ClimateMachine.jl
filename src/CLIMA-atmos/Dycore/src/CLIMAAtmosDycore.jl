module CLIMAAtmosDycore

using Printf: @sprintf
using Canary, MPI, Requires
using Logging

using PlanetParameters: R_d, cp_d, grav
using Parameters
@parameter cv_d cp_d-R_d "Isochoric specific heat dry air"
@parameter gamma_d cp_d/cv_d "Heat capcity ratio of dry air"
@parameter gdm1 R_d/cv_d "(equivalent to gamma_d-1)"

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" include("gpu.jl")

include("vtk.jl")
include("callbacks.jl")

# {{{ constants
# note the order of the fields below is also assumed in the code.
const _nstate = 5
const _U, _V, _W, _ρ, _E = 1:_nstate
const stateid = (U = _U, V = _V, W = _W, ρ = _ρ, E = _E)

const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
       _x, _y, _z = 1:_nvgeo
const vgeoid = (ξx = _ξx, ηx = _ηx, ζx = _ζx,
                ξy = _ξy, ηy = _ηy, ζy = _ζy,
                ξz = _ξz, ηz = _ηz, ζz = _ζz,
                MJ = _MJ, MJI = _MJI,
                 x = _x,   y = _y,   z = _z)

const _nsgeo = 5
const _nx, _ny, _nz, _sMJ, _vMJI = 1:_nsgeo
const sgeoid = (nx = _nx, ny = _ny, nz = _nz, sMJ = _sMJ, vMJI = _vMJI)

# }}}

# {{{ Parameters (aka input file)
struct Parameters
  DFloat::Type
  DeviceArray::Type
  dim::Integer
  # Callback that given a part and number of parts generates the mesh data
  meshgenerator::Function
  meshwarp::Function
  N::Integer
  discretization::Symbol
  gravity::Bool
end
#}}}

# {{{ Configuration (aka auxiliary state needed to run an simulation)
struct Configuration{HostArray, DeviceArray}
  mpicomm
  mesh
  vgeo::DeviceArray
  sgeo::DeviceArray
  elemtobndy::DeviceArray
  vmapM::DeviceArray
  vmapP::DeviceArray
  sendelems::DeviceArray
  sendreq::HostArray
  recvreq::HostArray
  host_sendQ::HostArray
  host_recvQ::HostArray
  device_sendQ::DeviceArray
  device_recvQ::DeviceArray
  D::DeviceArray
  function Configuration(parameters, mpicomm)
    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)
    N = parameters.N
    dim = parameters.dim
    DFloat = parameters.DFloat

    mesh = parameters.meshgenerator(mpirank+1, mpisize)

    mpirank == 0 && @info "partiting mesh..."
    mesh = partition(mpicomm, mesh...)

    # Connect the mesh in parallel
    mpirank == 0 && @info "connecting mesh..."
    mesh = connectmesh(mpicomm, mesh...)

    # Get the vmaps
    mpirank == 0 && @info "computing mappings..."
    (vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface,
                              mesh.elemtoordr)

    # Create 1-D operators
    (ξ, ω) = lglpoints(DFloat, N)
    D = spectralderivative(ξ)

    # Compute the geometry
    mpirank == 0 && @info "computing metrics..."
    (vgeo, sgeo) = computegeometry(Val(dim), mesh, D, ξ, ω,
                                   parameters.meshwarp, vmapM)
    (nface, nelem) = size(mesh.elemtoelem)

    mpirank == 0 && @info "create RHS storage..."

    mpirank == 0 && @info "create send/recv request storage..."
    nnabr = length(mesh.nabrtorank)
    sendreq = fill(MPI.REQUEST_NULL, nnabr)
    recvreq = fill(MPI.REQUEST_NULL, nnabr)

    mpirank == 0 && @info "create send/recv storage..."
    sendQ = zeros(DFloat, (N+1)^dim, _nstate, length(mesh.sendelems))
    recvQ = zeros(DFloat, (N+1)^dim, _nstate, length(mesh.ghostelems))

    mpirank == 0 && @info "create configuration struct..."
    HostArray = Array
    DeviceArray = parameters.DeviceArray
    # FIXME: Handle better for GPU?
    new{HostArray, DeviceArray}(mpicomm, mesh, DeviceArray(vgeo),
                                DeviceArray(sgeo), DeviceArray(mesh.elemtobndy),
                                DeviceArray(vmapM), DeviceArray(vmapP),
                                DeviceArray(mesh.sendelems), sendreq, recvreq,
                                sendQ, recvQ, DeviceArray(sendQ),
                                DeviceArray(recvQ), DeviceArray(D))
  end
end

# {{{ compute geometry
function computegeometry(::Val{dim}, mesh, D, ξ, ω, meshwarp, vmapM) where dim
  # Compute metric terms
  Nq = size(D, 1)
  DFloat = eltype(D)

  (nface, nelem) = size(mesh.elemtoelem)

  crd = creategrid(Val(dim), mesh.elemtocoord, ξ)

  vgeo = zeros(DFloat, Nq^dim, _nvgeo, nelem)
  sgeo = zeros(DFloat, _nsgeo, Nq^(dim-1), nface, nelem)

  (ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, MJ, MJI, x, y, z) =
      ntuple(j->(@view vgeo[:, j, :]), _nvgeo)
  J = similar(x)
  (nx, ny, nz, sMJ, vMJI) = ntuple(j->(@view sgeo[ j, :, :, :]), _nsgeo)
  sJ = similar(sMJ)

  X = ntuple(j->(@view vgeo[:, _x+j-1, :]), dim)
  creategrid!(X..., mesh.elemtocoord, ξ)

  @inbounds for j = 1:length(x)
    (x[j], y[j], z[j]) = meshwarp(x[j], y[j], z[j])
  end

  # Compute the metric terms
  if dim == 1
    computemetric!(x, J, ξx, sJ, nx, D)
  elseif dim == 2
    computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)
  elseif dim == 3
    computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ,
                   nx, ny, nz, D)
  end

  M = kron(1, ntuple(j->ω, dim)...)
  MJ .= M .* J
  MJI .= 1 ./ MJ
  vMJI .= MJI[vmapM]

  sM = dim > 1 ? kron(1, ntuple(j->ω, dim-1)...) : one(DFloat)
  sMJ .= sM .* sJ

  (vgeo, sgeo)
end
# }}}
# }}}

# {{{ State (aka the time varying problem arrays)
mutable struct State{DeviceArray}
  time
  Q::DeviceArray
  function State(parameters::Parameters,
                 configuration::Configuration{HostArray, DeviceArray},
                 time) where {HostArray, DeviceArray}
    N = parameters.N
    dim = parameters.dim
    DFloat = parameters.DFloat
    nelem = size(configuration.mesh.elemtoelem, 2)

    # FIXME: Handle better for GPU?
    Q = DeviceArray(zeros(DFloat, (N+1)^dim, _nstate, nelem))
    new{DeviceArray}(time, Q)
  end
end
#}}}

rhs!(rhs, s, p::Parameters, c...) = rhs!(Val{p.discretization}, rhs, s, p, c...)

include("vanilla_euler.jl")

include("lsrk.jl")

# {{{ L2 Energy (for all dimensions)
function L2energysquared(::Val{dim}, ::Val{N}, Q, vgeo, elems) where {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  (~, nstate, nelem) = size(Q)

  energy = zero(DFloat)

  @inbounds for e = elems, q = 1:nstate, i = 1:Np
    energy += vgeo[i, _MJ, e] * Q[i, q, e]^2
  end

  energy
end
# }}}

# {{{ cfl
function cfl(::Val{dim}, ::Val{N}, vgeo, Q, mpicomm) where {dim, N}
  DFloat = eltype(Q)

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  dt = [floatmax(DFloat)]

  if dim == 2
    @inbounds for e = 1:nelem, n = 1:Np
      ρ, U, V = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e]
      E = Q[n, _E, e]
      y = vgeo[n, _y, e]
      P = gdm1*(E - (U^2 + V^2)/(2*ρ) - ρ*grav*y)

      ξx, ξy, ηx, ηy = vgeo[n, _ξx, e], vgeo[n, _ξy, e],
                       vgeo[n, _ηx, e], vgeo[n, _ηy, e]

      loc_dt = 2ρ / max(abs(U * ξx + V * ξy) + ρ * sqrt(gamma_d * P / ρ),
                        abs(U * ηx + V * ηy) + ρ * sqrt(gamma_d * P / ρ))
      dt[1] = min(dt[1], loc_dt)
    end
  end

  if dim == 3
    @inbounds for e = 1:nelem, n = 1:Np
      ρ, U, V, W = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e]
      E = Q[n, _E, e]
      z = vgeo[n, _z, e]
      P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*grav*z)

      ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
      ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
      ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]

      loc_dt = 2ρ / max(abs(U * ξx + V * ξy + W * ξz) + ρ * sqrt(gamma_d * P / ρ),
                        abs(U * ηx + V * ηy + W * ηz) + ρ * sqrt(gamma_d * P / ρ),
                        abs(U * ζx + V * ζy + W * ζz) + ρ * sqrt(gamma_d * P / ρ))
      dt[1] = min(dt[1], loc_dt)
    end
  end

  MPI.Allreduce(dt[1], MPI.MIN, mpicomm)
end
# }}}

end # module

#=
=#

#=
# Compute time step
mpirank == 0 && println("[CPU] computing dt (CPU)...")
base_dt = cfl(Val(dim), Val(N), vgeo, Q, mpicomm) / N^√2
mpirank == 0 && @show base_dt

nsteps = ceil(Int64, tend / base_dt)
dt = tend / nsteps
mpirank == 0 && @show (dt, nsteps, dt * nsteps, tend)

# Do time stepping
stats = zeros(DFloat, 2)
mpirank == 0 && println("[CPU] computing initial energy...")
stats[1] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)

# plot the initial condition
mkpath("viz")
# TODO: Fix VTK for 1-D
if dim > 1
  X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                        nelem), dim)
  ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
  U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
  V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
  W = reshape((@view Q[:, _W, :]), ntuple(j->(N+1),dim)..., nelem)
  E = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
  writemesh(@sprintf("viz/euler%dD_%s_rank_%04d_step_%05d",
                     dim, ArrType, mpirank, 0), X...;
            fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
            realelems=mesh.realelems)
end

mpirank == 0 && println("[DEV] starting time stepper...")

# TODO: Fix VTK for 1-D
if dim > 1
  X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                        nelem), dim)
  ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
  U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
  V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
  W = reshape((@view Q[:, _W, :]), ntuple(j->(N+1),dim)..., nelem)
  E = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
  writemesh(@sprintf("viz/euler%dD_%s_rank_%04d_step_%05d",
                     dim, ArrType, mpirank, nsteps), X...;
            fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
            realelems=mesh.realelems)
end

mpirank == 0 && println("[CPU] computing final energy...")
stats[2] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)

stats = sqrt.(MPI.allreduce(stats, MPI.SUM, mpicomm))

if  mpirank == 0
  @show eng0 = stats[1]
  @show engf = stats[2]
  @show Δeng = engf - eng0
end
=#
