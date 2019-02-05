module VanillaAtmosDiscretizations
using MPI

using ..CLIMAAtmosDycore
AD = CLIMAAtmosDycore
using CLIMAAtmosDycore.Grids
using CLIMAAtmosDycore.AtmosStateArrays

export VanillaAtmosDiscretization

using ParametersType
using PlanetParameters: cp_d, cv_d, R_d, grav
@parameter gamma_d cp_d/cv_d "Heat capcity ratio of dry air"
@parameter gdm1 R_d/cv_d "(equivalent to gamma_d-1)"

@parameter prandtl 71//10 "Prandtl number: ratio of momentum diffusivity to thermal diffusivity"
@parameter stokes  -2//3  "scaling for viscous effect associated with volume change"
@parameter k_μ cp_d/prandtl "thermal conductivity / dynamic viscosity"

const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)
const _nstategrad = 15
const (_ρx, _ρy, _ρz, _ux, _uy, _uz, _vx, _vy, _vz, _wx, _wy, _wz,
       _Tx, _Ty, _Tz) = 1:_nstategrad

struct VanillaAtmosDiscretization{T, dim, polynomialorder, numberofDOFs,
                                  DeviceArray, ntrace, nmoist, DAT3, DASAT3,
                                  GT } <: AD.AbstractAtmosDiscretization
  "grid"
  grid::GT

  "gravitational acceleration (m/s^2)"
  gravity::T

  "viscosity constant"
  viscosity::T

  "MPI send request storage"
  sendreq::Array{MPI.Request, 1}

  "MPI recv request storage"
  recvreq::Array{MPI.Request, 1}

  "host storage for state to be sent"
  host_sendQ::Array{T, 3}

  "host storage for state to be recv'd"
  host_recvQ::Array{T, 3}

  "device storage for state to be sent"
  device_sendQ::DAT3

  "device storage for state to be recv'd"
  device_recvQ::DAT3

  "storage for the grad"
  grad::DASAT3

  "host storage for grad to be sent"
  host_sendgrad::Array{T, 3}

  "host storage for grad to be recv'd"
  host_recvgrad::Array{T, 3}

  "device storage for grad to be sent"
  device_sendgrad::DAT3

  "device storage for grad to be recv'd"
  device_recvgrad::DAT3

  VanillaAtmosDiscretization(grid;
                             # How many tracer variables
                             ntrace=0,
                             # How many moisture variables
                             nmoist=0,
                             kw...) =
  VanillaAtmosDiscretization{ntrace, nmoist}(grid; kw...)

  function VanillaAtmosDiscretization{ntrace, nmoist
                                     }(grid::AbstractGrid{T, dim, N, Np, DA};
                                       # Use gravity?
                                       gravity = true,
                                       # viscosity constant
                                       viscosity = 0
                                      ) where {T, dim, N, Np, DA,
                                               ntrace, nmoist}
    topology = grid.topology

    nnabr = length(topology.nabrtorank)
    sendreq = fill(MPI.REQUEST_NULL, nnabr)
    recvreq = fill(MPI.REQUEST_NULL, nnabr)

    nvar = _nstate + nmoist + ntrace
    host_sendQ = zeros(T, (N+1)^dim, nvar, length(topology.sendelems))
    host_recvQ = zeros(T, (N+1)^dim, nvar, length(topology.ghostelems))
    device_sendQ = DA(host_sendQ)
    device_recvQ = DA(host_recvQ)

    ngrad = _nstategrad + 3*nmoist
    host_sendgrad = zeros(T, (N+1)^dim, ngrad, length(topology.sendelems))
    host_recvgrad = zeros(T, (N+1)^dim, ngrad, length(topology.ghostelems))
    device_sendgrad = DA(host_sendgrad)
    device_recvgrad = DA(host_recvgrad)
    grad = AtmosStateArray{Tuple{Np, ngrad}, T, DA}(topology.mpicomm,
                                                    length(topology.elems),
                                                    realelems=topology.realelems,
                                                    ghostelems=topology.ghostelems,
                                                    sendelems=topology.sendelems,
                                                    nabrtorank=topology.nabrtorank,
                                                    nabrtorecv=topology.nabrtorecv,
                                                    nabrtosend=topology.nabrtosend)

    DAT3 = typeof(device_sendQ)
    GT = typeof(grid)
    DASAT3 = typeof(grad)

    new{T, dim, N, Np, DA, ntrace, nmoist, DAT3, DASAT3, GT
       }(grid, gravity ? grav : 0, viscosity, sendreq, recvreq, host_sendQ,
         host_recvQ, device_sendQ, device_recvQ, grad, host_sendgrad,
         host_recvgrad, device_sendgrad, device_recvgrad)
  end
end

function Base.getproperty(X::VanillaAtmosDiscretization, s::Symbol)
  if s ∈ keys(stateid)
    stateid[s]
  elseif s == :dim
  else
    getfield(X, s)
  end
end

function Base.propertynames(X::VanillaAtmosDiscretization)
  (fieldnames(VanillaAtmosDiscretization)..., keys(stateid)...)
end

function AtmosStateArrays.AtmosStateArray(disc::VanillaAtmosDiscretization{
                                                 T, dim, N, Np, DA, ntrace,
                                                 nmoist}
                                         ) where {T, dim, N, Np, DA, ntrace,
                                                  nmoist}
  topology = disc.grid.topology
  nvar = _nstate + nmoist + ntrace
  AtmosStateArray{Tuple{Np, nvar}, T, DA}(topology.mpicomm,
                                          length(topology.elems),
                                          realelems=topology.realelems,
                                          ghostelems=topology.ghostelems,
                                          sendelems=topology.sendelems,
                                          nabrtorank=topology.nabrtorank,
                                          nabrtorecv=topology.nabrtorecv,
                                          nabrtosend=topology.nabrtosend)
end

function AtmosStateArrays.AtmosStateArray(disc::VanillaAtmosDiscretization{
                                                 T, dim, N, Np, DA, ntrace,
                                                 nmoist}, ic::Function
                                         ) where {T, dim, N, Np, DA, ntrace,
                                                  nmoist}
  Q = AtmosStateArray(disc)

  nvar = _nstate + nmoist + ntrace
  G = disc.grid
  vgeo = G.vgeo

  # FIXME: GPUify me
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q) = host_array ? (vgeo, Q) : (Array(vgeo), Array(Q))
  @inbounds for e = 1:size(Q, 3), i = 1:Np
    x, y, z = h_vgeo[i, G.xid, e], h_vgeo[i, G.yid, e], h_vgeo[i, G.zid, e]
    q0 = ic(x, y, z)

    # Assume that this will be compile time?
    @assert nmoist == length(q0.Qmoist) && ntrace == length(q0.Qtrace)

    h_Q[i, [_ρ, _U, _V, _W, _E]           , e] .= (q0.ρ, q0.U, q0.V, q0.W, q0.E)
    h_Q[i, _nstate .+           (1:nmoist), e] .= q0.Qmoist
    h_Q[i, _nstate .+ nmoist .+ (1:ntrace), e] .= q0.Qtrace
  end
  if !host_array
    Q .= h_Q
  end

  Q
end

AtmosStateArrays.AtmosStateArray(f::Function,
                                 d::VanillaAtmosDiscretization
                                ) = AtmosStateArray(d, f)

function estimatedt(disc::VanillaAtmosDiscretization{T, dim, N, Np, DA},
                    Q::AtmosStateArray) where {T, dim, N, Np, DA}
  @assert T == eltype(Q)
  G = disc.grid
  vgeo = G.vgeo
  # FIXME: GPUify me
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q) = host_array ? (vgeo, Q) : (Array(vgeo), Array(Q))
  estimatedt(Val(dim), Val(N), G, disc.gravity, h_Q, h_vgeo, G.topology.mpicomm)
end

# FIXME: This needs cleaning up
function estimatedt(::Val{dim}, ::Val{N}, G, gravity, Q, vgeo,
                    mpicomm) where {dim, N}

  DFloat = eltype(Q)

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  dt = [floatmax(DFloat)]

  if dim == 2
    @inbounds for e = 1:nelem, n = 1:Np
      ρ, U, V = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e]
      E = Q[n, _E, e]
      y = vgeo[n, G.yid, e]
      P = gdm1*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)

      ξx, ξy, ηx, ηy = vgeo[n, G.ξxid, e], vgeo[n, G.ξyid, e],
                       vgeo[n, G.ηxid, e], vgeo[n, G.ηyid, e]

      loc_dt = 2ρ / max(abs(U * ξx + V * ξy) + ρ * sqrt(gamma_d * P / ρ),
                        abs(U * ηx + V * ηy) + ρ * sqrt(gamma_d * P / ρ))
      dt[1] = min(dt[1], loc_dt)
    end
  end

  if dim == 3
    @inbounds for e = 1:nelem, n = 1:Np
      ρ, U, V, W = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e]
      E = Q[n, _E, e]
      z = vgeo[n, G.zid, e]
      P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

      ξx, ξy, ξz = vgeo[n, G.ξxid, e], vgeo[n, G.ξyid, e], vgeo[n, G.ξzid, e]
      ηx, ηy, ηz = vgeo[n, G.ηxid, e], vgeo[n, G.ηyid, e], vgeo[n, G.ηzid, e]
      ζx, ζy, ζz = vgeo[n, G.ζxid, e], vgeo[n, G.ζyid, e], vgeo[n, G.ζzid, e]

      loc_dt = 2ρ / max(abs(U * ξx + V * ξy + W * ξz) + ρ * sqrt(gamma_d*P/ρ),
                        abs(U * ηx + V * ηy + W * ηz) + ρ * sqrt(gamma_d*P/ρ),
                        abs(U * ζx + V * ζy + W * ζz) + ρ * sqrt(gamma_d*P/ρ))
      dt[1] = min(dt[1], loc_dt)
    end
  end

  MPI.Allreduce(dt[1], MPI.MIN, mpicomm) / N^√2
end

#{{{ rhs!

function AD.getrhsfunction(disc::VanillaAtmosDiscretization)
  (x...) -> rhs!(x..., disc)
end

rhs!(dQ, Q, p::Nothing, t, sd::VanillaAtmosDiscretization) = rhs!(dQ, Q, t, sd)

function rhs!(dQ::AtmosStateArray{S, T}, Q::AtmosStateArray{S, T}, t::T,
              disc::VanillaAtmosDiscretization{T, dim, N, Np, DA, ntrace,
                                               nmoist}
             ) where {S, T, dim, N, Np, DA, ntrace, nmoist}
  grid = disc.grid
  topology = grid.topology

  # FIXME: Shove all these into AtmosStateArrays
  mpicomm = topology.mpicomm
  sendreq = disc.sendreq
  recvreq = disc.recvreq
  host_recvQ = disc.host_recvQ
  host_sendQ = disc.host_sendQ
  device_recvQ = disc.device_recvQ
  device_sendQ = disc.device_sendQ
  host_recvgrad = disc.host_recvgrad
  host_sendgrad = disc.host_sendgrad
  device_recvgrad = disc.device_recvgrad
  device_sendgrad = disc.device_sendgrad
  grad = disc.grad

  sendelems = grid.sendelems

  gravity = disc.gravity

  vgeo = grid.vgeo
  sgeo = grid.sgeo
  Dmat = grid.D
  vmapM = grid.vmapM
  vmapP = grid.vmapP
  elemtobndy = grid.elemtobndy

  DFloat = eltype(Q)
  viscosity::DFloat = disc.viscosity

  nnabr = length(topology.nabrtorank)
  nrealelem = length(topology.realelems)

  @assert DFloat == eltype(Q)
  @assert DFloat == eltype(vgeo)
  @assert DFloat == eltype(sgeo)
  @assert DFloat == eltype(Dmat)
  @assert DFloat == eltype(grad)
  @assert DFloat == eltype(dQ)

  # post MPI receives
  for n = 1:nnabr
    recvreq[n] = MPI.Irecv!((@view host_recvQ[:, :, topology.nabrtorecv[n]]),
                            topology.nabrtorank[n], 777, mpicomm)
  end

  # wait on (prior) MPI sends
  MPI.Waitall!(sendreq)

  # pack data in send buffer
  fillsendbuf!(host_sendQ, device_sendQ, Q, sendelems)

  # post MPI sends
  for n = 1:nnabr
    sendreq[n] = MPI.Isend((@view host_sendQ[:, :, topology.nabrtosend[n]]),
                           topology.nabrtorank[n], 777, mpicomm)
  end

  # volume grad computation
  volumegrad!(Val(dim), Val(N), Val(nmoist), Val(ntrace), grad.Q, Q.Q, vgeo,
              gravity, Dmat, topology.realelems)

  # wait on MPI receives
  MPI.Waitall!(recvreq)

  # copy data to state vectors
  transferrecvbuf!(device_recvQ, host_recvQ, Q, nrealelem)

  # post MPI receives
  for n = 1:nnabr
    recvreq[n] = MPI.Irecv!((@view host_recvgrad[:, :, topology.nabrtorecv[n]]),
                            topology.nabrtorank[n], 777, mpicomm)
  end

  # face grad computation
  facegrad!(Val(dim), Val(N), Val(nmoist), Val(ntrace), grad.Q, Q.Q, vgeo,
            sgeo, gravity, topology.realelems, vmapM, vmapP, elemtobndy)

  # wait on MPI sends
  MPI.Waitall!(sendreq)

  # pack data in send buffer
  fillsendbuf!(host_sendgrad, device_sendgrad, grad, sendelems)

  # post MPI sends
  for n = 1:nnabr
    sendreq[n] = MPI.Isend((@view host_sendgrad[:, :, topology.nabrtosend[n]]),
                           topology.nabrtorank[n], 777, mpicomm)
  end

  # volume RHS computation
  volumerhs!(Val(dim), Val(N), Val(nmoist), Val(ntrace), dQ.Q, Q.Q, grad.Q,
             vgeo, gravity, viscosity, Dmat, topology.realelems)

  # wait on MPI receives
  MPI.Waitall!(recvreq)

  # copy data to state vectors
  transferrecvbuf!(device_recvgrad, host_recvgrad, grad, nrealelem)

  # face RHS computation
  facerhs!(Val(dim), Val(N), Val(nmoist), Val(ntrace), dQ.Q, Q.Q, grad.Q,
           vgeo, sgeo, gravity, viscosity, topology.realelems, vmapM, vmapP,
           elemtobndy)
end
# }}}

# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
       _x, _y, _z = 1:_nvgeo

const _nsgeo = 5
const _nx, _ny, _nz, _sMJ, _vMJI = 1:_nsgeo
# }}}

# {{{ L2 Energy (for all dimensions)
# TODO: Better would be to register this as the norm function for the
# AtmosStateArray, possibly also this should be moved to Grid
function L2norm(Q::AtmosStateArray, spacedisc::VanillaAtmosDiscretization)

  vgeo = spacedisc.grid.vgeo

  # TODO: GPUify
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q) = host_array ? (vgeo, Q) : (Array(vgeo), Array(Q))
  Np = size(Q, 1)
  locnorm2 = knl_L2norm(Val(Np), h_Q, h_vgeo, Q.realelems)
  sqrt(MPI.allreduce([locnorm2], MPI.SUM, spacedisc.grid.topology.mpicomm)[1])
end

function knl_L2norm(::Val{Np}, Q, vgeo, elems) where {Np}
  DFloat = eltype(Q)
  (~, nstate, nelem) = size(Q)

  energy = zero(DFloat)

  @inbounds for e = elems, q = 1:nstate, i = 1:Np
    energy += vgeo[i, _MJ, e] * Q[i, q, e]^2
  end

  energy
end
# }}}

#=
# {{{ L2 Error (for all dimensions)
function AD.L2errornorm(runner::Runner{DeviceArray}, Qexact;
                        host=false, Q = nothing, vgeo = nothing,
                        time = nothing) where DeviceArray
  host || error("Currently requires host configuration")
  state = runner.state
  config = runner.config
  params = runner.params
  cpubackend = DeviceArray == Array
  if vgeo == nothing
    vgeo = cpubackend ? config.vgeo : Array(config.vgeo)
  end
  if Q == nothing
    Q = cpubackend ? state.Q : Array(state.Q)
  end
  if time == nothing
    time = state.time[1]
  end

  dim = params.dim
  N = params.N
  realelems = config.mesh.realelems
  locnorm2 = L2errornorm(Val(dim), Val(N), time, Q, vgeo, realelems, Qexact)
  sqrt(MPI.allreduce([locnorm2], MPI.SUM, config.mpicomm)[1])
end

function L2errornorm(::Val{dim}, ::Val{N}, time, Q, vgeo, elems,
                     Qexact) where
  {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  (~, nstate, nelem) = size(Q)

  errorsq = zero(DFloat)

  @inbounds for e = elems,  i = 1:Np
    x, y, z = vgeo[i, _x, e], vgeo[i, _y, e], vgeo[i, _z, e]
    ρex, Uex, Vex, Wex, Eex = Qexact(time, x, y, z)

    errorsq += vgeo[i, _MJ, e] * (Q[i, _ρ, e] - ρex)^2
    errorsq += vgeo[i, _MJ, e] * (Q[i, _U, e] - Uex)^2
    errorsq += vgeo[i, _MJ, e] * (Q[i, _V, e] - Vex)^2
    errorsq += vgeo[i, _MJ, e] * (Q[i, _W, e] - Wex)^2
    errorsq += vgeo[i, _MJ, e] * (Q[i, _E, e] - Eex)^2
  end

  errorsq
end
# }}}
=#


using Requires

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
  using .CUDAnative.CUDAdrv

  include("VanillaEuler_cuda.jl")
end

include("VanillaAtmosDiscretizations_kernels.jl")

end
