module VanillaEuler
include("vtk.jl")

using Requires
@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
  using .CUDAnative.CUDAdrv

  include("VanillaEuler_cuda.jl")
end

using ..CLIMAAtmosDycore
AD = CLIMAAtmosDycore

using Canary
using MPI
using ParametersType
using PlanetParameters: cp_d, cv_d, R_d, grav
@parameter gamma_d cp_d/cv_d "Heat capcity ratio of dry air"
@parameter gdm1 R_d/cv_d "(equivalent to gamma_d-1)"

using Base: @kwdef

# {{{ constants

# note the order of the fields below is also assumed in the code.
const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρ = _ρ, U = _U, V = _V, W = _W, E = _E)

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

"""
    Parameters

Data structure containing the parameters for the vanilla DG discretization of
the compressible Euler equations

To get information about parameters do for example
`?Parameters.DFloat`

!!! note
Would be nice for the the docs to be recursively generated.
See Julia github issue [#25167](https://github.com/JuliaLang/julia/issues/25167)

"""
# {{{ Parameters
@kwdef struct Parameters # <: AD.AbstractSpaceParameter
  """
  Compute data type

  default: `Float64`
  """
  DFloat::Type = Float64

  """
  Device array type

  if 'Array' the cpu implemtation will be used

  if 'CuArray' the cuda implemtation will be used

  default: `Array`
  """
  DeviceArray::Type = Array

  """
  Function with arguments `(part, numparts)`, which returns initial partition
  `part` of the mesh of with number of partitions `numparts`

  no default value
  """
  meshgenerator::Function

  """
  Function to warp the coordinate points after mesh generation

  Syntax TBD

  default: `(x...)->identity(x)`
  """
  meshwarp::Function = (x...)->identity(x)

  """
  Intial time of the simulation

  default: 0
  """
  initialtime::AbstractFloat = 0

  """
  Number of spatial dimensiona

  default: `3`
  """
  dim::Int = 3

  """
  Boolean specifying whether or not to use gravity

  default: `true`
  """
  gravity::Bool = true

  """
  Polynomial order for discontinuous Galerkin method

  no default value
  """
  N::Int

  """
  number of moist variables

  default: 0
  """
  nmoist::Int = 0

  """
  number of tracer variables

  default: 0
  """
  ntrace::Int = 0
end
# }}}

"""
    Configuration

Data structure containing the configuration data for the vanilla DG
discretization of the compressible Euler equations
"""
# {{{ Configuration
struct Configuration{DeviceArray, HostArray} #<: AD.AbstractSpaceConfiguration
  "mpi communicator use for spatial discretization are using"
  mpicomm
  "mesh data structure from Canary"
  mesh
  "volume metric terms"
  vgeo::DeviceArray
  "surface metric terms"
  sgeo::DeviceArray
  "gravitational acceleration (m/s^2)"
  gravity
  "element to boundary condition map"
  elemtobndy::DeviceArray
  "volume DOF to element minus side map"
  vmapM::DeviceArray
  "volume DOF to element plus side map"
  vmapP::DeviceArray
  "list of elements that need to be communicated (in neighbors order)"
  sendelems::DeviceArray
  "MPI send request storage"
  sendreq::HostArray
  "MPI recv request storage"
  recvreq::HostArray
  "host storage for state to be sent"
  host_sendQ::HostArray
  "host storage for state to be recv'd"
  host_recvQ::HostArray
  "device storage for state to be sent"
  device_sendQ::DeviceArray
  "device storage for state to be recv'd"
  device_recvQ::DeviceArray
  "1-D derivative operator on the device"
  D::DeviceArray

  function Configuration(params::Parameters, mpicomm)
    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)
    N = params.N
    dim = params.dim
    DFloat = params.DFloat

    mesh = params.meshgenerator(mpirank+1, mpisize)

    mpirank == 0 && @debug "partiting mesh..."
    mesh = partition(mpicomm, mesh...)

    # Connect the mesh in parallel
    mpirank == 0 && @debug "connecting mesh..."
    mesh = connectmesh(mpicomm, mesh...)

    # Get the vmaps
    mpirank == 0 && @debug "computing mappings..."
    (vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface,
                              mesh.elemtoordr)

    # Create 1-D operators
    (ξ, ω) = lglpoints(DFloat, N)
    D = spectralderivative(ξ)

    # Compute the geometry
    mpirank == 0 && @debug "computing metrics..."
    (vgeo, sgeo) = computegeometry(Val(dim), mesh, D, ξ, ω,
                                   params.meshwarp, vmapM)
    gravity::DFloat = (params.gravity) ? grav : 0
    (nface, nelem) = size(mesh.elemtoelem)

    mpirank == 0 && @debug "create RHS storage..."

    mpirank == 0 && @debug "create send/recv request storage..."
    nnabr = length(mesh.nabrtorank)
    sendreq = fill(MPI.REQUEST_NULL, nnabr)
    recvreq = fill(MPI.REQUEST_NULL, nnabr)

    mpirank == 0 && @debug "create send/recv storage..."
    nvar = _nstate + params.nmoist + params.ntrace
    sendQ = zeros(DFloat, (N+1)^dim, nvar, length(mesh.sendelems))
    recvQ = zeros(DFloat, (N+1)^dim, nvar, length(mesh.ghostelems))

    mpirank == 0 && @debug "create configuration struct..."
    HostArray = Array
    DeviceArray = params.DeviceArray
    # FIXME: Handle better for GPU?
    new{DeviceArray, HostArray}(mpicomm, mesh, DeviceArray(vgeo),
                                DeviceArray(sgeo), gravity,
                                DeviceArray(mesh.elemtobndy),
                                DeviceArray(vmapM), DeviceArray(vmapP),
                                DeviceArray(mesh.sendelems), sendreq, recvreq,
                                sendQ, recvQ, DeviceArray(sendQ),
                                DeviceArray(recvQ), DeviceArray(D))
  end
end
# }}}

"""
    State

Data structure containing the state data for the vanilla DG discretization of
the compressible Euler equations
"""
# {{{ State
struct State{DeviceArray} #<: AD.AbstractSpaceState
  time
  Q::DeviceArray
  function State(config::Configuration{DeviceArray, HostArray},
                 params::Parameters) where {DeviceArray, HostArray}
    nvar = _nstate + params.nmoist + params.ntrace
    Q = similar(config.vgeo, (size(config.vgeo,1), nvar, size(config.vgeo,3)))
    # Shove into array so we can leave the type immutable
    # (is this worthwhile?)
    time = [params.initialtime]
    new{DeviceArray}(time, Q)
  end
end
# }}}

"""
    Runner

Data structure containing the runner for the vanilla DG discretization of
the compressible Euler equations

"""
# {{{ Runner
struct Runner{DeviceArray<:AbstractArray} <: AD.AbstractSpaceRunner
  params::Parameters
  config::Configuration
  state::State
  function Runner(mpicomm; args...)
    params = Parameters(;args...)
    config = Configuration(params, mpicomm)
    state = State(config, params)
    new{params.DeviceArray}(params, config, state)
  end
end
AD.createrunner(::Val{:VanillaEuler}, m; a...) = Runner(m; a...)
# }}}

# {{{ show
function Base.show(io::IO, runner::Runner)
  eng = AD.L2solutionnorm(runner; host=true)
  print(io, "VanillaEuler with norm2(Q) = ", eng, " at time = ",
        runner[:time])
end

function Base.show(io::IO, ::MIME"text/plain",
                   runner::Runner{DeviceArray}) where DeviceArray
  state = runner.state
  params = runner.params
  config = runner.config
  println(io, "VanillaEuler with:")
  DFloat = eltype(state.Q)
  eng = AD.L2solutionnorm(runner; host=true)
  println(io, "   DeviceArray = ", DeviceArray)
  println(io, "   DFloat      = ", DFloat)
  println(io, "   norm2(Q)    = ", eng)
  println(io, "   time        = ", runner[:time])
  println(io, "   N           = ", params.N)
  println(io, "   dim         = ", params.dim)
  println(io, "   mpisize     = ", MPI.Comm_size(config.mpicomm))
end
# }}}

Base.getindex(r::Runner, s) = r[Symbol(s)]
function Base.getindex(r::Runner, s::Symbol)
  s == :time && return r.state.time[1]
  s == :Q && return r.state.Q
  s == :mesh && return r.config.mesh
  s == :stateid && return stateid
  s == :moistid && return _nstate .+ (1:r.params.nmoist)
  s == :traceid && return _nstate .+ r.params.nmoist .+ (1:r.params.ntrace)
  error("""
        getindex for the $(typeof(r)) supports:
        `:time`    => gets the runners time
        `:Q`       => gets the runners state Q
        `:mesh`    => gets the runners mesh
        `:hostQ`   => not implemented yet
        `:stateid` => Euler state storage order
        `:moistid` => moist state storage order
        `:traceid` => trace state storage order
        """)
end
Base.setindex!(r::Runner, v, s) = Base.setindex!(r, v, Symbol(s))
function Base.setindex!(r::Runner, v,  s::Symbol)
  s == :time && return r.state.time[1] = v
  error("""
        setindex! for the $(typeof(r)) supports:
        `:time` => sets the runners time
        """)
end

# {{{ initstate!
function AD.initstate!(runner::Runner{DeviceArray}, ic::Function;
                       host=false) where DeviceArray

  host || error("Currently requires host configuration")

  # Pull out the config and state
  params::Parameters = runner.params
  config::Configuration = runner.config
  state::State = runner.state

  # Get the number of elements
  cpubackend = DeviceArray == Array
  vgeo = cpubackend ? config.vgeo : Array(config.vgeo)
  Q = cpubackend ? state.Q : Array(state.Q)

  nvar = _nstate + params.nmoist + params.ntrace
  @inbounds for e = 1:size(Q, 3), i = 1:size(Q, 1)
    x, y, z = vgeo[i, _x, e], vgeo[i, _y, e], vgeo[i, _z, e]
    Q0 = ic(x, y, z)
    for n = 1:nvar
      Q[i, n, e] = Q0[n]
    end
  end
  if !cpubackend
    state.Q .= Q
  end
end
#}}}

# {{{ compute geometry
function computegeometry(::Val{dim}, mesh, D, ξ, ω, meshwarp, vmapM) where dim
  # Compute metric terms
  Nq = size(D, 1)
  DFloat = eltype(D)

  (nface, nelem) = size(mesh.elemtoelem)

  # crd = creategrid(Val(dim), mesh.elemtocoord, ξ)

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

# {{{ cfl
function AD.estimatedt(runner::Runner{DeviceArray};
                       host=false) where {DeviceArray}
  host || error("Currently requires host configuration")
  state = runner.state
  config = runner.config
  params = runner.params
  cpubackend = DeviceArray == Array
  vgeo = cpubackend ? config.vgeo : Array(config.vgeo)
  Q = cpubackend ? state.Q : Array(state.Q)
  estimatedt(Val(params.dim), Val(params.N), vgeo, config.gravity, Q,
             config.mpicomm)
end

function estimatedt(::Val{dim}, ::Val{N}, vgeo, gravity, Q,
                    mpicomm) where {dim, N}
  DFloat = eltype(Q)

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  dt = [floatmax(DFloat)]

  if dim == 2
    @inbounds for e = 1:nelem, n = 1:Np
      ρ, U, V = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e]
      E = Q[n, _E, e]
      y = vgeo[n, _y, e]
      P = gdm1*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)

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
      P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

      ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
      ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
      ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]

      loc_dt = 2ρ / max(abs(U * ξx + V * ξy + W * ξz) + ρ * sqrt(gamma_d*P/ρ),
                        abs(U * ηx + V * ηy + W * ηz) + ρ * sqrt(gamma_d*P/ρ),
                        abs(U * ζx + V * ζy + W * ζz) + ρ * sqrt(gamma_d*P/ρ))
      dt[1] = min(dt[1], loc_dt)
    end
  end

  MPI.Allreduce(dt[1], MPI.MIN, mpicomm) / N^√2
end
#}}}

# {{{ writevtk
function AD.writevtk(runner::Runner{DeviceArray}, prefix;
                     Q = nothing, vgeo = nothing) where DeviceArray
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

  Nq  = params.N+1
  dim = params.dim

  nelem = size(Q)[end]
  X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->Nq,dim)...,
                        nelem), dim)
  ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->Nq, dim)..., nelem)
  U = reshape((@view Q[:, _U, :]), ntuple(j->Nq, dim)..., nelem)
  V = reshape((@view Q[:, _V, :]), ntuple(j->Nq, dim)..., nelem)
  W = reshape((@view Q[:, _W, :]), ntuple(j->Nq, dim)..., nelem)
  E = reshape((@view Q[:, _E, :]), ntuple(j->Nq, dim)..., nelem)
  writemesh(prefix, X...;
            fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
            realelems=config.mesh.realelems)
end
# }}}

# {{{ L2 Energy (for all dimensions)
function AD.L2solutionnorm(runner::Runner{DeviceArray};
                           host=false, Q = nothing, vgeo = nothing
                          ) where DeviceArray
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

  dim = params.dim
  N = params.N
  realelems = config.mesh.realelems
  locnorm2 = L2solutionnorm(Val(dim), Val(N), Q, vgeo, realelems)
  sqrt(MPI.allreduce([locnorm2], MPI.SUM, config.mpicomm)[1])
end

function L2solutionnorm(::Val{dim}, ::Val{N}, Q, vgeo, elems) where {dim, N}
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

# {{{ RHS function
function AD.rhs!(rhs::DeviceArray,
                 runner::Runner{DeviceArray}) where DeviceArray
  state = runner.state
  config = runner.config
  params = runner.params
  mesh = config.mesh
  mpicomm = config.mpicomm
  sendreq = config.sendreq
  recvreq = config.recvreq
  host_recvQ = config.host_recvQ
  host_sendQ = config.host_sendQ
  device_recvQ = config.device_recvQ
  device_sendQ = config.device_sendQ
  sendelems = config.sendelems

  vgeo = config.vgeo
  sgeo = config.sgeo
  Dmat = config.D
  vmapM = config.vmapM
  vmapP = config.vmapP
  gravity = config.gravity
  elemtobndy = config.elemtobndy

  Q = state.Q

  N   = params.N
  dim = params.dim
  ntrace = params.ntrace
  nmoist = params.nmoist

  nnabr = length(mesh.nabrtorank)
  nrealelem = length(mesh.realelems)

  # post MPI receives
  for n = 1:nnabr
    recvreq[n] = MPI.Irecv!((@view host_recvQ[:, :, mesh.nabrtorecv[n]]),
                            mesh.nabrtorank[n], 777, mpicomm)
  end

  # wait on (prior) MPI sends
  MPI.Waitall!(sendreq)

  # pack data in send buffer
  fillsendQ!(Val(dim), Val(N), host_sendQ, device_sendQ, Q, sendelems)

  # post MPI sends
  for n = 1:nnabr
    sendreq[n] = MPI.Isend((@view host_sendQ[:, :, mesh.nabrtosend[n]]),
                           mesh.nabrtorank[n], 777, mpicomm)
  end

  # volume RHS computation
  volumerhs!(Val(dim), Val(N), Val(nmoist), Val(ntrace), rhs, Q, vgeo, gravity,
             Dmat, mesh.realelems)

  # wait on MPI receives
  MPI.Waitall!(recvreq)

  # copy data to state vectors
  transferrecvQ!(Val(dim), Val(N), device_recvQ, host_recvQ, Q, nrealelem)

  # face RHS computation
  facerhs!(Val(dim), Val(N), rhs, Q, vgeo, sgeo, gravity, mesh.realelems,
           vmapM, vmapP, elemtobndy)
end
# }}}

# {{{ MPI Buffer handling
function fillsendQ!(::Val{dim}, ::Val{N}, host_sendQ,
                            device_sendQ::Array, Q, sendelems) where {dim, N}
  host_sendQ[:, :, :] .= Q[:, :, sendelems]
end

function transferrecvQ!(::Val{dim}, ::Val{N}, device_recvQ::Array,
                                host_recvQ, Q, nrealelem) where {dim, N}
  Q[:, :, nrealelem+1:end] .= host_recvQ[:, :, :]
end
# }}}

# {{{ Volume RHS for 2-D
function volumerhs!(::Val{2}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                    rhs::Array, Q, vgeo, gravity, D,
                    elems) where {N, nmoist, ntrace}
  nvar = _nstate + nmoist + ntrace

  DFloat = eltype(Q)

  Nq = N + 1

  nelem = size(Q)[end]

  Q = reshape(Q, Nq, Nq, nvar, nelem)
  rhs = reshape(rhs, Nq, Nq, nvar, nelem)
  vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

  s_F = Array{DFloat}(undef, Nq, Nq, _nstate)
  s_G = Array{DFloat}(undef, Nq, Nq, _nstate)
  MJI = Array{DFloat}(undef, Nq, Nq, _nstate)

  @inbounds for e in elems
    for j = 1:Nq, i = 1:Nq
      MJ = vgeo[i, j, _MJ, e]
      ξx, ξy = vgeo[i,j,_ξx,e], vgeo[i,j,_ξy,e]
      ηx, ηy = vgeo[i,j,_ηx,e], vgeo[i,j,_ηy,e]
      y = vgeo[i,j,_y,e]

      U, V = Q[i, j, _U, e], Q[i, j, _V, e]
      ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]

      P = gdm1*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)

      ρinv = 1 / ρ
      fluxρ_x = U
      fluxU_x = ρinv * U * U + P
      fluxV_x = ρinv * U * V
      fluxE_x = ρinv * U * (E + P)

      fluxρ_y = V
      fluxU_y = ρinv * V * U
      fluxV_y = ρinv * V * V + P
      fluxE_y = ρinv * V * (E + P)

      s_F[i, j, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y)
      s_F[i, j, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y)
      s_F[i, j, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y)
      s_F[i, j, _W] = 0
      s_F[i, j, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y)

      s_G[i, j, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y)
      s_G[i, j, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y)
      s_G[i, j, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y)
      s_G[i, j, _W] = 0
      s_G[i, j, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y)

      # buoyancy term
      rhs[i, j, _V, e] -= ρ * gravity
    end

    # loop of ξ-grid lines
    for s = 1:_nstate, j = 1:Nq, i = 1:Nq
      MJI = vgeo[i, j, _MJI, e]
      for n = 1:Nq
        rhs[i, j, s, e] += MJI * D[n, i] * s_F[n, j, s]
      end
    end
    # loop of η-grid lines
    for s = 1:_nstate, j = 1:Nq, i = 1:Nq
      MJI = vgeo[i, j, _MJI, e]
      for n = 1:Nq
        rhs[i, j, s, e] += MJI * D[n, j] * s_G[i, n, s]
      end
    end
  end
end
# }}}

# {{{ Volume RHS for 3-D
function volumerhs!(::Val{3}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                    rhs::Array, Q, vgeo, gravity, D,
                    elems) where {N, nmoist, ntrace}
  DFloat = eltype(Q)

  nvar = _nstate + nmoist + ntrace

  Nq = N + 1

  nelem = size(Q)[end]

  Q = reshape(Q, Nq, Nq, Nq, nvar, nelem)
  rhs = reshape(rhs, Nq, Nq, Nq, nvar, nelem)
  vgeo = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)

  s_F = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
  s_G = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
  s_H = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)

  @inbounds for e in elems
    for k = 1:Nq, j = 1:Nq, i = 1:Nq
      MJ = vgeo[i, j, k, _MJ, e]
      MJI = vgeo[i, j, k, _MJI, e]
      ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
      ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
      ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
      z = vgeo[i,j,k,_z,e]

      U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
      ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

      P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

      ρinv = 1 / ρ
      fluxρ_x = U
      fluxU_x = ρinv * U * U  + P
      fluxV_x = ρinv * U * V
      fluxW_x = ρinv * U * W
      fluxE_x = ρinv * U * (E + P)

      fluxρ_y = V
      fluxU_y = ρinv * V * U
      fluxV_y = ρinv * V * V + P
      fluxW_y = ρinv * V * W
      fluxE_y = ρinv * V * (E + P)

      fluxρ_z = W
      fluxU_z = ρinv * W * U
      fluxV_z = ρinv * W * V
      fluxW_z = ρinv * W * W + P
      fluxE_z = ρinv * W * (E + P)

      s_F[i, j, k, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
      s_F[i, j, k, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
      s_F[i, j, k, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
      s_F[i, j, k, _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
      s_F[i, j, k, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

      s_G[i, j, k, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
      s_G[i, j, k, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
      s_G[i, j, k, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
      s_G[i, j, k, _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
      s_G[i, j, k, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

      s_H[i, j, k, _ρ] = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
      s_H[i, j, k, _U] = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
      s_H[i, j, k, _V] = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
      s_H[i, j, k, _W] = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
      s_H[i, j, k, _E] = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

      # buoyancy term
      rhs[i, j, k, _W, e] -= ρ * gravity
    end

    # loop of ξ-grid lines
    for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq
      MJI = vgeo[i, j, k, _MJI, e]
      for n = 1:Nq
        rhs[i, j, k, s, e] += MJI * D[n, i] * s_F[n, j, k, s]
      end
    end
    # loop of η-grid lines
    for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq
      MJI = vgeo[i, j, k, _MJI, e]
      for n = 1:Nq
        rhs[i, j, k, s, e] += MJI * D[n, j] * s_G[i, n, k, s]
      end
    end
    # loop of ζ-grid lines
    for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq
      MJI = vgeo[i, j, k, _MJI, e]
      for n = 1:Nq
        rhs[i, j, k, s, e] += MJI * D[n, k] * s_H[i, j, n, s]
      end
    end
  end
end
# }}}

# {{{ Face RHS for 2-D
function facerhs!(::Val{2}, ::Val{N}, rhs::Array, Q, vgeo, sgeo, gravity,
                  elems, vmapM, vmapP, elemtobndy) where N
  DFloat = eltype(Q)

  Np = (N+1)^2
  Nfp = N+1
  nface = 4

  @inbounds for e in elems
    for f = 1:nface
      for n = 1:Nfp
        nxM, nyM = sgeo[_nx, n, f, e], sgeo[_ny, n, f, e]
        sMJ, vMJI = sgeo[_sMJ, n, f, e], sgeo[_vMJI, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        ρM = Q[vidM, _ρ, eM]
        UM = Q[vidM, _U, eM]
        VM = Q[vidM, _V, eM]
        EM = Q[vidM, _E, eM]
        yM = vgeo[vidM, _y, eM]

        bc = elemtobndy[f, e]
        PM = gdm1*(EM - (UM^2 + VM^2)/(2*ρM) - ρM*gravity*yM)
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          EP = Q[vidP, _E, eP]
          yP = vgeo[vidP, _y, eP]
          PP = gdm1*(EP - (UP^2 + VP^2)/(2*ρP) - ρP*gravity*yP)
        elseif bc == 1
          UnM = nxM * UM + nyM * VM
          UP = UM - 2 * UnM * nxM
          VP = VM - 2 * UnM * nyM
          ρP = ρM
          EP = EM
          PP = PM
        else
          error("Invalid boundary conditions $bc on face $f of element $e")
        end

        ρMinv = 1 / ρM
        fluxρM_x = UM
        fluxUM_x = ρMinv * UM * UM + PM
        fluxVM_x = ρMinv * UM * VM
        fluxEM_x = ρMinv * UM * (EM + PM)

        fluxρM_y = VM
        fluxUM_y = ρMinv * VM * UM
        fluxVM_y = ρMinv * VM * VM + PM
        fluxEM_y = ρMinv * VM * (EM + PM)

        ρPinv = 1 / ρP
        fluxρP_x = UP
        fluxUP_x = ρPinv * UP * UP + PP
        fluxVP_x = ρPinv * UP * VP
        fluxEP_x = ρPinv * UP * (EP + PP)

        fluxρP_y = VP
        fluxUP_y = ρPinv * VP * UP
        fluxVP_y = ρPinv * VP * VP + PP
        fluxEP_y = ρPinv * VP * (EP + PP)

        λM = ρMinv * abs(nxM * UM + nyM * VM) + sqrt(ρMinv * gamma_d * PM)
        λP = ρPinv * abs(nxM * UP + nyM * VP) + sqrt(ρPinv * gamma_d * PP)
        λ  =  max(λM, λP)

        #Compute Numerical Flux and Update
        fluxρS = (nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y) +
                  - λ * (ρP - ρM)) / 2
        fluxUS = (nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) +
                  - λ * (UP - UM)) / 2
        fluxVS = (nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) +
                  - λ * (VP - VM)) / 2
        fluxES = (nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y) +
                  - λ * (EP - EM)) / 2


        #Update RHS
        rhs[vidM, _ρ, eM] -= vMJI * sMJ * fluxρS
        rhs[vidM, _U, eM] -= vMJI * sMJ * fluxUS
        rhs[vidM, _V, eM] -= vMJI * sMJ * fluxVS
        rhs[vidM, _E, eM] -= vMJI * sMJ * fluxES
      end
    end
  end
end
# }}}

# {{{ Face RHS for 3-D
function facerhs!(::Val{3}, ::Val{N}, rhs::Array, Q, vgeo, sgeo, gravity,
                  elems, vmapM, vmapP, elemtobndy) where N
  DFloat = eltype(Q)

  Np = (N+1)^3
  Nfp = (N+1)^2
  nface = 6

  @inbounds for e in elems
    for f = 1:nface
      for n = 1:Nfp
        (nxM, nyM, nzM, sMJ, vMJI) = sgeo[:, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        ρM = Q[vidM, _ρ, eM]
        UM = Q[vidM, _U, eM]
        VM = Q[vidM, _V, eM]
        WM = Q[vidM, _W, eM]
        EM = Q[vidM, _E, eM]
        zM = vgeo[vidM, _z, eM]

        bc = elemtobndy[f, e]
        PM = gdm1*(EM - (UM^2 + VM^2 + WM^2)/(2*ρM) - ρM*gravity*zM)
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          WP = Q[vidP, _W, eP]
          EP = Q[vidP, _E, eP]
          zP = vgeo[vidP, _z, eP]
          PP = gdm1*(EP - (UP^2 + VP^2 + WP^2)/(2*ρP) - ρP*gravity*zP)
        elseif bc == 1
          UnM = nxM * UM + nyM * VM + nzM * WM
          UP = UM - 2 * UnM * nxM
          VP = VM - 2 * UnM * nyM
          WP = WM - 2 * UnM * nzM
          ρP = ρM
          EP = EM
          PP = PM
        else
          error("Invalid boundary conditions $bc on face $f of element $e")
        end

        ρMinv = 1 / ρM
        fluxρM_x = UM
        fluxUM_x = ρMinv * UM * UM + PM
        fluxVM_x = ρMinv * UM * VM
        fluxWM_x = ρMinv * UM * WM
        fluxEM_x = ρMinv * UM * (EM + PM)

        fluxρM_y = VM
        fluxUM_y = ρMinv * VM * UM
        fluxVM_y = ρMinv * VM * VM + PM
        fluxWM_y = ρMinv * VM * WM
        fluxEM_y = ρMinv * VM * (EM + PM)

        fluxρM_z = WM
        fluxUM_z = ρMinv * WM * UM
        fluxVM_z = ρMinv * WM * VM
        fluxWM_z = ρMinv * WM * WM + PM
        fluxEM_z = ρMinv * WM * (EM + PM)

        ρPinv = 1 / ρP
        fluxρP_x = UP
        fluxUP_x = ρPinv * UP * UP + PP
        fluxVP_x = ρPinv * UP * VP
        fluxWP_x = ρPinv * UP * WP
        fluxEP_x = ρPinv * UP * (EP + PP)

        fluxρP_y = VP
        fluxUP_y = ρPinv * VP * UP
        fluxVP_y = ρPinv * VP * VP + PP
        fluxWP_y = ρPinv * VP * WP
        fluxEP_y = ρPinv * VP * (EP + PP)

        fluxρP_z = WP
        fluxUP_z = ρPinv * WP * UP
        fluxVP_z = ρPinv * WP * VP
        fluxWP_z = ρPinv * WP * WP + PP
        fluxEP_z = ρPinv * WP * (EP + PP)

        λM = ρMinv * abs(nxM * UM + nyM * VM + nzM * WM) + sqrt(ρMinv * gamma_d * PM)
        λP = ρPinv * abs(nxM * UP + nyM * VP + nzM * WP) + sqrt(ρPinv * gamma_d * PP)
        λ  =  max(λM, λP)

        #Compute Numerical Flux and Update
        fluxρS = (nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y) +
                  nzM * (fluxρM_z + fluxρP_z) - λ * (ρP - ρM)) / 2
        fluxUS = (nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) +
                  nzM * (fluxUM_z + fluxUP_z) - λ * (UP - UM)) / 2
        fluxVS = (nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) +
                  nzM * (fluxVM_z + fluxVP_z) - λ * (VP - VM)) / 2
        fluxWS = (nxM * (fluxWM_x + fluxWP_x) + nyM * (fluxWM_y + fluxWP_y) +
                  nzM * (fluxWM_z + fluxWP_z) - λ * (WP - WM)) / 2
        fluxES = (nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y) +
                  nzM * (fluxEM_z + fluxEP_z) - λ * (EP - EM)) / 2


        #Update RHS
        rhs[vidM, _ρ, eM] -= vMJI * sMJ * fluxρS
        rhs[vidM, _U, eM] -= vMJI * sMJ * fluxUS
        rhs[vidM, _V, eM] -= vMJI * sMJ * fluxVS
        rhs[vidM, _W, eM] -= vMJI * sMJ * fluxWS
        rhs[vidM, _E, eM] -= vMJI * sMJ * fluxES
      end
    end
  end
end
# }}}

end
