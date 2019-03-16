module AtmosStateArrays
using LinearAlgebra

using MPI

export AtmosStateArray

# TODO: Make MPI Aware
struct AtmosStateArray{S <: Tuple, T, DeviceArray, N,
                       DATN<:AbstractArray{T,N}, Nm1, DAI1} <: AbstractArray{T, N}
  mpicomm::MPI.Comm
  Q::DATN

  realelems::UnitRange{Int64}
  ghostelems::UnitRange{Int64}
  sendelems::DAI1

  sendreq::Array{MPI.Request, 1}
  recvreq::Array{MPI.Request, 1}

  host_sendQ::Array{T, N}
  host_recvQ::Array{T, N}

  nabrtorank::Array{Int64, 1}
  nabrtorecv::Array{UnitRange{Int64}, 1}
  nabrtosend::Array{UnitRange{Int64}, 1}

  device_sendQ::DATN
  device_recvQ::DATN

  # FIXME: Later we should relax this if we compute on the GPU and probably
  # should let this be a more generic type...
  weights::Array{T, Nm1}
  function AtmosStateArray{S, T, DA}(mpicomm, numelem, realelems, ghostelems,
                                     sendelems, nabrtorank, nabrtorecv,
                                     nabrtosend, weights) where {S, T, DA}
    N = length(S.parameters)+1
    numsendelem = length(sendelems)
    numrecvelem = length(ghostelems)
    (Q, device_sendQ, device_recvQ) = try
      (DA{T, N}(undef, S.parameters..., numelem),
       DA{T, N}(undef, S.parameters..., numsendelem),
       DA{T, N}(undef, S.parameters..., numrecvelem))
    catch
      try
        # TODO: Remove me after CUDA upgrade...
        (DA{T, N}(S.parameters..., numelem),
         DA{T, N}(S.parameters..., numsendelem),
         DA{T, N}(S.parameters..., numrecvelem))
      catch
        error("AtmosStateArray:Cannot construct array")
      end
    end

    host_sendQ = zeros(T, S.parameters..., numsendelem)
    host_recvQ = zeros(T, S.parameters..., numsendelem)

    fill!(Q, 0)
    fill!(device_sendQ, 0)
    fill!(device_sendQ, 0)

    nnabr = length(nabrtorank)
    sendreq = fill(MPI.REQUEST_NULL, nnabr)
    recvreq = fill(MPI.REQUEST_NULL, nnabr)

    sendelems = typeof(sendelems) <: DA ? sendelems : DA(sendelems)
    DAI1 = typeof(sendelems)
    new{S, T, DA, N, typeof(Q), N-1, DAI1}(mpicomm, Q, realelems, ghostelems,
                                           sendelems, sendreq, recvreq,
                                           host_sendQ, host_recvQ, nabrtorank,
                                           nabrtorecv, nabrtosend,
                                           device_sendQ, device_recvQ, weights)
  end


end
function AtmosStateArray{S, T, DA}(mpicomm, numelem;
                                   realelems=1:numelem,
                                   ghostelems=numelem:numelem-1,
                                   sendelems=1:0,
                                   nabrtorank=Array{Int64}(undef, 0),
                                   nabrtorecv=Array{UnitRange{Int64}}(undef, 0),
                                   nabrtosend=Array{UnitRange{Int64}}(undef, 0),
                                   weights=nothing,
                                  ) where {S<:Tuple, T, DA}

  N = length(S.parameters)+1
  if weights == nothing
    weights = Array{T}(undef, ntuple(j->0, N-1))
  elseif !(typeof(weights) <: Array)
    weights = Array(weights)
  end
  AtmosStateArray{S, T, DA}(mpicomm, numelem, realelems, ghostelems,
                            sendelems, nabrtorank, nabrtorecv,
                            nabrtosend, weights)
end
# FIXME: should general cases should be handled?
function Base.similar(Q::AtmosStateArray{S, T, DA}) where {S, T, DA}
  AtmosStateArray{S, T, DA}(Q.mpicomm, size(Q.Q)[end], Q.realelems, Q.ghostelems,
                            Q.sendelems, Q.nabrtorank, Q.nabrtorecv,
                            Q.nabrtosend, Q.weights)
end

# FIXME: Only show real size
Base.size(Q::AtmosStateArray, x...;kw...) = size(Q.Q, x...;kw...)

# FIXME: Only let get index access real elements?
Base.getindex(Q::AtmosStateArray, x...;kw...) = getindex(Q.Q, x...;kw...)

# FIXME: Only let set index access real elements?
Base.setindex!(Q::AtmosStateArray, x...;kw...) = setindex!(Q.Q, x...;kw...)

Base.eltype(Q::AtmosStateArray, x...;kw...) = eltype(Q.Q, x...;kw...)

Base.Array(Q::AtmosStateArray) = Array(Q.Q)

function Base.copyto!(dst::AtmosStateArray, src::Array)
  copyto!(dst.Q, src)
  dst
end
Base.copyto!(dst::Array, src::AtmosStateArray) = copyto!(dst, src.Q)
function Base.copyto!(dst::AtmosStateArray, src::AtmosStateArray)
  copyto!(dst.Q, src.Q)
  dst
end

function postrecvs!(Q::AtmosStateArray)
  nnabr = length(Q.nabrtorank)
  for n = 1:nnabr
    # If this fails we haven't waited on previous recv!
    @assert Q.recvreq[n].buffer == nothing

    Q.recvreq[n] = MPI.Irecv!((@view Q.host_recvQ[:, :, Q.nabrtorecv[n]]),
                              Q.nabrtorank[n], 888, Q.mpicomm)
  end
end

function startexchange!(Q::AtmosStateArray; dorecvs=true)
  dorecvs && postrecvs!(Q)

  # wait on (prior) MPI sends
  MPI.Waitall!(Q.sendreq)

  # pack data in send buffer
  fillsendbuf!(Q.host_sendQ, Q.device_sendQ, Q.Q, Q.sendelems)

  # post MPI sends
  nnabr = length(Q.nabrtorank)
  for n = 1:nnabr
    Q.sendreq[n] = MPI.Isend((@view Q.host_sendQ[:, :, Q.nabrtosend[n]]),
                           Q.nabrtorank[n], 888, Q.mpicomm)
  end
end

function finishexchange!(Q::AtmosStateArray)
  # wait on MPI receives
  MPI.Waitall!(Q.recvreq)

  # copy data to state vectors
  transferrecvbuf!(Q.device_recvQ, Q.host_recvQ, Q, length(Q.realelems))
end

# {{{ MPI Buffer handling
fillsendbuf!(h, d, b::AtmosStateArray, e) = fillsendbuf!(h, d, b.Q, e)
transferrecvbuf!(h, d, b::AtmosStateArray, e) = transferrecvbuf!(h, d, b.Q, e)

function fillsendbuf!(host_sendbuf, device_sendbuf::Array, buf::Array, sendelems)
  host_sendbuf[:, :, :] .= buf[:, :, sendelems]
end

function transferrecvbuf!(device_recvbuf::Array, host_recvbuf, buf::Array,
                          nrealelem)
  buf[:, :, nrealelem+1:end] .= host_recvbuf[:, :, :]
end
# }}}

# {{{ L2 Energy (for all dimensions)
function LinearAlgebra.norm(Q::AtmosStateArray; p::Real=2)

  @assert p == 2

  host_array = Array ∈ typeof(Q).parameters
  h_Q = host_array ? Q : Array(Q)
  Np = size(Q, 1)

  if isempty(Q.weights)
    locnorm2 = knl_norm2(Val(Np), h_Q, Q.realelems)
  else
    locnorm2 = knl_L2norm(Val(Np), h_Q, Q.weights, Q.realelems)
  end

  sqrt(MPI.allreduce([locnorm2], MPI.SUM, Q.mpicomm)[1])
end

function knl_norm2(::Val{Np}, Q, elems) where {Np}
  DFloat = eltype(Q)
  (_, nstate, nelem) = size(Q)

  energy = zero(DFloat)

  @inbounds for e = elems, q = 1:nstate, i = 1:Np
    energy += Q[i, q, e]^2
  end

  energy
end

function knl_L2norm(::Val{Np}, Q, weights, elems) where {Np}
  DFloat = eltype(Q)
  (_, nstate, nelem) = size(Q)

  energy = zero(DFloat)

  @inbounds for e = elems, q = 1:nstate, i = 1:Np
    energy += weights[i, e] * Q[i, q, e]^2
  end

  energy
end

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
  (_, nstate, nelem) = size(Q)

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

# }}}

using Requires

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  using .CuArrays
  using .CuArrays.CUDAnative
  using .CuArrays.CUDAnative.CUDAdrv

  include("AtmosStateArrays_cuda.jl")
end

end
