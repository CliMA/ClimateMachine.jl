module MPIStateArrays
using LinearAlgebra
using DoubleFloats
using LazyArrays
using StaticArrays

using MPI

using Base.Broadcast: Broadcasted, BroadcastStyle, ArrayStyle

export MPIStateArray, euclidean_distance, weightedsum

"""
    MPIStateArray{S <: Tuple, T, DeviceArray, N,
                  DATN<:AbstractArray{T,N}, Nm1, DAI1} <: AbstractArray{T, N}


`N`-dimensional MPI-aware array with elements of type `T`. The dimension `N` is
`length(S) + 1`. `S` is a tuple of the first `N-1` array dimensions.

!!! todo

    It should be reevaluated whether all this stuff in the type domain is
    really necessary (some of it was optimistically added for functionality that
    never panned out)

"""
struct MPIStateArray{S <: Tuple, T, DeviceArray, N,
                     DATN<:AbstractArray{T,N}, Nm1, DAI1, DAV} <: AbstractArray{T, N}
  mpicomm::MPI.Comm
  Q::DATN
  realQ::DAV

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

  weights::DATN

  commtag::Int
  function MPIStateArray{S, T, DA}(mpicomm, numelem, realelems, ghostelems,
                                   sendelems, nabrtorank, nabrtorecv,
                                   nabrtosend, weights, commtag
                                  ) where {S, T, DA}
    N = length(S.parameters)+1
    numsendelem = length(sendelems)
    numrecvelem = length(ghostelems)
    (Q, device_sendQ, device_recvQ) =
      (DA{T, N}(undef, S.parameters..., numelem),
       DA{T, N}(undef, S.parameters..., numsendelem),
       DA{T, N}(undef, S.parameters..., numrecvelem))

    realQ = view(Q, ntuple(i -> Colon(), ndims(Q) - 1)..., realelems)
    DAV = typeof(realQ)

    host_sendQ = zeros(T, S.parameters..., numsendelem)
    host_recvQ = zeros(T, S.parameters..., numrecvelem)

    nnabr = length(nabrtorank)
    sendreq = fill(MPI.REQUEST_NULL, nnabr)
    recvreq = fill(MPI.REQUEST_NULL, nnabr)

    sendelems = typeof(sendelems) <: DA ? sendelems : DA(sendelems)
    DAI1 = typeof(sendelems)
    new{S, T, DA, N, typeof(Q), N-1, DAI1, DAV}(mpicomm, Q, realQ,
                                                realelems, ghostelems,
                                                sendelems, sendreq, recvreq,
                                                host_sendQ, host_recvQ, nabrtorank,
                                                nabrtorecv, nabrtosend,
                                                device_sendQ, device_recvQ, weights,
                                                commtag)
  end
end

Base.fill!(Q::MPIStateArray, x) = fill!(Q.Q, x)

"""
   MPIStateArray{S, T, DA}(mpicomm, numelem; realelems=1:numelem,
                           ghostelems=numelem:numelem-1,
                           sendelems=1:0,
                           nabrtorank=Array{Int64}(undef, 0),
                           nabrtorecv=Array{UnitRange{Int64}}(undef, 0),
                           nabrtosend=Array{UnitRange{Int64}}(undef, 0),
                           weights,
                           commtag=888)

Construct an `MPIStateArray` over the communicator `mpicomm` with `numelem`
elements, using array type `DA` with element type `eltype`. The arrays that are
held in this created `MPIStateArray` will be of size `(S..., numelem)`.

The range `realelems` is the number of elements that this mpirank owns, whereas
the range `ghostelems` is the elements that are owned by other mpiranks.
Elements are stored as 'realelems` followed by `ghostelems`.

  * `sendelems` is an ordered array of elements to be sent to neighboring
    mpiranks
  * `nabrtorank` is the list of neighboring mpiranks
  * `nabrtorecv` is an `Array` of `UnitRange` that give the `ghostelems`
    received from neighboring mpiranks (indexes into the ghost elements arrays,
    not the full element array)
  * nabrtosend` is an `Array` of `UnitRange` for which elements to send to
    which neighboring mpiranks indexing into the `sendelems` ordering
  * `weights` is an optional array which gives weight for each degree of freedom
    to be used when computing the 2-norm of the array
"""
function MPIStateArray{S, T, DA}(mpicomm, numelem;
                                 realelems=1:numelem,
                                 ghostelems=numelem:numelem-1,
                                 sendelems=1:0,
                                 nabrtorank=Array{Int64}(undef, 0),
                                 nabrtorecv=Array{UnitRange{Int64}}(undef, 0),
                                 nabrtosend=Array{UnitRange{Int64}}(undef, 0),
                                 weights=nothing,
                                 commtag=888
                                ) where {S<:Tuple, T, DA}

  N = length(S.parameters)+1
  if weights == nothing
    weights = DA{T}(undef, ntuple(j->0, N))
  elseif !(typeof(weights) <: DA)
    weights = DA(weights)
  end
  MPIStateArray{S, T, DA}(mpicomm, numelem, realelems, ghostelems,
                          sendelems, nabrtorank, nabrtorecv,
                          nabrtosend, weights, commtag)
end

# FIXME: should general cases be handled?
function Base.similar(Q::MPIStateArray{S, T, DA}, ::Type{TN}, ::Type{DAN}; commtag=Q.commtag
                     ) where {S, T, DA, TN, DAN <: AbstractArray}
  MPIStateArray{S, TN, DAN}(Q.mpicomm, size(Q.Q)[end], Q.realelems, Q.ghostelems,
                            Q.sendelems, Q.nabrtorank, Q.nabrtorecv,
                            Q.nabrtosend, Q.weights, commtag)
end

function Base.similar(Q::MPIStateArray{S, T, DA}, ::Type{TN}; commtag=Q.commtag
                     ) where {S, T, DA, TN}
  similar(Q, TN, DA, commtag = commtag)
end

function Base.similar(Q::MPIStateArray{S, T}, ::Type{DAN}; commtag=Q.commtag
                     ) where {S, T, DAN <: AbstractArray}
  similar(Q, T, DAN, commtag = commtag)
end

function Base.similar(Q::MPIStateArray{S, T}; commtag=Q.commtag
                     ) where {S, T}
  similar(Q, T, commtag = commtag)
end

Base.size(Q::MPIStateArray, x...;kw...) = size(Q.realQ, x...;kw...)

Base.getindex(Q::MPIStateArray, x...;kw...) = getindex(Q.realQ, x...;kw...)

Base.setindex!(Q::MPIStateArray, x...;kw...) = setindex!(Q.realQ, x...;kw...)

Base.eltype(Q::MPIStateArray, x...;kw...) = eltype(Q.Q, x...;kw...)

Base.Array(Q::MPIStateArray) = Array(Q.Q)

# broadcasting stuff

# find the first MPIStateArray among `bc` arguments
# based on https://docs.julialang.org/en/v1/manual/interfaces/#Selecting-an-appropriate-output-array-1
find_mpisa(bc::Broadcasted) = find_mpisa(bc.args)
find_mpisa(args::Tuple) = find_mpisa(find_mpisa(args[1]), Base.tail(args))
find_mpisa(x) = x
find_mpisa(a::MPIStateArray, rest) = a
find_mpisa(::Any, rest) = find_mpisa(rest)

Base.BroadcastStyle(::Type{<:MPIStateArray}) = ArrayStyle{MPIStateArray}()
function Base.similar(bc::Broadcasted{ArrayStyle{MPIStateArray}}, ::Type{T}) where T
  similar(find_mpisa(bc), T)
end

# transform all arguments of `bc` from MPIStateArrays to Arrays
function transform_broadcasted(bc::Broadcasted, ::Array)
  transform_array(bc)
end
function transform_array(bc::Broadcasted)
  Broadcasted(bc.f, transform_array.(bc.args), bc.axes)
end
transform_array(mpisa::MPIStateArray) = mpisa.realQ
transform_array(x) = x

Base.copyto!(dest::Array, src::MPIStateArray) = copyto!(dest, src.Q)

function Base.copyto!(dest::MPIStateArray, src::MPIStateArray)
  copyto!(dest.realQ, src.realQ)
  dest
end

@inline function Base.copyto!(dest::MPIStateArray, bc::Broadcasted{Nothing})
  # check for the case a .= b, where b is an array
  if bc.f == identity
    if typeof(bc.args[1]) <: MPIStateArray
      realindices = CartesianIndices((axes(dest.Q)[1:end-1]..., dest.realelems))
      Base.copyto!(dest.Q, realindices, bc.args[1].Q, realindices)
    else
      Base.copyto!(dest.Q, bc.args[1])
    end
  else
    Base.copyto!(dest.realQ, transform_broadcasted(bc, dest.Q))
  end
  dest
end

"""
    post_Irecvs!(Q::MPIStateArray)

posts the `MPI.Irecv!` for `Q`
"""
function post_Irecvs!(Q::MPIStateArray, D)
  nnabr = length(Q.nabrtorank)
  for n = 1:nnabr
    # If this fails we haven't waited on previous recv!
    @assert Q.recvreq[n].buffer == nothing

    Q.recvreq[n] = MPI.Irecv!((@view Q.host_recvQ[D..., Q.nabrtorecv[n]]),
                              Q.nabrtorank[n], Q.commtag, Q.mpicomm)
  end
end

"""
    start_ghost_exchange!(Q::MPIStateArray; dorecvs=true)

Start the MPI exchange of the data stored in `Q`. If `dorecvs` is `true` then
`post_Irecvs!(Q)` is called, otherwise the caller is responsible for this.

This function will fill the send buffer (on the device), copies the data from
the device to the host, and then issues the send. Previous sends are waited on
to ensure that they are complete.
"""
function start_ghost_exchange!(Q::MPIStateArray, D = ntuple(i->:, ndims(Q)-1);
                               dorecvs=true)

  dorecvs && post_Irecvs!(Q, D)

  # wait on (prior) MPI sends
  finish_ghost_send!(Q)

  # pack data in send buffer
  fillsendbuf!(Q.host_sendQ, Q.device_sendQ, Q.Q, Q.sendelems, D)

  # post MPI sends
  nnabr = length(Q.nabrtorank)
  for n = 1:nnabr
    Q.sendreq[n] = MPI.Isend((@view Q.host_sendQ[D..., Q.nabrtosend[n]]),
                           Q.nabrtorank[n], Q.commtag, Q.mpicomm)
  end
end

"""
    finish_ghost_exchange!(Q::MPIStateArray)

Complete the exchange of data and fill the data array on the device. Note this
completes both the send and the receive communication. For more fine level
control see [finish_ghost_recv!](@ref) and
[finish_ghost_send!](@ref)
"""
function finish_ghost_exchange!(Q::MPIStateArray)
  finish_ghost_recv!(Q::MPIStateArray)
  finish_ghost_send!(Q::MPIStateArray)
end

"""
    finish_ghost_recv!(Q::MPIStateArray)

Complete the receive of data and fill the data array on the device
"""
function finish_ghost_recv!(Q::MPIStateArray, D = ntuple(i->:, ndims(Q)-1))
  # wait on MPI receives
  MPI.Waitall!(Q.recvreq)

  # copy data to state vectors
  transferrecvbuf!(Q.device_recvQ, Q.host_recvQ, Q.Q, length(Q.realelems), D)
end

"""
    finish_ghost_send!(Q::MPIStateArray)

Waits on the send of data to be complete
"""
finish_ghost_send!(Q::MPIStateArray) = MPI.Waitall!(Q.sendreq)

# {{{ MPI Buffer handling
function fillsendbuf!(host_sendbuf, device_sendbuf::Array, buf::Array,
                      sendelems::Array, D)
  # TODO: Revisit when not D ≠ (:, :), may want to pack data perhaps
  # differently for the GPU?
  copyto!(@view(host_sendbuf[D..., :]), @view(buf[D..., sendelems]))
end

function transferrecvbuf!(device_recvbuf, host_recvbuf, buf::Array, nrealelem,
                          D)
  # TODO: Revisit when not D ≠ (:, :), may want to pack data perhaps
  # differently for the GPU?
  copyto!(@view(buf[D..., nrealelem+1:end]), @view(host_recvbuf[D..., :]))
end
# }}}

# Integral based metrics
function LinearAlgebra.norm(Q::MPIStateArray, p::Real=Int32(2))
  T = eltype(Q)

  if isfinite(p)
    E = @~ abs.(Q.realQ).^p
    op, mpiop, init = +, MPI.SUM, zero(T)
  else
    E = @~ abs.(Q.realQ)
    op, mpiop, init = max, MPI.MAX, typemin(T)
  end

  if ~isempty(Q.weights)
    # TODO for more accurate L^p norms we would want to intepolate the fields
    # to a finer mesh.
    w = @view Q.weights[:, :, Q.realelems]
    E = @~ E .* w
  end

  locnorm = mapreduce(identity, op, E, init=init)
  r = MPI.Allreduce([locnorm], mpiop, Q.mpicomm)[1]

  isfinite(p) ? r.^(1//p) : r
end

function euclidean_distance(A::MPIStateArray, B::MPIStateArray)
  E = @~ (A.realQ .- B.realQ).^2

  if ~isempty(A.weights)
    w = @view A.weights[:, :, A.realelems]
    E = @~ E .* w
  end

  locnorm = mapreduce(identity, +, E, init=zero(eltype(A)))
  sqrt(MPI.Allreduce([locnorm], MPI.SUM, A.mpicomm)[1])
end

"""
    weightedsum(A[, states])

Compute the weighted sum of the `MPIStateArray` `A`. If `states` is specified on
the listed states are summed, otherwise all the states in `A` are used.

A typical use case for this is when the weights have been initialized with
quadrature weights from a grid, thus this becomes an integral approximation.
"""
function weightedsum(A::MPIStateArray, states=1:size(A, 2))
  isempty(A.weights) && error("`weightedsum` requires weights")

  T = eltype(A)
  states = SVector{length(states)}(states)

  C = @view A.Q[:, states, A.realelems]
  w = @view A.weights[:, :, A.realelems]
  init = zero(DoubleFloat{T})

  E = @~ DoubleFloat{T}.(C) .* DoubleFloat{T}.(w)

  locwsum = mapreduce(identity, +, E, init=init)

  # Need to use anomous function version of sum else MPI.jl using MPI_SUM
  T(MPI.Allreduce([locwsum], (x,y)->x+y, A.mpicomm)[1])
end

using Requires

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  using .CuArrays
  using .CuArrays.CUDAnative
  using .CuArrays.CUDAnative.CUDAdrv

  # transform all arguments of `bc` from MPIStateArrays to CuArrays
  # and replace CPU function with GPU variants
  function transform_broadcasted(bc::Broadcasted, ::CuArray)
    transform_cuarray(bc)
  end

  function transform_cuarray(bc::Broadcasted)
    Broadcasted(CuArrays.cufunc(bc.f), transform_cuarray.(bc.args), bc.axes)
  end
  transform_cuarray(mpisa::MPIStateArray) = mpisa.realQ
  transform_cuarray(x) = x

  include("MPIStateArrays_cuda.jl")
end

end
