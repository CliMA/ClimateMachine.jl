module MPIStateArrays
using LinearAlgebra
using DoubleFloats
using LazyArrays
using StaticArrays
using GPUifyLoops

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
                     DATN<:AbstractArray{T,N}, Nm1, DAI1, DAV,
                     DAT2<:AbstractArray{T,2}} <: AbstractArray{T, N}
  mpicomm::MPI.Comm
  data::DATN
  realdata::DAV

  realelems::UnitRange{Int64}
  ghostelems::UnitRange{Int64}

  vmaprecv::DAI1
  vmapsend::DAI1

  sendreq::Array{MPI.Request, 1}
  recvreq::Array{MPI.Request, 1}

  host_send_buffer::Array{T, 2}
  host_recv_buffer::Array{T, 2}

  nabrtorank::Array{Int64, 1}
  nabrtovmaprecv::Array{UnitRange{Int64}, 1}
  nabrtovmapsend::Array{UnitRange{Int64}, 1}

  device_send_buffer::DAT2
  device_recv_buffer::DAT2

  weights::DATN

  commtag::Int
  function MPIStateArray{S, T, DA}(mpicomm, numelem, realelems, ghostelems,
                                   vmaprecv, vmapsend, nabrtorank, nabrtovmaprecv,
                                   nabrtovmapsend, weights, commtag
                                  ) where {S, T, DA}
    N = length(S.parameters)+1
    data = DA{T, N}(undef, S.parameters..., numelem)

    device_recv_buffer = DA{T}(undef, S.parameters[2], length(vmaprecv))
    device_send_buffer = DA{T}(undef, S.parameters[2], length(vmapsend))

    realdata = view(data, ntuple(i -> Colon(), ndims(data) - 1)..., realelems)
    DAV = typeof(realdata)

    host_send_buffer = zeros(T, S.parameters[2], length(vmapsend))
    host_recv_buffer = zeros(T, S.parameters[2], length(vmaprecv))

    nnabr = length(nabrtorank)
    sendreq = fill(MPI.REQUEST_NULL, nnabr)
    recvreq = fill(MPI.REQUEST_NULL, nnabr)

    vmaprecv = typeof(vmaprecv) <: DA ? vmaprecv : DA(vmaprecv)
    vmapsend = typeof(vmapsend) <: DA ? vmapsend : DA(vmapsend)
    DAI1 = typeof(vmaprecv)
    DAT2 = typeof(device_recv_buffer)
    new{S, T, DA, N, typeof(data), N-1, DAI1, DAV, DAT2}(mpicomm, data, realdata,
                                                      realelems, ghostelems,
                                                      vmaprecv, vmapsend,
                                                      sendreq, recvreq,
                                                      host_send_buffer, host_recv_buffer,
                                                      nabrtorank, nabrtovmaprecv,
                                                      nabrtovmapsend,
                                                      device_send_buffer,
                                                      device_recv_buffer, weights,
                                                      commtag)
  end
end

Base.fill!(Q::MPIStateArray, x) = fill!(Q.data, x)

"""
   MPIStateArray{S, T, DA}(mpicomm, numelem; realelems=1:numelem,
                           ghostelems=numelem:numelem-1,
                           vmaprecv=1:0,
                           vmapsend=1:0,
                           nabrtorank=Array{Int64}(undef, 0),
                           nabrtovmaprecv=Array{UnitRange{Int64}}(undef, 0),
                           nabrtovmapsend=Array{UnitRange{Int64}}(undef, 0),
                           weights,
                           commtag=888)

Construct an `MPIStateArray` over the communicator `mpicomm` with `numelem`
elements, using array type `DA` with element type `eltype`. The arrays that are
held in this created `MPIStateArray` will be of size `(S..., numelem)`.

The range `realelems` is the number of elements that this mpirank owns, whereas
the range `ghostelems` is the elements that are owned by other mpiranks.
Elements are stored as 'realelems` followed by `ghostelems`.

  * `vmaprecv` is an ordered array of elements to be received from neighboring
     mpiranks.  This is a vmap index into the MPIStateArray A[:,*,:].
  * `vmapsend` is an ordered array of elements to be sent to neighboring
     mpiranks.  This is a vmap index into the MPIStateArray A[:,*,:].
  * `nabrtorank` is the list of neighboring mpiranks
  * `nabrtovmaprecv` is an `Array` of `UnitRange` that give the ghost data to be
    received from neighboring mpiranks (indexes into `vmaprecv`)
  * nabrtovmapsend` is an `Array` of `UnitRange` for which elements to send to
    which neighboring mpiranks indexing into the `vmapsend`
  * `weights` is an optional array which gives weight for each degree of freedom
    to be used when computing the 2-norm of the array
"""
function MPIStateArray{S, T, DA}(mpicomm, numelem;
                                 realelems=1:numelem,
                                 ghostelems=numelem:numelem-1,
                                 vmaprecv=1:0,
                                 vmapsend=1:0,
                                 nabrtorank=Array{Int64}(undef, 0),
                                 nabrtovmaprecv=Array{UnitRange{Int64}}(undef, 0),
                                 nabrtovmapsend=Array{UnitRange{Int64}}(undef, 0),
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
                          vmaprecv, vmapsend, nabrtorank, nabrtovmaprecv,
                          nabrtovmapsend, weights, commtag)
end

# FIXME: should general cases be handled?
function Base.similar(Q::MPIStateArray{S, T, DA}, ::Type{TN}, ::Type{DAN}; commtag=Q.commtag
                     ) where {S, T, DA, TN, DAN <: AbstractArray}
  MPIStateArray{S, TN, DAN}(Q.mpicomm, size(Q.data)[end], Q.realelems, Q.ghostelems,
                            Q.vmaprecv, Q.vmapsend, Q.nabrtorank, Q.nabrtovmaprecv,
                            Q.nabrtovmapsend, Q.weights, commtag)
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

Base.size(Q::MPIStateArray, x...;kw...) = size(Q.realdata, x...;kw...)

Base.getindex(Q::MPIStateArray, x...;kw...) = getindex(Q.realdata, x...;kw...)

Base.setindex!(Q::MPIStateArray, x...;kw...) = setindex!(Q.realdata, x...;kw...)

Base.eltype(Q::MPIStateArray, x...;kw...) = eltype(Q.data, x...;kw...)

Base.Array(Q::MPIStateArray) = Array(Q.data)

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
transform_array(mpisa::MPIStateArray) = mpisa.realdata
transform_array(x) = x

Base.copyto!(dest::Array, src::MPIStateArray) = copyto!(dest, src.data)

function Base.copyto!(dest::MPIStateArray, src::MPIStateArray)
  copyto!(dest.realdata, src.realdata)
  dest
end

@inline function Base.copyto!(dest::MPIStateArray, bc::Broadcasted{Nothing})
  # check for the case a .= b, where b is an array
  if bc.f === identity && bc.args isa Tuple{AbstractArray}
    if bc.args isa Tuple{MPIStateArray}
      realindices = CartesianIndices((axes(dest.data)[1:end-1]..., dest.realelems))
      copyto!(dest.data, realindices, bc.args[1].data, realindices)
    else
      copyto!(dest.data, bc.args[1])
    end
  else
    copyto!(dest.realdata, transform_broadcasted(bc, dest.data))
  end
  dest
end

"""
    post_Irecvs!(Q::MPIStateArray)

posts the `MPI.Irecv!` for `Q`
"""
function post_Irecvs!(Q::MPIStateArray)
  nnabr = length(Q.nabrtorank)
  for n = 1:nnabr
    # If this fails we haven't waited on previous recv!
    @assert Q.recvreq[n].buffer == nothing

    Q.recvreq[n] = MPI.Irecv!((@view Q.host_recv_buffer[:, Q.nabrtovmaprecv[n]]),
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
function start_ghost_exchange!(Q::MPIStateArray; dorecvs=true)

  dorecvs && post_Irecvs!(Q)

  # wait on (prior) MPI sends
  finish_ghost_send!(Q)

  # pack data in send buffer
  fillsendbuf!(Q.host_send_buffer, Q.device_send_buffer, Q.data, Q.vmapsend)

  # post MPI sends
  nnabr = length(Q.nabrtorank)
  for n = 1:nnabr
    Q.sendreq[n] = MPI.Isend((@view Q.host_send_buffer[:, Q.nabrtovmapsend[n]]),
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
function finish_ghost_recv!(Q::MPIStateArray)
  # wait on MPI receives
  MPI.Waitall!(Q.recvreq)

  # copy data to state vectors
  transferrecvbuf!(Q.device_recv_buffer, Q.host_recv_buffer, Q.data, Q.vmaprecv)
end

"""
    finish_ghost_send!(Q::MPIStateArray)

Waits on the send of data to be complete
"""
finish_ghost_send!(Q::MPIStateArray) = MPI.Waitall!(Q.sendreq)

# {{{ MPI Buffer handling
function _fillsendbuf!(sendbuf, buf, vmapsend)
  if length(vmapsend) > 0
    Np = size(buf, 1)
    nvar = size(buf, 2)

    threads = 256
    blocks = div(length(vmapsend) + threads - 1, threads)
    @launch(device(buf), threads=threads, blocks=blocks,
            knl_fillsendbuf!(Val(Np), Val(nvar), sendbuf, buf, vmapsend,
                             length(vmapsend)))
  end
end

function fillsendbuf!(host_sendbuf::Array, device_sendbuf::Array, buf::Array,
                      vmapsend::Array)
  _fillsendbuf!(host_sendbuf, buf, vmapsend)
end

function fillsendbuf!(host_sendbuf, device_sendbuf, buf, vmapsend)
  _fillsendbuf!(device_sendbuf, buf, vmapsend,)

  if length(vmapsend) > 0
    copyto!(host_sendbuf, device_sendbuf)
  end
end

function _transferrecvbuf!(buf, recvbuf, vmaprecv)
  if length(vmaprecv) > 0
    Np = size(buf, 1)
    nvar = size(buf, 2)

    threads = 256
    blocks = div(length(vmaprecv) + threads - 1, threads)
    @launch(device(buf), threads=threads, blocks=blocks,
            knl_transferrecvbuf!(Val(Np), Val(nvar), buf, recvbuf,
                                 vmaprecv, length(vmaprecv)))
  end
end

function transferrecvbuf!(device_recvbuf::Array, host_recvbuf::Array,
                          buf::Array, vmaprecv::Array)
  _transferrecvbuf!(buf, host_recvbuf, vmaprecv)
end

function transferrecvbuf!(device_recvbuf, host_recvbuf, buf, vmaprecv)
  if length(vmaprecv) > 0
    copyto!(device_recvbuf, host_recvbuf)
  end

  _transferrecvbuf!(buf, device_recvbuf, vmaprecv)
end
# }}}

# Integral based metrics
function LinearAlgebra.norm(Q::MPIStateArray, p::Real=2, weighted::Bool=true; dims=nothing)
  if weighted && ~isempty(Q.weights)
    W = @view Q.weights[:, :, Q.realelems]
    locnorm = weighted_norm_impl(Q.realdata, W, Val(p), dims)
  else
    locnorm = norm_impl(Q.realdata, Val(p), dims)
  end

  mpiop = isfinite(p) ? MPI.SUM : MPI.MAX
  if locnorm isa AbstractArray
    locnorm = convert(Array, locnorm)
  end
  r = MPI.Allreduce(locnorm, mpiop, Q.mpicomm)
  isfinite(p) ? r .^ (1 // p) : r
end
LinearAlgebra.norm(Q::MPIStateArray, weighted::Bool; dims=nothing) = norm(Q, 2, weighted; dims=dims)

function LinearAlgebra.dot(Q1::MPIStateArray, Q2::MPIStateArray, weighted::Bool=true)
  @assert length(Q1.realdata) == length(Q2.realdata)

  if weighted && ~isempty(Q1.weights)
    W = @view Q1.weights[:, :, Q1.realelems]
    locnorm = weighted_dot_impl(Q1.realdata, Q2.realdata, W)
  else
    locnorm = dot_impl(Q1.realdata, Q2.realdata)
  end

  MPI.Allreduce([locnorm], MPI.SUM, Q1.mpicomm)[1]
end

function euclidean_distance(A::MPIStateArray, B::MPIStateArray)
  # work around https://github.com/JuliaArrays/LazyArrays.jl/issues/66
  ArealQ = A.realdata
  BrealQ = B.realdata
  E = @~ (ArealQ .- BrealQ).^2

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

  C = @view A.data[:, states, A.realelems]
  w = @view A.weights[:, :, A.realelems]
  init = zero(DoubleFloat{T})

  E = @~ DoubleFloat{T}.(C) .* DoubleFloat{T}.(w)

  locwsum = mapreduce(identity, +, E, init=init)

  # Need to use anomous function version of sum else MPI.jl using MPI_SUM
  T(MPI.Allreduce([locwsum], (x,y)->x+y, A.mpicomm)[1])
end

# fast CPU local norm & dot implementations
function norm_impl(Q::SubArray{T, N, A}, ::Val{p}, dims::Nothing) where {T, N, A<:Array, p}
  accum = isfinite(p) ? -zero(T) : typemin(T)
  @inbounds @simd for i in eachindex(Q)
    if isfinite(p)
      accum += abs(Q[i]) ^ p
    else
      aQ_i = abs(Q[i])
      accum = ifelse(aQ_i > accum, aQ_i, accum)
    end
  end
  accum
end

function weighted_norm_impl(Q::SubArray{T, N, A}, W, ::Val{p}, dims::Nothing) where {T, N, A<:Array, p}
  nq, ns, ne = size(Q)
  accum = isfinite(p) ? -zero(T) : typemin(T)
  @inbounds for k = 1:ne, j = 1:ns
    @simd for i = 1:nq
      if isfinite(p)
        accum += W[i, 1, k] * abs(Q[i, j, k]) ^ p
      else
        waQ_ijk = W[i, j, k] * abs(Q[i, j, k])
        accum = ifelse(waQ_ijk > accum, waQ_ijk, accum)
      end
    end
  end
  accum
end

function dot_impl(Q1::SubArray{T, N, A}, Q2::SubArray{T, N, A}) where {T, N, A<:Array}
  accum = -zero(T)
  @inbounds @simd for i in eachindex(Q1)
    accum += Q1[i] * Q2[i]
  end
  accum
end


function weighted_dot_impl(Q1::SubArray{T, N, A}, Q2::SubArray{T, N, A}, W) where {T, N, A<:Array}
  nq, ns, ne = size(Q1)
  accum = -zero(T)
  @inbounds for k = 1:ne, j = 1:ns
    @simd for i = 1:nq
      accum += W[i, 1, k] * Q1[i, j, k] * Q2[i, j, k]
    end
  end
  accum
end

# GPU/generic local norm & dot implementations
function norm_impl(Q, ::Val{p}, dims=nothing) where p
  T = eltype(Q)
  if !isfinite(p)
    f, op = abs, max
  elseif p == 1
    f, op = abs, +
  elseif p == 2
    f, op = abs2, +
  else
    f, op = x -> abs(x)^p, +
  end
  dims==nothing ? mapreduce(f, op, Q) :
                  mapreduce(f, op, Q, dims=dims)
end

function weighted_norm_impl(Q, W, ::Val{p}, dims=nothing) where p
  T = eltype(Q)
  if isfinite(p)
    E = @~ @. W * abs(Q) ^ p
    op, init = +, zero(T)
  else
    E = @~ @. W * abs(Q)
    op, init = max, typemin(T)
  end
  dims==nothing ? reduce(op, E, init=init) :
                  reduce(op, E, init=init, dims=dims)
end

function dot_impl(Q1, Q2)
  T = eltype(Q1)
  E = @~ @. Q1 * Q2
  mapreduce(identity, +, E, init=zero(T))
end

weighted_dot_impl(Q1, Q2, W) = dot_impl(@~ @. W * Q1, Q2)

using Requires

# `realview` and `device` are helpers that enable
# testing ODESolvers and LinearSolvers without using MPIStateArrays
# They could be potentially useful elsewhere and exported but probably need
# better names, for example `device` is also defined in CUDAdrv

device(::Union{Array, SArray, MArray}) = CPU()
device(Q::MPIStateArray) = device(Q.data)

realview(Q::Union{Array, SArray, MArray}) = Q
realview(Q::MPIStateArray) = Q.realdata

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  using .CuArrays
  using .CuArrays.CUDAnative

  device(::CuArray) = CUDA()
  realview(Q::CuArray) = Q

  # transform all arguments of `bc` from MPIStateArrays to CuArrays
  # and replace CPU function with GPU variants
  function transform_broadcasted(bc::Broadcasted, ::CuArray)
    transform_cuarray(bc)
  end

  function transform_cuarray(bc::Broadcasted)
    Broadcasted(CuArrays.cufunc(bc.f), transform_cuarray.(bc.args), bc.axes)
  end
  transform_cuarray(mpisa::MPIStateArray) = mpisa.realdata
  transform_cuarray(x) = x
end

include("MPIStateArrays_kernels.jl")

end
