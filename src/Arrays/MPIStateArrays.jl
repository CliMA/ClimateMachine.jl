module MPIStateArrays
using LinearAlgebra

using MPI

export MPIStateArray, euclidean_distance

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

  commtag::Int
  function MPIStateArray{S, T, DA}(mpicomm, numelem, realelems, ghostelems,
                                   sendelems, nabrtorank, nabrtorecv,
                                   nabrtosend, weights, commtag
                                  ) where {S, T, DA}
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
        error("MPIStateArray:Cannot construct array")
      end
    end

    host_sendQ = zeros(T, S.parameters..., numsendelem)
    host_recvQ = zeros(T, S.parameters..., numrecvelem)

    # Length check is to work around a CuArrays bug.
    length(Q) > 0 && fill!(Q, 0)
    length(device_sendQ) > 0 && fill!(device_sendQ, 0)
    length(device_recvQ) > 0 && fill!(device_recvQ, 0)

    nnabr = length(nabrtorank)
    sendreq = fill(MPI.REQUEST_NULL, nnabr)
    recvreq = fill(MPI.REQUEST_NULL, nnabr)

    sendelems = typeof(sendelems) <: DA ? sendelems : DA(sendelems)
    DAI1 = typeof(sendelems)
    new{S, T, DA, N, typeof(Q), N-1, DAI1}(mpicomm, Q, realelems, ghostelems,
                                           sendelems, sendreq, recvreq,
                                           host_sendQ, host_recvQ, nabrtorank,
                                           nabrtorecv, nabrtosend,
                                           device_sendQ, device_recvQ, weights,
                                           commtag)
  end


end

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
    weights = Array{T}(undef, ntuple(j->0, N-1))
  elseif !(typeof(weights) <: Array)
    weights = Array(weights)
  end
  MPIStateArray{S, T, DA}(mpicomm, numelem, realelems, ghostelems,
                          sendelems, nabrtorank, nabrtorecv,
                          nabrtosend, weights, commtag)
end

# FIXME: should general cases should be handled?
function Base.similar(Q::MPIStateArray{S, T, DA}; commtag=Q.commtag
                     ) where {S, T, DA}
  MPIStateArray{S, T, DA}(Q.mpicomm, size(Q.Q)[end], Q.realelems, Q.ghostelems,
                          Q.sendelems, Q.nabrtorank, Q.nabrtorecv,
                          Q.nabrtosend, Q.weights, commtag)
end

# FIXME: Only show real size
Base.size(Q::MPIStateArray, x...;kw...) = size(Q.Q, x...;kw...)

# FIXME: Only let get index access real elements?
Base.getindex(Q::MPIStateArray, x...;kw...) = getindex(Q.Q, x...;kw...)

# FIXME: Only let set index access real elements?
Base.setindex!(Q::MPIStateArray, x...;kw...) = setindex!(Q.Q, x...;kw...)

Base.eltype(Q::MPIStateArray, x...;kw...) = eltype(Q.Q, x...;kw...)

Base.Array(Q::MPIStateArray) = Array(Q.Q)

function Base.copyto!(dst::MPIStateArray, src::Array)
  copyto!(dst.Q, src)
  dst
end
Base.copyto!(dst::Array, src::MPIStateArray) = copyto!(dst, src.Q)
function Base.copyto!(dst::MPIStateArray, src::MPIStateArray)
  copyto!(dst.Q, src.Q)
  dst
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

    Q.recvreq[n] = MPI.Irecv!((@view Q.host_recvQ[:, :, Q.nabrtorecv[n]]),
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
  MPI.Waitall!(Q.sendreq)

  # pack data in send buffer
  fillsendbuf!(Q.host_sendQ, Q.device_sendQ, Q.Q, Q.sendelems)

  # post MPI sends
  nnabr = length(Q.nabrtorank)
  for n = 1:nnabr
    Q.sendreq[n] = MPI.Isend((@view Q.host_sendQ[:, :, Q.nabrtosend[n]]),
                           Q.nabrtorank[n], Q.commtag, Q.mpicomm)
  end
end

"""
    finish_ghost_exchange!(Q::MPIStateArray)

Complete the exchange of data and fill the data array on the device
"""
function finish_ghost_exchange!(Q::MPIStateArray)
  # wait on MPI receives
  MPI.Waitall!(Q.recvreq)

  # copy data to state vectors
  transferrecvbuf!(Q.device_recvQ, Q.host_recvQ, Q, length(Q.realelems))
end

# {{{ MPI Buffer handling
fillsendbuf!(h, d, b::MPIStateArray, e) = fillsendbuf!(h, d, b.Q, e)
transferrecvbuf!(h, d, b::MPIStateArray, e) = transferrecvbuf!(h, d, b.Q, e)

function fillsendbuf!(host_sendbuf, device_sendbuf::Array, buf::Array, sendelems)
  host_sendbuf[:, :, :] .= buf[:, :, sendelems]
end

function transferrecvbuf!(device_recvbuf::Array, host_recvbuf, buf::Array,
                          nrealelem)
  buf[:, :, nrealelem+1:end] .= host_recvbuf[:, :, :]
end
# }}}

# {{{ L2 Energy (for all dimensions)
function LinearAlgebra.norm(Q::MPIStateArray; p::Real=2)

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

function euclidean_distance(A::MPIStateArray, B::MPIStateArray)

  host_array = Array ∈ typeof(A).parameters
  h_A = host_array ? A : Array(A)
  Np = size(A, 1)

  host_array = Array ∈ typeof(B).parameters
  h_B = host_array ? B : Array(B)
  @assert Np === size(B, 1)

  if isempty(A.weights)
    locdist = knl_dist(Val(Np), h_A, h_B, A.realelems)
  else
    locdist = knl_L2dist(Val(Np), h_A, h_B, A.weights, A.realelems)
  end

  sqrt(MPI.allreduce([locdist], MPI.SUM, A.mpicomm)[1])
end

function knl_dist(::Val{Np}, A, B, elems) where {Np}
  DFloat = eltype(A)
  (_, nstate, nelem) = size(A)

  dist = zero(DFloat)

  @inbounds for e = elems, q = 1:nstate, i = 1:Np
    dist += (A[i, q, e] - B[i, q, e])^2
  end

  dist
end

function knl_L2dist(::Val{Np}, A, B, weights, elems) where {Np}
  DFloat = eltype(A)
  (_, nstate, nelem) = size(A)

  dist = zero(DFloat)

  @inbounds for e = elems, q = 1:nstate, i = 1:Np
    dist += weights[i, e] * (A[i, q, e] - B[i, q, e])^2
  end

  dist
end

# }}}

using Requires

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  using .CuArrays
  using .CuArrays.CUDAnative
  using .CuArrays.CUDAnative.CUDAdrv

  include("MPIStateArrays_cuda.jl")
end

end
