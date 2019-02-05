module AtmosStateArrays

using MPI

export AtmosStateArray

# TODO: Make MPI Aware
struct AtmosStateArray{S <: Tuple, T, N, DeviceArray,
                       DATN<:AbstractArray{T,N}} <: AbstractArray{T, N}
  mpicomm::MPI.Comm
  Q::DATN

  realelems::UnitRange{Int64}
  ghostelems::UnitRange{Int64}
  sendelems::Array{Int64, 1}

  sendreq::Array{MPI.Request, 1}
  recvreq::Array{MPI.Request, 1}

  host_sendQ::Array{T, N}
  host_recvQ::Array{T, N}

  nabrtorank::Array{Int64, 1}
  nabrtorecv::Array{UnitRange{Int64}, 1}
  nabrtosend::Array{UnitRange{Int64}, 1}

  device_sendQ::DATN
  device_recvQ::DATN

  # FIXME: handle MPI properly
  function AtmosStateArray{S, T, DA}(mpicomm, numelem;
                                     realelems=1:numelem,
                                     ghostelems=numelem:numelem-1,
                                     sendelems=1:0,
                                     nabrtorank=Array{Int64}(undef, 0),
                                     nabrtorecv=Array{UnitRange{Int64}}(undef, 0),
                                     nabrtosend=Array{UnitRange{Int64}}(undef, 0),
                                    ) where {S<:Tuple, T, DA}
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

    new{S, T, N, DA, typeof(Q)}(mpicomm, Q, realelems, ghostelems, sendelems,
                                sendreq, recvreq, host_sendQ, host_recvQ,
                                nabrtorank, nabrtorecv, nabrtosend,
                                device_sendQ, device_recvQ)
  end
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

# FIXME: should general cases should be handled?
function Base.similar(Q::AtmosStateArray{S, T, N, DA, DATN}) where {S, T, N, DA,
                                                                    DATN}
  AtmosStateArray{S, T, DA}(Q.mpicomm, size(Q.Q)[end], realelems=Q.realelems)
end

end
