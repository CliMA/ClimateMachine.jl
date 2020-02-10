module CMBuffers

using Requires
using MPI
using CUDAapi

export CMBuffer
export SingleCMBuffer, DoubleCMBuffer
export get_stage, get_transfer, prepare_transfer!, prepare_stage!

@enum CMBufferKind begin
    SingleCMBuffer
    DoubleCMBuffer
end

###
# Note: We use pinned and device-mapped hostbuffers in double staging
# Potential improvements
# - Investigate if we need to device-map
# - Implement single buffered with pinned + device-mapped hostbuffers

"""
    CMBuffer{T}(::Type{Arr}, kind, dims...; pinned = true)

CUDA/MPI buffer abstracts storage for MPI communication. The buffer is
used for staging data and for MPI transfers. When running on:

  - CPU -- a single buffer is used for staging and MPI transfers can be
    initiated directly to/from it.
  - CUDA -- either:
    - MPI is CUDA-aware: a single buffer on the device for staging and MPI
      transfers can be initiated directly to/from it, or
    - MPI is not CUDA-aware: a double buffering scheme with the staging
      buffer on the device and a transfer buffer on the host

# Arguments
- `T`: element type
- `Arr::Type`: what kind of array to allocate for `stage`
- `kind::CMBufferKind`: either `SingleCMBuffer` or `DoubleCMBuffer`
- `dims...`: dimensions of the array

# Keyword Arguments
- `pin`:  register the `transfer` buffer with CUDA
"""
struct CMBuffer{T, Arr, Buff}
    stage :: Arr     # Same type as Q.data, used for staging
    transfer :: Buff # Union{Nothing,Buff}

    function CMBuffer{T}(::Type{Arr}, kind, dims...; pin=true) where {T, Arr}
        if kind == SingleCMBuffer
            transfer = nothing
        elseif kind == DoubleCMBuffer
            transfer = zeros(T, dims...)
            if pin
                # XXX: make this
                # transfer = register(transfer) to obtain a HostCMBuffer
                # or try:
                # ```
                #   transfer = register(transfer)::HostCMBuffer
                #   stage    = # CuArray(transfer)
                # ```
                register(transfer)
                # XXX: Should this finalizer be attached to the CMBuffer struct?
                finalizer(unregister, transfer)
            end
        else
            error("CMBufferkind $kind is not implemented yet")
        end

        stage = similar(Arr, T, dims...)
        buffer = new{T, typeof(stage), typeof(transfer)}(stage, transfer)

        return buffer
    end
end

function get_stage(buf::CMBuffer)
    return buf.stage
end

function get_transfer(buf::CMBuffer)
    if buf.transfer === nothing
        return buf.stage
    else
        return buf.transfer
    end
end

function prepare_transfer!(buf::CMBuffer)
    if buf.transfer === nothing
        # nothing to do here
    else
        copybuffer!(buf.transfer, buf.stage, async=false)
    end
end

function prepare_stage!(buf::CMBuffer)
    if buf.transfer === nothing
        # nothing to do here
    else
        copybuffer!(buf.stage, buf.transfer, async=false)
    end
end

######
# Internal methods
######
register(x) = nothing
unregister(x) = nothing

"""
    copybuffer!(A, B; async=true)

Copy a buffer from device to host or vice-versa. Internally this uses
`cudaMemcpyAsync` on the `CuDefaultStream`. The keyword argument 
`async` determines whether it is asynchronous with regard to the host.
"""
function copybuffer! end

function copybuffer!(A::AbstractArray, B::AbstractArray; async=true)
    copy!(A, B)
end

@init @require CUDAdrv = "c5f51814-7f29-56b8-a69c-e4d8f6be1fde" @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    using .CUDAdrv
    using .CuArrays

    import CUDAdrv.Mem

    function register(arr)
        if sizeof(arr) == 0
            #@warn "Array is of size 0. Can't pin register array with CUDA" size(arr) typeof(arr)
            return arr
        end
        GC.@preserve arr begin
            # XXX: is HOSTREGISTER_DEVICEMAP useful?
            Mem.register(Mem.HostBuffer, pointer(arr), sizeof(arr), Mem.HOSTREGISTER_DEVICEMAP)
        end
    end

    function unregister(arr)
        if sizeof(arr) == 0
            return
        end
        GC.@preserve arr begin
            Mem.unregister(Mem.HostBuffer(pointer(arr), sizeof(arr), CuCurrentContext(), true))
        end
    end

    # CUDAdrv.jl throws on CUDA_ERROR_NOT_READY
    function queryStream(stream)
        err = CUDAapi.@runtime_ccall((:cuStreamQuery, CUDAdrv.libcuda), CUDAdrv.CUresult,
                                     (CUDAdrv.CUstream,), stream)

        if err === CUDAdrv.CUDA_ERROR_NOT_READY
            return false
        elseif err === CUDAdrv.CUDA_SUCCESS
            return true
        else
            CUDAdrv.throw_api_error(err)
        end
    end

    """
        friendlysynchronize(stream)

    MPI defines a notion of progress which means that MPI operations
    need the program to call MPI functions (potentially multiple times)
    to make progress and eventually complete. In some implementations,
    progress on one rank may need MPI to be called on another rank.

    As a result blocking by for example calling cudaStreamSynchronize,
    may create a deadlock in some cases because not calling MPI will
    not make other ranks progress.
    """
    function friendlysynchronize(stream)
        status = false
        while !status
            status = queryStream(stream)
            MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, MPI.COMM_WORLD)
        end
        return
    end

    function async_copy!(A, B, N, stream)
        GC.@preserve A B begin
            #copyto!(A, B)
            ptrA = pointer(A)
            ptrB = pointer(B)
            unsafe_copyto!(ptrA, ptrB, N, async=true, stream=stream)
        end
    end
    function copybuffer!(A::Array, B::CuArray; async=true)
        @assert sizeof(A) == sizeof(B)
        stream = CuDefaultStream()
        async_copy!(A, B, length(A), stream)
        if !async
            friendlysynchronize(stream)
        end
    end
    function copybuffer!(A::CuArray, B::Array; async=true)
        @assert sizeof(A) == sizeof(B)
        stream = CuDefaultStream()
        async_copy!(A, B, length(A), stream)
        if !async
            friendlysynchronize(stream)
        end
    end
end

end # module
