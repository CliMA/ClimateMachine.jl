module CMBuffers

import CUDA

using MPI
using KernelAbstractions
using CUDAKernels
using StaticArrays

export CMBuffer
export SingleCMBuffer, DoubleCMBuffer
export get_stage, get_transfer, prepare_transfer!, prepare_stage!

@enum CMBufferKind begin
    SingleCMBuffer
    DoubleCMBuffer
end

device(::Union{Array, SArray, MArray}) = CPU()
device(::CUDA.CuArray) = CUDADevice()

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
"""
struct CMBuffer{T, Arr, Buff}
    stage::Arr     # Same type as Q.data, used for staging
    transfer::Buff # Union{Nothing,Buff}

    function CMBuffer{T}(::Type{Arr}, kind, dims...) where {T, Arr}
        if kind == SingleCMBuffer
            transfer = nothing
        elseif kind == DoubleCMBuffer
            transfer = zeros(T, dims...)
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

function prepare_transfer!(
    buf::CMBuffer;
    dependencies = nothing,
    progress = yield,
)
    if buf.transfer === nothing || length(buf.transfer) == 0
        return MultiEvent(dependencies)
    end

    event = async_copy!(
        device(buf.stage),
        buf.transfer,
        buf.stage;
        dependencies = dependencies,
        progress = progress,
    )

    return event
end

function prepare_stage!(buf::CMBuffer; dependencies = nothing, progress = yield)
    if buf.transfer === nothing || length(buf.stage) == 0
        return MultiEvent(dependencies)
    end

    event = async_copy!(
        device(buf.stage),
        buf.stage,
        buf.transfer;
        dependencies = dependencies,
        progress = progress,
    )

    return event
end

end # module
