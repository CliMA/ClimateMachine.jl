module CMBuffers

using Requires
using MPI
using CUDAapi
using KernelAbstractions
using StaticArrays

export CMBuffer
export SingleCMBuffer, DoubleCMBuffer
export get_stage, get_transfer, prepare_transfer!, prepare_stage!

@enum CMBufferKind begin
    SingleCMBuffer
    DoubleCMBuffer
end

device(::Union{Array, SArray, MArray}) = CPU()
@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    using .CuArrays
    device(::CuArray) = CUDA()
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

function prepare_transfer!(buf::CMBuffer; dependencies = nothing)
    if buf.transfer === nothing
        return MultiEvent(dependencies)
    end

    event = async_copy!(
        device(buf.stage),
        buf.transfer,
        buf.stage;
        dependencies = dependencies,
    )

    return event
end

function prepare_stage!(buf::CMBuffer; dependencies = nothing)
    if buf.transfer === nothing
        return MultiEvent(dependencies)
    end

    event = async_copy!(
        device(buf.stage),
        buf.stage,
        buf.transfer;
        dependencies = dependencies,
    )

    return event
end

end # module
