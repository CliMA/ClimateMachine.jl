module Callbacks

using KernelAbstractions
using MPI
using Printf
using Requires
using Statistics

using ..GenericCallbacks

_sync_device(::Type{Array}) = nothing
@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    using .CuArrays, .CuArrays.CUDAdrv
    _sync_device(::Type{CuArray}) = synchronize()
end

"""
    wall_clock_time_per_time_step(interval, array_type, comm)

Returns a callback function that gives wall-clock time per time-step statistics
across MPI ranks in the communicator `comm`.  The times are averaged over the
`interval` of time steps requested for output.  The `array_type` is used
to synchronize the compute device.
"""
function wall_clock_time_per_time_step(interval, array_type, comm)
    _sync_device(array_type)
    before = time_ns()

    callback = GenericCallbacks.EveryXSimulationSteps(interval) do
        _sync_device(array_type)
        after = time_ns()

        time_per_timesteps = after - before

        times = MPI.Gather(time_per_timesteps, 0, comm)
        if MPI.Comm_rank(comm) == 0
            ns_per_s = 1e9
            times = times ./ ns_per_s ./ interval

            @info @sprintf(
                """Wall-clock time per time-step (statistics across MPI ranks)
                   maximum (s) = %25.16e
                   minimum (s) = %25.16e
                   median  (s) = %25.16e
                   std     (s) = %25.16e
                """,
                maximum(times),
                minimum(times),
                median(times),
                std(times),
            )
        end

        _sync_device(array_type)
        before = time_ns()

        nothing
    end

    return callback
end

end
