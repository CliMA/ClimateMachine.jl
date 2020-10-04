abstract type TracerBC <: BoundaryCondition end

"""
    ImpermeableTracer() :: TracerBC

No tracer diffusion across boundary
"""
struct ImpermeableTracer <: TracerBC end
