abstract type TracerBC end

"""
    ImpermeableTracer() :: TracerBC

No tracer diffusion across boundary
"""
struct ImpermeableTracer <: TracerBC end
function boundary_state!(
    nf,
    bc_tracer::ImpermeableTracer,
    atmos::AtmosModel,
    args...,
)
    nothing
end
