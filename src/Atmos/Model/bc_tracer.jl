abstract type TracerBC end

"""
    ImpermeableTracer() :: TracerBC

No tracer diffusion across boundary
"""
struct ImpermeableTracer <: TracerBC end
function atmos_tracer_boundary_state!(
    nf,
    bc_tracer::ImpermeableTracer,
    atmos,
    args...,
)
    nothing
end
function atmos_tracer_normal_boundary_flux_second_order!(
    nf,
    bc_tracer::ImpermeableTracer,
    atmos,
    args...,
)
    nothing
end
