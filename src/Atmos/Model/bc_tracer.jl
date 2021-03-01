"""
    ImpermeableTracer()

No tracer diffusion across boundary
"""
struct ImpermeableTracer{PV <: Tracers{N} where {N}} <: BCDef{PV} end

# No tracers by default:
ImpermeableTracer(N) = ImpermeableTracer{Tracers{N}}()

# TODO: remove
ImpermeableTracer() = ImpermeableTracer(0)

bc_val(bc::ImpermeableTracer, atmos::AtmosModel, ::NF12∇, args) =
    DefaultBCValue()

function atmos_tracer_normal_boundary_flux_second_order!(
    nf,
    bc_tracer::ImpermeableTracer,
    atmos,
    args...,
)
    nothing
end
