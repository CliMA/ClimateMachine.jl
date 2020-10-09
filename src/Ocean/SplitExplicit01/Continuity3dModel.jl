struct Continuity3dModel{M} <: AbstractOceanModel
    ocean::M
    function Continuity3dModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end
vars_state(cm::Continuity3dModel, ::Prognostic, FT) =
    vars_state(cm.ocean, Prognostic(), FT)

# Continuity3dModel is used to compute the horizontal divergence of u

vars_state(cm::Continuity3dModel, ::Auxiliary, T) = @vars()
vars_state(cm::Continuity3dModel, ::Gradient, T) = @vars()
vars_state(cm::Continuity3dModel, ::GradientFlux, T) = @vars()
vars_state(cm::Continuity3dModel, ::UpwardIntegrals, T) = @vars()
init_state_auxiliary!(cm::Continuity3dModel, _...) = nothing
init_state_prognostic!(cm::Continuity3dModel, _...) = nothing
@inline flux_second_order!(cm::Continuity3dModel, _...) = nothing
@inline source!(cm::Continuity3dModel, _...) = nothing
@inline update_penalty!(::RusanovNumericalFlux, ::Continuity3dModel, _...) =
    nothing

# This allows the balance law framework to compute the horizontal gradient of u
# (which will be stored back in the field θ)
@inline function flux_first_order!(
    m::Continuity3dModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    @inbounds begin
        u = state.u # Horizontal components of velocity
        v = @SVector [u[1], u[2], -0]

        # ∇ • (v)
        # Just using θ to store w = ∇h • u
        flux.θ += v
    end

    return nothing
end

# This is zero because when taking the horizontal gradient we're piggy-backing
# on θ and want to ensure we do not use it's jump
@inline wavespeed(cm::Continuity3dModel, n⁻, _...) = -zero(eltype(n⁻))

boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    cm::Continuity3dModel,
    _...,
) = nothing

"""
    boundary_state!(nf, ::Continuity3dModel, Q⁺, A⁺, Q⁻, A⁻, bctype)

applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, cm::Continuity3dModel, args...)
    # hack for handling multiple boundaries for now
    # will fix with a future update
    boundary_conditions = (
        cm.ocean.problem.boundary_conditions[1],
        cm.ocean.problem.boundary_conditions[1],
        cm.ocean.problem.boundary_conditions[1],
    )
    return ocean_boundary_state!(nf, boundary_conditions, cm, args...)
end
