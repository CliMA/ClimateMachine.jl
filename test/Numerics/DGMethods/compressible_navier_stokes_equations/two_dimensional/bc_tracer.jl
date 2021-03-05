"""
    cnse_boundary_state!(::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, ::Insulating, ::HBModel)

apply insulating boundary condition for temperature
sets transmissive ghost point
"""
function cnse_boundary_state!(
    ::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::Insulating,
    ::CNSE2D,
    state⁺,
    aux⁺,
    n⁻,
    state⁻,
    aux⁻,
    t,
)
    state⁺.ρθ = state⁻.ρθ

    return nothing
end

"""
    cnse_boundary_state!(::NumericalFluxSecondOrder, ::Insulating, ::HBModel)

apply insulating boundary condition for velocity
sets ghost point to have no numerical flux on the boundary for κ∇θ
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Insulating,
    ::CNSE2D,
    state⁺,
    gradflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    aux⁻,
    t,
)
    state⁺.ρθ = state⁻.ρθ
    gradflux⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    cnse_boundary_state!(::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, ::TemperatureFlux, ::HBModel)

apply temperature flux boundary condition for velocity
applies insulating conditions for first-order and gradient fluxes
"""
function cnse_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::TemperatureFlux,
    model::CNSE2D,
    args...,
)
    return cnse_boundary_state!(nf, Insulating(), model, args...)
end

"""
    cnse_boundary_state!(::NumericalFluxSecondOrder, ::TemperatureFlux, ::HBModel)

apply insulating boundary condition for velocity
sets ghost point to have specified flux on the boundary for κ∇θ
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    bc::TemperatureFlux,
    model::CNSE2D,
    state⁺,
    gradflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    aux⁻,
    t,
)
    state⁺.ρθ = state⁻.ρθ
    gradflux⁺.κ∇θ = n⁻ * bc.flux(state⁻, aux⁻, t)

    return nothing
end
