"""
    calc_boundary_state!(::NumericalFluxSecondOrder, ::Impenetrable{FreeSlip}, ::ModelSetup)
apply free slip boundary condition for velocity
apply zero numerical flux in the normal direction
"""
function calc_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{FreeSlip},
    ::ModelSetup,
    ::Nothing,
    state⁺,
    gradflux⁺,
    hyperflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    hyperflux⁻,
    aux⁻,
    t,
    _...,
)
    state⁺.ρu = state⁻.ρu

    return nothing
end

function calc_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{FreeSlip},
    ::ModelSetup,
    ::ConstantViscosity,
    state⁺,
    gradflux⁺,
    hyperflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    hyperflux⁻,
    aux⁻,
    t,
    _...,
)
    state⁺.ρu = state⁻.ρu
    gradflux⁺.ν∇u = n⁻ * (@SVector [-0, -0, -0])'

    return nothing
end

"""
    calc_boundary_state!(::NumericalFluxSecondOrder, ::Impenetrable{NoSlip}, ::ModelSetup)
apply no slip boundary condition for velocity
sets ghost point to have no numerical flux on the boundary for U
"""
@inline function calc_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{NoSlip},
    ::ModelSetup,
    ::Nothing,
    state⁺,
    gradflux⁺,
    hyperflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    hyperflux⁻,
    aux⁻,
    t,
    _...,
)
    state⁺.ρu = -state⁻.ρu

    return nothing
end

@inline function calc_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{NoSlip},
    ::ModelSetup,
    ::ConstantViscosity,
    state⁺,
    gradflux⁺,
    hyperflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    hyperflux⁻,
    aux⁻,
    t,
    _...,
)
    state⁺.ρu = -state⁻.ρu
    gradflux⁺.ν∇u = gradflux⁻.ν∇u

    return nothing
end

"""
    calc_boundary_state!(::NumericalFluxSecondOrder, ::Insulating, ::HBModel)

apply insulating boundary condition for velocity
sets ghost point to have no numerical flux on the boundary for κ∇θ
"""
@inline function calc_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Insulating,
    ::ModelSetup,
    ::Nothing,
    state⁺,
    gradflux⁺,
    hyperflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    hyperflux⁻,
    aux⁻,
    t,
    _...,
)
    state⁺.ρθ = state⁻.ρθ

    return nothing
end

@inline function calc_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Insulating,
    ::ModelSetup,
    ::ConstantViscosity,  
    state⁺,
    gradflux⁺,
    hyperflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    hyperflux⁻,
    aux⁻,
    t,
    _...,
)
    state⁺.ρθ = state⁻.ρθ
    gradflux⁺.κ∇θ = n⁻ * -0

    return nothing
end

# """
#     calc_boundary_state!(::NumericalFluxSecondOrder, ::Penetrable{FreeSlip}, ::ModelSetup)
# apply free slip boundary condition for velocity
# apply zero numerical flux in the normal direction
# """
# @inline function calc_boundary_state!(
#     ::NumericalFluxSecondOrder,
#     ::Penetrable{FreeSlip},
#     ::ModelSetup,
#     ::ConstantViscosity,
#     state⁺,
#     gradflux⁺,
#     aux⁺,
#     n⁻,
#     state⁻,
#     gradflux⁻,
#     aux⁻,
#     t,
#     args...,
# )
#     state⁺.ρu = state⁻.ρu
#     gradflux⁺.ν∇u = n⁻ * (@SVector [-0, -0, -0])'

#     return nothing
# end

# """
#     calc_boundary_state!(::NumericalFluxSecondOrder, ::Impenetrable{MomentumFlux}, ::HBModel)
# apply kinematic stress boundary condition for velocity
# sets ghost point to have specified flux on the boundary for ν∇u
# """
# @inline function calc_boundary_state!(
#     ::NumericalFluxSecondOrder,
#     bc::Impenetrable{<:MomentumFlux},
#     model::ModelSetup,
#     ::ConstantViscosity,
#     state⁺,
#     gradflux⁺,
#     aux⁺,
#     n⁻,
#     state⁻,
#     gradflux⁻,
#     aux⁻,
#     t,
#     args...,
# )
#     state⁺.ρu = state⁻.ρu
#     gradflux⁺.ν∇u = n⁻ * bc.drag.flux(state⁻, aux⁻, t)'

#     return nothing
# end

# """
#     calc_boundary_state!(::NumericalFluxSecondOrder, ::Penetrable{MomentumFlux}, ::HBModel)
# apply kinematic stress boundary condition for velocity
# sets ghost point to have specified flux on the boundary for ν∇u
# """
# @inline function calc_boundary_state!(
#     ::NumericalFluxSecondOrder,
#     bc::Penetrable{<:MomentumFlux},
#     shallow::ModelSetup,
#     ::ConstantViscosity,
#     state⁺,
#     gradflux⁺,
#     aux⁺,
#     n⁻,
#     state⁻,
#     gradflux⁻,
#     aux⁻,
#     t,
#     args...,
# )
#     state⁺.ρu = state⁻.ρu
#     gradflux⁺.ν∇u = n⁻ * bc.drag.flux(state⁻, aux⁻, t)'

#     return nothing
# end

# """
#     calc_boundary_state!(::NumericalFluxSecondOrder, ::TemperatureFlux, ::HBModel)

# apply insulating boundary condition for velocity
# sets ghost point to have specified flux on the boundary for κ∇θ
# """
# @inline function calc_boundary_state!(
#     ::NumericalFluxSecondOrder,
#     bc::TemperatureFlux,
#     ::ModelSetup,
#     state⁺,
#     gradflux⁺,
#     aux⁺,
#     n⁻,
#     state⁻,
#     gradflux⁻,
#     aux⁻,
#     t,
# )
#     state⁺.ρθ = state⁻.ρθ
#     gradflux⁺.κ∇θ = n⁻ * bc.flux(state⁻, aux⁻, t)

#     return nothing
# end
