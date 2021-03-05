"""
    cnse_boundary_state!(::NumericalFluxFirstOrder, ::Impenetrable{FreeSlip}, ::CNSE2D)

apply free slip boundary condition for momentum
sets reflective ghost point
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::Impenetrable{FreeSlip},
    ::CNSE2D,
    ::TurbulenceClosure,
    state⁺,
    aux⁺,
    n⁻,
    state⁻,
    aux⁻,
    t,
    args...,
)
    state⁺.ρ = state⁻.ρ

    ρu⁻ = @SVector [state⁻.ρu[1], state⁻.ρu[2], -0]
    ρu⁺ = ρu⁻ - 2 * n⁻ ⋅ ρu⁻ .* SVector(n⁻)

    state⁺.ρu = @SVector [ρu⁺[1], ρu⁺[2]]

    return nothing
end

"""
    cnse_boundary_state!(::Union{NumericalFluxGradient, NumericalFluxSecondOrder}, ::Impenetrable{FreeSlip}, ::CNSE2D)

no second order flux computed for linear drag
"""
cnse_boundary_state!(
    ::Union{NumericalFluxGradient, NumericalFluxSecondOrder},
    ::MomentumBC,
    ::CNSE2D,
    ::LinearDrag,
    _...,
) = nothing

"""
    cnse_boundary_state!(::NumericalFluxGradient, ::Impenetrable{FreeSlip}, ::CNSE2D)

apply free slip boundary condition for momentum
sets non-reflective ghost point
"""
function cnse_boundary_state!(
    ::NumericalFluxGradient,
    ::Impenetrable{FreeSlip},
    ::CNSE2D,
    ::ConstantViscosity,
    state⁺,
    aux⁺,
    n⁻,
    state⁻,
    aux⁻,
    t,
    args...,
)
    state⁺.ρ = state⁻.ρ

    ρu⁻ = @SVector [state⁻.ρu[1], state⁻.ρu[2], -0]
    ρu⁺ = ρu⁻ - n⁻ ⋅ ρu⁻ .* SVector(n⁻)

    state⁺.ρu = @SVector [ρu⁺[1], ρu⁺[2]]

    return nothing
end

"""
    shallow_normal_boundary_flux_second_order!(::NumericalFluxSecondOrder, ::Impenetrable{FreeSlip}, ::CNSE2D)

apply free slip boundary condition for momentum
apply zero numerical flux in the normal direction
"""
function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{FreeSlip},
    ::CNSE2D,
    ::ConstantViscosity,
    state⁺,
    gradflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    aux⁻,
    t,
    args...,
)
    state⁺.ρu = state⁻.ρu
    gradflux⁺.ν∇u = n⁻ * (@SVector [-0, -0])'

    return nothing
end

"""
    cnse_boundary_state!(::NumericalFluxFirstOrder, ::Impenetrable{NoSlip}, ::CNSE2D)

apply no slip boundary condition for momentum
sets reflective ghost point
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::Impenetrable{NoSlip},
    ::CNSE2D,
    ::TurbulenceClosure,
    state⁺,
    aux⁺,
    n⁻,
    state⁻,
    aux⁻,
    t,
    args...,
)
    state⁺.ρ = state⁻.ρ
    state⁺.ρu = -state⁻.ρu

    return nothing
end

"""
    cnse_boundary_state!(::NumericalFluxGradient, ::Impenetrable{NoSlip}, ::CNSE2D)

apply no slip boundary condition for momentum
set numerical flux to zero for U
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxGradient,
    ::Impenetrable{NoSlip},
    ::CNSE2D,
    ::ConstantViscosity,
    state⁺,
    aux⁺,
    n⁻,
    state⁻,
    aux⁻,
    t,
    args...,
)
    FT = eltype(state⁺)
    state⁺.ρu = @SVector zeros(FT, 2)

    return nothing
end

"""
    cnse_boundary_state!(::NumericalFluxSecondOrder, ::Impenetrable{NoSlip}, ::CNSE2D)

apply no slip boundary condition for momentum
sets ghost point to have no numerical flux on the boundary for U
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{NoSlip},
    ::CNSE2D,
    ::ConstantViscosity,
    state⁺,
    gradflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    aux⁻,
    t,
    args...,
)
    state⁺.ρu = -state⁻.ρu
    gradflux⁺.ν∇u = gradflux⁻.ν∇u

    return nothing
end

"""
    cnse_boundary_state!(::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, ::Penetrable{FreeSlip}, ::CNSE2D)

no mass boundary condition for penetrable
"""
cnse_boundary_state!(
    ::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::Penetrable{FreeSlip},
    ::CNSE2D,
    ::ConstantViscosity,
    _...,
) = nothing

"""
    cnse_boundary_state!(::NumericalFluxSecondOrder, ::Penetrable{FreeSlip}, ::CNSE2D)

apply free slip boundary condition for momentum
apply zero numerical flux in the normal direction
"""
function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Penetrable{FreeSlip},
    ::CNSE2D,
    ::ConstantViscosity,
    state⁺,
    gradflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    aux⁻,
    t,
    args...,
)
    state⁺.ρu = state⁻.ρu
    gradflux⁺.ν∇u = n⁻ * (@SVector [-0, -0])'

    return nothing
end

"""
    cnse_boundary_state!(::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, ::Impenetrable{MomentumFlux}, ::HBModel)

apply kinematic stress boundary condition for momentum
applies free slip conditions for first-order and gradient fluxes
"""
function cnse_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::Impenetrable{<:MomentumFlux},
    model::CNSE2D,
    turb::TurbulenceClosure,
    args...,
)
    return cnse_boundary_state!(
        nf,
        Impenetrable(FreeSlip()),
        model,
        turb,
        args...,
    )
end

"""
    cnse_boundary_state!(::NumericalFluxSecondOrder, ::Impenetrable{MomentumFlux}, ::HBModel)

apply kinematic stress boundary condition for momentum
sets ghost point to have specified flux on the boundary for ν∇u
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{<:MomentumFlux},
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
    state⁺.ρu = state⁻.ρu
    gradflux⁺.ν∇u = n⁻ * bc.drag.flux(state⁻, aux⁻, t)'

    return nothing
end

"""
    cnse_boundary_state!(::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, ::Penetrable{MomentumFlux}, ::HBModel)

apply kinematic stress boundary condition for momentum
applies free slip conditions for first-order and gradient fluxes
"""
function cnse_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::Penetrable{<:MomentumFlux},
    model::CNSE2D,
    turb::TurbulenceClosure,
    args...,
)
    return cnse_boundary_state!(
        nf,
        Penetrable(FreeSlip()),
        model,
        turb,
        args...,
    )
end

"""
    cnse_boundary_state!(::NumericalFluxSecondOrder, ::Penetrable{MomentumFlux}, ::HBModel)

apply kinematic stress boundary condition for momentum
sets ghost point to have specified flux on the boundary for ν∇u
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    bc::Penetrable{<:MomentumFlux},
    shallow::CNSE2D,
    ::TurbulenceClosure,
    state⁺,
    gradflux⁺,
    aux⁺,
    n⁻,
    state⁻,
    gradflux⁻,
    aux⁻,
    t,
)
    state⁺.ρu = state⁻.ρu
    gradflux⁺.ν∇u = n⁻ * bc.drag.flux(state⁻, aux⁻, t)'

    return nothing
end
