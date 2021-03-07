"""
    cnse_boundary_state!(::NumericalFluxFirstOrder, ::Impenetrable{FreeSlip}, ::CNSE3D)
apply free slip boundary condition for velocity
sets reflective ghost point
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::Impenetrable{FreeSlip},
    ::CNSE3D,
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

    ρu⁻ = state⁻.ρu
    state⁺.ρu = ρu⁻ - 2 * n⁻ ⋅ ρu⁻ .* SVector(n⁻)

    return nothing
end

"""
    cnse_boundary_state!(::NumericalFluxGradient, ::Impenetrable{FreeSlip}, ::CNSE3D)
apply free slip boundary condition for velocity
sets non-reflective ghost point
"""
function cnse_boundary_state!(
    ::NumericalFluxGradient,
    ::Impenetrable{FreeSlip},
    ::CNSE3D,
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

    ρu⁻ = state⁻.ρu
    state⁺.ρu = ρu⁻ - n⁻ ⋅ ρu⁻ .* SVector(n⁻)

    return nothing
end

"""
    shallow_normal_boundary_flux_second_order!(::NumericalFluxSecondOrder, ::Impenetrable{FreeSlip}, ::CNSE3D)
apply free slip boundary condition for velocity
apply zero numerical flux in the normal direction
"""
function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{FreeSlip},
    ::CNSE3D,
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
    gradflux⁺.ν∇u = n⁻ * (@SVector [-0, -0, -0])'

    return nothing
end

"""
    cnse_boundary_state!(::NumericalFluxFirstOrder, ::Impenetrable{NoSlip}, ::CNSE3D)
apply no slip boundary condition for velocity
sets reflective ghost point
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::Impenetrable{NoSlip},
    ::CNSE3D,
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
    cnse_boundary_state!(::NumericalFluxGradient, ::Impenetrable{NoSlip}, ::CNSE3D)
apply no slip boundary condition for velocity
set numerical flux to zero for U
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxGradient,
    ::Impenetrable{NoSlip},
    ::CNSE3D,
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
    state⁺.ρu = @SVector zeros(FT, 3)

    return nothing
end

"""
    cnse_boundary_state!(::NumericalFluxSecondOrder, ::Impenetrable{NoSlip}, ::CNSE3D)
apply no slip boundary condition for velocity
sets ghost point to have no numerical flux on the boundary for U
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{NoSlip},
    ::CNSE3D,
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
    cnse_boundary_state!(::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, ::Penetrable{FreeSlip}, ::CNSE3D)
no mass boundary condition for penetrable
"""
cnse_boundary_state!(
    ::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::Penetrable{FreeSlip},
    ::CNSE3D,
    ::ConstantViscosity,
    _...,
) = nothing

"""
    cnse_boundary_state!(::NumericalFluxSecondOrder, ::Penetrable{FreeSlip}, ::CNSE3D)
apply free slip boundary condition for velocity
apply zero numerical flux in the normal direction
"""
function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Penetrable{FreeSlip},
    ::CNSE3D,
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
    gradflux⁺.ν∇u = n⁻ * (@SVector [-0, -0, -0])'

    return nothing
end

"""
    cnse_boundary_state!(::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, ::Impenetrable{MomentumFlux}, ::HBModel)
apply kinematic stress boundary condition for velocity
applies free slip conditions for first-order and gradient fluxes
"""
function cnse_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::Impenetrable{<:MomentumFlux},
    model::CNSE3D,
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
apply kinematic stress boundary condition for velocity
sets ghost point to have specified flux on the boundary for ν∇u
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    bc::Impenetrable{<:MomentumFlux},
    model::CNSE3D,
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
    gradflux⁺.ν∇u = n⁻ * bc.drag.flux(state⁻, aux⁻, t)'

    return nothing
end

"""
    cnse_boundary_state!(::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, ::Penetrable{MomentumFlux}, ::HBModel)
apply kinematic stress boundary condition for velocity
applies free slip conditions for first-order and gradient fluxes
"""
function cnse_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::Penetrable{<:MomentumFlux},
    model::CNSE3D,
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
apply kinematic stress boundary condition for velocity
sets ghost point to have specified flux on the boundary for ν∇u
"""
@inline function cnse_boundary_state!(
    ::NumericalFluxSecondOrder,
    bc::Penetrable{<:MomentumFlux},
    shallow::CNSE3D,
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
    gradflux⁺.ν∇u = n⁻ * bc.drag.flux(state⁻, aux⁻, t)'

    return nothing
end
