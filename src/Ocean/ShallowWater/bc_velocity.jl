"""
    ocean_boundary_state!(::NumericalFluxFirstOrder, ::Impenetrable{FreeSlip}, ::SWModel)

apply free slip boundary condition for velocity
sets reflective ghost point
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::Impenetrable{FreeSlip},
    ::SWModel,
    ::TurbulenceClosure,
    q⁺,
    a⁺,
    n⁻,
    q⁻,
    a⁻,
    t,
    args...,
)
    q⁺.η = q⁻.η

    V⁻ = @SVector [q⁻.U[1], q⁻.U[2], -0]
    V⁺ = V⁻ - 2 * n⁻ ⋅ V⁻ .* SVector(n⁻)
    q⁺.U = @SVector [V⁺[1], V⁺[2]]

    return nothing
end

"""
    ocean_boundary_state!(::Union{NumericalFluxGradient, NumericalFluxSecondOrder}, ::Impenetrable{FreeSlip}, ::SWModel)

no second order flux computed for linear drag
"""
ocean_boundary_state!(
    ::Union{NumericalFluxGradient, NumericalFluxSecondOrder},
    ::VelocityBC,
    ::SWModel,
    ::LinearDrag,
    _...,
) = nothing

"""
    ocean_boundary_state!(::NumericalFluxGradient, ::Impenetrable{FreeSlip}, ::SWModel)

apply free slip boundary condition for velocity
sets non-reflective ghost point
"""
function ocean_boundary_state!(
    ::NumericalFluxGradient,
    ::Impenetrable{FreeSlip},
    ::SWModel,
    ::ConstantViscosity,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
    args...,
)
    V⁻ = @SVector [Q⁻.U[1], Q⁻.U[2], -0]
    V⁺ = V⁻ - n⁻ ⋅ V⁻ .* SVector(n⁻)
    Q⁺.U = @SVector [V⁺[1], V⁺[2]]

    return nothing
end

"""
    shallow_normal_boundary_flux_second_order!(::NumericalFluxSecondOrder, ::Impenetrable{FreeSlip}, ::SWModel)

apply free slip boundary condition for velocity
apply zero numerical flux in the normal direction
"""
function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{FreeSlip},
    ::SWModel,
    ::ConstantViscosity,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
    args...,
)
    Q⁺.U = Q⁻.U
    D⁺.ν∇U = n⁻ * (@SVector [-0, -0])'

    return nothing
end

"""
    ocean_boundary_state!(::NumericalFluxFirstOrder, ::Impenetrable{NoSlip}, ::SWModel)

apply no slip boundary condition for velocity
sets reflective ghost point
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::Impenetrable{NoSlip},
    ::SWModel,
    ::TurbulenceClosure,
    q⁺,
    α⁺,
    n⁻,
    q⁻,
    α⁻,
    t,
    args...,
)
    q⁺.η = q⁻.η
    q⁺.U = -q⁻.U

    return nothing
end

"""
    ocean_boundary_state!(::NumericalFluxGradient, ::Impenetrable{NoSlip}, ::SWModel)

apply no slip boundary condition for velocity
set numerical flux to zero for U
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxGradient,
    ::Impenetrable{NoSlip},
    ::SWModel,
    ::ConstantViscosity,
    q⁺,
    α⁺,
    n⁻,
    q⁻,
    α⁻,
    t,
    args...,
)
    FT = eltype(q⁺)
    q⁺.U = @SVector zeros(FT, 2)

    return nothing
end

"""
    ocean_boundary_state!(::NumericalFluxSecondOrder, ::Impenetrable{NoSlip}, ::SWModel)

apply no slip boundary condition for velocity
sets ghost point to have no numerical flux on the boundary for U
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{NoSlip},
    ::SWModel,
    ::ConstantViscosity,
    q⁺,
    σ⁺,
    α⁺,
    n⁻,
    q⁻,
    σ⁻,
    α⁻,
    t,
    args...,
)
    q⁺.U = -q⁻.U
    σ⁺.ν∇U = σ⁻.ν∇U

    return nothing
end

"""
    ocean_boundary_state!(::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, ::Penetrable{FreeSlip}, ::SWModel)

no mass boundary condition for penetrable
"""
ocean_boundary_state!(
    ::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::Penetrable{FreeSlip},
    ::SWModel,
    ::ConstantViscosity,
    _...,
) = nothing

"""
    ocean_boundary_state!(::NumericalFluxSecondOrder, ::Penetrable{FreeSlip}, ::SWModel)

apply free slip boundary condition for velocity
apply zero numerical flux in the normal direction
"""
function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Penetrable{FreeSlip},
    ::SWModel,
    ::ConstantViscosity,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
    args...,
)
    Q⁺.U = Q⁻.U
    D⁺.ν∇U = n⁻ * (@SVector [-0, -0])'

    return nothing
end

"""
    ocean_boundary_state!(::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, ::Impenetrable{KinematicStress}, ::HBModel)

apply kinematic stress boundary condition for velocity
applies free slip conditions for first-order and gradient fluxes
"""
function ocean_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::Impenetrable{KinematicStress},
    shallow::SWModel,
    turb::TurbulenceClosure,
    args...,
)
    return ocean_boundary_state!(
        nf,
        Impenetrable(FreeSlip()),
        shallow,
        turb,
        args...,
    )
end

"""
    ocean_boundary_state!(::NumericalFluxSecondOrder, ::Impenetrable{KinematicStress}, ::HBModel)

apply kinematic stress boundary condition for velocity
sets ghost point to have specified flux on the boundary for ν∇u
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{KinematicStress},
    shallow::SWModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    Q⁺.U = Q⁻.U
    D⁺.ν∇U = n⁻ * kinematic_stress(shallow.problem, A⁻.y, 1000)'
    # applies windstress for now, will be fixed in a later PR

    return nothing
end

"""
    ocean_boundary_state!(::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, ::Penetrable{KinematicStress}, ::HBModel)

apply kinematic stress boundary condition for velocity
applies free slip conditions for first-order and gradient fluxes
"""
function ocean_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::Penetrable{KinematicStress},
    shallow::SWModel,
    turb::TurbulenceClosure,
    args...,
)
    return ocean_boundary_state!(
        nf,
        Penetrable(FreeSlip()),
        shallow,
        turb,
        args...,
    )
end

"""
    ocean_boundary_state!(::NumericalFluxSecondOrder, ::Penetrable{KinematicStress}, ::HBModel)

apply kinematic stress boundary condition for velocity
sets ghost point to have specified flux on the boundary for ν∇u
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Penetrable{KinematicStress},
    shallow::SWModel,
    ::TurbulenceClosure,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    Q⁺.u = Q⁻.u
    D⁺.ν∇u = n⁻ * kinematic_stress(shallow.problem, A⁻.y, 1000)'
    # applies windstress for now, will be fixed in a later PR

    return nothing
end
