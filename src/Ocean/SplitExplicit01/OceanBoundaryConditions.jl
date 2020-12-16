abstract type OceanBoundaryCondition end

"""
    Defining dummy structs to dispatch on for boundary conditions.
"""
struct CoastlineFreeSlip <: OceanBoundaryCondition end
struct CoastlineNoSlip <: OceanBoundaryCondition end
struct OceanFloorFreeSlip <: OceanBoundaryCondition end
struct OceanFloorNoSlip <: OceanBoundaryCondition end
struct OceanFloorLinearDrag <: OceanBoundaryCondition end
struct OceanSurfaceNoStressNoForcing <: OceanBoundaryCondition end
struct OceanSurfaceStressNoForcing <: OceanBoundaryCondition end
struct OceanSurfaceNoStressForcing <: OceanBoundaryCondition end
struct OceanSurfaceStressForcing <: OceanBoundaryCondition end

# these functions just trim off the extra arguments
function ocean_model_boundary!(
    model::AbstractOceanModel,
    bc,
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    Q⁺,
    A⁺,
    n,
    Q⁻,
    A⁻,
    t,
    _...,
)
    return ocean_boundary_state!(model, bc, nf, Q⁺, A⁺, n, Q⁻, A⁻, t)
end

function ocean_model_boundary!(
    model::AbstractOceanModel,
    bc,
    nf::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n,
    Q⁻,
    D⁻,
    A⁻,
    t,
    _...,
)
    return ocean_boundary_state!(model, bc, nf, Q⁺, D⁺, A⁺, n, Q⁻, D⁻, A⁻, t)
end

"""
    CoastlineFreeSlip

applies boundary condition ∇u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineFreeSlip, ::NumericalFluxFirstOrder)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineFreeSlip,
    ::NumericalFluxFirstOrder,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    u⁻ = Q⁻.u
    n = @SVector [n⁻[1], n⁻[2]]

    # Q⁺.u = u⁻ - 2 * (n⋅u⁻) * n
    Q⁺.u = u⁻ - 2 * (n ∘ u⁻) * n

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineFreeSlip,
    ::NumericalFluxFirstOrder,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    U⁻ = Q⁻.U
    n = @SVector [n⁻[1], n⁻[2]]

    # Q⁺.U = U⁻ - 2 * (n⋅U⁻) * n
    Q⁺.U = U⁻ - 2 * (n ∘ U⁻) * n

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineFreeSlip, ::NumericalFluxGradient)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineFreeSlip,
    ::NumericalFluxGradient,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    u⁻ = Q⁻.u
    ud⁻ = A⁻.u_d
    n = @SVector [n⁻[1], n⁻[2]]

    # Q⁺.u = u⁻ - (n⋅u⁻) * n
    Q⁺.u = u⁻ - (n ∘ u⁻) * n
    A⁺.u_d = ud⁻ - (n ∘ ud⁻) * n

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineFreeSlip,
    ::NumericalFluxGradient,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    U⁻ = Q⁻.U
    n = @SVector [n⁻[1], n⁻[2]]

    # Q⁺.U = U⁻ - (n⋅U⁻) * n
    Q⁺.U = U⁻ - (n ∘ U⁻) * n

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineFreeSlip, ::NumericalFluxSecondOrder)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineFreeSlip,
    ::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineFreeSlip,
    ::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇U = n⁻ * (@SVector [-0, -0])'

    return nothing
end

"""
    CoastlineNoSlip

applies boundary condition u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineNoSlip, ::NumericalFluxFirstOrder)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineNoSlip,
    ::NumericalFluxFirstOrder,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    Q⁺.u = -Q⁻.u

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineNoSlip,
    ::NumericalFluxFirstOrder,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    Q⁺.U = -Q⁻.U

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineNoSlip, ::NumericalFluxGradient)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineNoSlip,
    ::NumericalFluxGradient,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    FT = eltype(Q⁺)
    Q⁺.u = SVector(-zero(FT), -zero(FT))
    A⁺.u_d = SVector(-zero(FT), -zero(FT))

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineNoSlip,
    ::NumericalFluxGradient,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    FT = eltype(Q⁺)
    Q⁺.U = SVector(-zero(FT), -zero(FT))

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineNoSlip, ::NumericalFluxSecondOrder)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineNoSlip,
    ::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇u = D⁻.ν∇u
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineNoSlip,
    ::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇U = D⁻.ν∇U

    return nothing
end

"""
    OceanFloorFreeSlip

applies boundary condition ∇u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorFreeSlip, ::NumericalFluxFirstOrder)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorFreeSlip,
    ::NumericalFluxFirstOrder,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    A⁺.w = -A⁻.w

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorFreeSlip, ::NumericalFluxGradient)

apply free slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorFreeSlip,
    ::NumericalFluxGradient,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    FT = eltype(Q⁺)
    A⁺.w = -zero(FT)

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorFreeSlip, ::NumericalFluxSecondOrder)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorFreeSlip,
    ::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    OceanFloorNoSlip

applies boundary condition u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::Union{OceanFloorNoSlip, OceanFloorLinearDrag}, ::NumericalFluxFirstOrder)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::Union{OceanFloorNoSlip, OceanFloorLinearDrag},
    ::NumericalFluxFirstOrder,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    Q⁺.u = -Q⁻.u
    A⁺.w = -A⁻.w

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::Union{OceanFloorNoSlip, OceanFloorLinearDrag}, ::NumericalFluxGradient)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::Union{OceanFloorNoSlip, OceanFloorLinearDrag},
    ::NumericalFluxGradient,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    FT = eltype(Q⁺)
    Q⁺.u = SVector(-zero(FT), -zero(FT))
    A⁺.w = -zero(FT)

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorNoSlip, ::NumericalFluxSecondOrder)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorNoSlip,
    ::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇u = D⁻.ν∇u
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    OceanFloorLinearDrag

applies boundary condition u = 0 with linear drag on viscous-flux and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorLinearDrag, ::NumericalFluxSecondOrder)

apply linear drag boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    m::AbstractOceanModel,
    ::OceanFloorLinearDrag,
    ::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    u, v = Q⁻.u

    D⁺.ν∇u = -m.problem.Cd_lin * @SMatrix [-0 -0; -0 -0; u v]
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::Union{OceanSurface*}, ::Union{NumericalFluxFirstOrder, NumericalFluxGradient})
applying neumann boundary conditions, so don't need to do anything for these numerical fluxes
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::Union{
        OceanSurfaceNoStressNoForcing,
        OceanSurfaceStressNoForcing,
        OceanSurfaceNoStressForcing,
        OceanSurfaceStressForcing,
    },
    ::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceNoStressNoForcing, ::NumericalFluxSecondOrder)

apply no flux boundary condition for velocity
apply no flux boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanSurfaceNoStressNoForcing,
    ::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceStressNoForcing, ::NumericalFluxSecondOrder)

apply wind-stress boundary condition for velocity
apply no flux boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    m::AbstractOceanModel,
    ::OceanSurfaceStressNoForcing,
    ::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    τᶻ = velocity_flux(m.problem, A⁻.y, m.ρₒ)
    D⁺.ν∇u = n⁻ * (@SVector [-τᶻ, -0])'
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceNoStressForcing, ::NumericalFluxSecondOrder)

apply no flux boundary condition for velocity
apply forcing boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    m::AbstractOceanModel,
    ::OceanSurfaceNoStressForcing,
    ::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    σᶻ = temperature_flux(m.problem, A⁻.y, Q⁻.θ)
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'
    D⁺.κ∇θ = -n⁻ * σᶻ

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceStressForcing, ::NumericalFluxSecondOrder)

apply wind-stress boundary condition for velocity
apply forcing boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    m::AbstractOceanModel,
    ::OceanSurfaceStressForcing,
    ::NumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    τᶻ = velocity_flux(m.problem, A⁻.y, m.ρₒ)
    σᶻ = temperature_flux(m.problem, A⁻.y, Q⁻.θ)
    D⁺.ν∇u = n⁻ * (@SVector [-τᶻ, -0])'
    D⁺.κ∇θ = -n⁻ * σᶻ

    return nothing
end

@inline velocity_flux(p::AbstractOceanProblem, y, ρ) =
    -(p.τₒ / ρ) * cos(y * π / p.Lʸ)

@inline function temperature_flux(p::AbstractOceanProblem, y, θ)
    θʳ = p.θᴱ * (1 - y / p.Lʸ)
    return p.λʳ * (θʳ - θ)
end
