abstract type OceanBoundaryCondition end

"""
    Defining dummy structs to dispatch on for boundary conditions.
"""
struct CoastlineFreeSlip <: OceanBoundaryCondition end
struct CoastlineNoSlip <: OceanBoundaryCondition end
struct OceanFloorFreeSlip <: OceanBoundaryCondition end
struct OceanFloorNoSlip <: OceanBoundaryCondition end
struct OceanSurfaceNoStressNoForcing <: OceanBoundaryCondition end
struct OceanSurfaceStressNoForcing <: OceanBoundaryCondition end
struct OceanSurfaceNoStressForcing <: OceanBoundaryCondition end
struct OceanSurfaceStressForcing <: OceanBoundaryCondition end

"""
    CoastlineFreeSlip

applies boundary condition ∇u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineFreeSlip, ::Union{Rusanov, CentralNumericalFluxNonDiffusive})

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineFreeSlip,
    ::Union{Rusanov, CentralNumericalFluxNonDiffusive},
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
    Q⁺.u = u⁻ - 2 * (n∘u⁻) * n

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineFreeSlip,
    ::Union{Rusanov, CentralNumericalFluxNonDiffusive},
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
    Q⁺.U = U⁻ - 2 * (n∘U⁻) * n

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineFreeSlip, ::CentralNumericalFluxGradient)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineFreeSlip,
    ::CentralNumericalFluxGradient,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    u⁻ = Q⁻.u
    n = @SVector [n⁻[1], n⁻[2]]

    # Q⁺.u = u⁻ - (n⋅u⁻) * n
    Q⁺.u = u⁻ - (n∘u⁻) * n

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineFreeSlip,
    ::CentralNumericalFluxGradient,
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
    Q⁺.U = U⁻ - (n∘U⁻) * n

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineFreeSlip, ::CentralNumericalFluxDiffusive)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineFreeSlip,
    ::CentralNumericalFluxDiffusive,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇u = -D⁻.ν∇u
    D⁺.κ∇θ = -D⁻.κ∇θ

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineFreeSlip,
    ::CentralNumericalFluxDiffusive,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇U = -D⁻.ν∇U

    return nothing
end

"""
    CoastlineNoSlip

applies boundary condition u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineNoSlip, ::Rusanov)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineNoSlip,
    ::Union{Rusanov, CentralNumericalFluxNonDiffusive},
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
    ::Union{Rusanov, CentralNumericalFluxNonDiffusive},
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
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineNoSlip, ::CentralNumericalFluxGradient)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineNoSlip,
    ::CentralNumericalFluxGradient,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    FT = eltype(Q⁺)
    Q⁺.u = SVector(-zero(FT), -zero(FT))

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineNoSlip,
    ::CentralNumericalFluxGradient,
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
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineNoSlip, ::CentralNumericalFluxDiffusive)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineNoSlip,
    ::CentralNumericalFluxDiffusive,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    Q⁺.u = -Q⁻.u

    D⁺.κ∇θ = -D⁻.κ∇θ

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineNoSlip,
    ::CentralNumericalFluxDiffusive,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    Q⁺.U = -Q⁻.U

    return nothing
end

"""
    OceanFloorFreeSlip

applies boundary condition ∇u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorFreeSlip, ::Rusanov)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorFreeSlip,
    ::Union{Rusanov, CentralNumericalFluxNonDiffusive},
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
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorFreeSlip, ::CentralNumericalFluxGradient)

apply free slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorFreeSlip,
    ::CentralNumericalFluxGradient,
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
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorFreeSlip, ::CentralNumericalFluxDiffusive)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorFreeSlip,
    ::CentralNumericalFluxDiffusive,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    A⁺.w = -A⁻.w
    D⁺.ν∇u = -D⁻.ν∇u

    D⁺.κ∇θ = -D⁻.κ∇θ

    return nothing
end

"""
    OceanFloorNoSlip

applies boundary condition u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorNoSlip, ::Rusanov)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorNoSlip,
    ::Union{Rusanov, CentralNumericalFluxNonDiffusive},
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
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorNoSlip, ::CentralNumericalFluxGradient)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorNoSlip,
    ::CentralNumericalFluxGradient,
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
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorNoSlip, ::CentralNumericalFluxDiffusive)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorNoSlip,
    ::CentralNumericalFluxDiffusive,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)

    Q⁺.u = -Q⁻.u
    A⁺.w = -A⁻.w

    D⁺.κ∇θ = -D⁻.κ∇θ

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::Union{OceanSurface*}, ::Union{Rusanov, CentralNumericalFluxGradient})

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
    ::Union{Rusanov, CentralNumericalFluxNonDiffusive, CentralNumericalFluxGradient},
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
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceNoStressNoForcing, ::CentralNumericalFluxDiffusive)

apply no flux boundary condition for velocity
apply no flux boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanSurfaceNoStressNoForcing,
    ::CentralNumericalFluxDiffusive,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇u = -D⁻.ν∇u

    D⁺.κ∇θ = -D⁻.κ∇θ

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceStressNoForcing, ::CentralNumericalFluxDiffusive)

apply wind-stress boundary condition for velocity
apply no flux boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    m::AbstractOceanModel,
    ::OceanSurfaceStressNoForcing,
    ::CentralNumericalFluxDiffusive,
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
    τ = @SMatrix [-0 -0; -0 -0; τᶻ -0]
    D⁺.ν∇u = -D⁻.ν∇u + 2 * τ

    D⁺.κ∇θ = -D⁻.κ∇θ

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceNoStressForcing, ::CentralNumericalFluxDiffusive)

apply no flux boundary condition for velocity
apply forcing boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    m::AbstractOceanModel,
    ::OceanSurfaceNoStressForcing,
    ::CentralNumericalFluxDiffusive,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇u = -D⁻.ν∇u

    σᶻ = temperature_flux(m.problem, A⁻.y, Q⁻.θ)
    σ = @SVector [-0, -0, σᶻ]
    D⁺.κ∇θ = -D⁻.κ∇θ + 2 * σ

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceStressForcing, ::CentralNumericalFluxDiffusive)

apply wind-stress boundary condition for velocity
apply forcing boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    m::AbstractOceanModel,
    ::OceanSurfaceStressForcing,
    ::CentralNumericalFluxDiffusive,
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
    τ = @SMatrix [-0 -0; -0 -0; τᶻ -0]
    D⁺.ν∇u = -D⁻.ν∇u + 2 * τ

    σᶻ = temperature_flux(m.problem, A⁻.y, Q⁻.θ)
    σ = @SVector [-0, -0, σᶻ]
    D⁺.κ∇θ = -D⁻.κ∇θ + 2 * σ

    return nothing
end

@inline velocity_flux(p::AbstractOceanProblem, y, ρ) =
    -(p.τₒ / ρ) * cos(y * π / p.Lʸ)

@inline function temperature_flux(p::AbstractOceanProblem, y, θ)
    θʳ = p.θᴱ * (1 - y / p.Lʸ)
    return p.λʳ * (θʳ - θ)
end
