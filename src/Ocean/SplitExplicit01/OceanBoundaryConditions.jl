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
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineFreeSlip, ::Union{RusanovNumericalFlux, CentralNumericalFluxFirstOrder})

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineFreeSlip,
    ::Union{RusanovNumericalFlux, CentralNumericalFluxFirstOrder},
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
    ::Union{RusanovNumericalFlux, CentralNumericalFluxFirstOrder},
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
    Q⁺.U = U⁻ - (n ∘ U⁻) * n

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineFreeSlip, ::CentralNumericalFluxSecondOrder)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineFreeSlip,
    ::CentralNumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    #   D⁺.ν∇u = -D⁻.ν∇u
    #   D⁺.κ∇θ = -D⁻.κ∇θ
    #-  new diffusive flux BC:
    Q⁺.u = Q⁻.u
    A⁺.u_d = A⁻.u_d
    A⁺.w = A⁻.w
    Q⁺.θ = Q⁻.θ
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineFreeSlip,
    ::CentralNumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    #   D⁺.ν∇U = -D⁻.ν∇U
    #-  new diffusive flux BC:
    Q⁺.U = Q⁻.U
    D⁺.ν∇U = n⁻ * (@SVector [-0, -0])'

    return nothing
end

"""
    CoastlineNoSlip

applies boundary condition u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineNoSlip, ::RusanovNumericalFlux)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineNoSlip,
    ::Union{RusanovNumericalFlux, CentralNumericalFluxFirstOrder},
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
    ::Union{RusanovNumericalFlux, CentralNumericalFluxFirstOrder},
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
    A⁺.u_d = SVector(-zero(FT), -zero(FT))

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
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineNoSlip, ::CentralNumericalFluxSecondOrder)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::CoastlineNoSlip,
    ::CentralNumericalFluxSecondOrder,
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
    A⁺.u_d = -A⁻.u_d

    #   D⁺.κ∇θ = -D⁻.κ∇θ

    #-  new diffusive flux BC:
    #   Q⁺.u = -Q⁻.u
    #   A⁺.w = -A⁻.w
    Q⁺.θ = Q⁻.θ
    D⁺.ν∇u = D⁻.ν∇u
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

@inline function ocean_boundary_state!(
    ::BarotropicModel,
    ::CoastlineNoSlip,
    ::CentralNumericalFluxSecondOrder,
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
    #-  new diffusive flux BC:
    D⁺.ν∇U = D⁻.ν∇U

    return nothing
end

"""
    OceanFloorFreeSlip

applies boundary condition ∇u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorFreeSlip, ::RusanovNumericalFlux)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorFreeSlip,
    ::Union{RusanovNumericalFlux, CentralNumericalFluxFirstOrder},
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
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorFreeSlip, ::CentralNumericalFluxSecondOrder)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorFreeSlip,
    ::CentralNumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    #   A⁺.w = -A⁻.w
    #   D⁺.ν∇u = -D⁻.ν∇u

    #   D⁺.κ∇θ = -D⁻.κ∇θ

    #-  new diffusive flux BC:
    Q⁺.u = Q⁻.u
    A⁺.u_d = A⁻.u_d
    A⁺.w = A⁻.w
    Q⁺.θ = Q⁻.θ
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    OceanFloorNoSlip

applies boundary condition u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorNoSlip, ::RusanovNumericalFlux)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorNoSlip,
    ::Union{RusanovNumericalFlux, CentralNumericalFluxFirstOrder},
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
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorNoSlip, ::CentralNumericalFluxSecondOrder)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanFloorNoSlip,
    ::CentralNumericalFluxSecondOrder,
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

    #   D⁺.κ∇θ = -D⁻.κ∇θ

    #-  new diffusive flux BC:
    #   Q⁺.u = -Q⁻.u
    #   A⁺.w = -A⁻.w
    Q⁺.θ = Q⁻.θ
    D⁺.ν∇u = D⁻.ν∇u
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::Union{OceanSurface*}, ::Union{RusanovNumericalFlux, CentralNumericalFluxGradient})

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
    ::Union{
        RusanovNumericalFlux,
        CentralNumericalFluxFirstOrder,
        CentralNumericalFluxGradient,
    },
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
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceNoStressNoForcing, ::CentralNumericalFluxSecondOrder)

apply no flux boundary condition for velocity
apply no flux boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    ::AbstractOceanModel,
    ::OceanSurfaceNoStressNoForcing,
    ::CentralNumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    #   D⁺.ν∇u = -D⁻.ν∇u

    #   D⁺.κ∇θ = -D⁻.κ∇θ

    #-  new diffusive flux BC:
    Q⁺.u = Q⁻.u
    A⁺.u_d = A⁻.u_d
    A⁺.w = A⁻.w
    Q⁺.θ = Q⁻.θ
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceStressNoForcing, ::CentralNumericalFluxSecondOrder)

apply wind-stress boundary condition for velocity
apply no flux boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    m::AbstractOceanModel,
    ::OceanSurfaceStressNoForcing,
    ::CentralNumericalFluxSecondOrder,
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
    #   τ = @SMatrix [-0 -0; -0 -0; τᶻ -0]
    #   D⁺.ν∇u = -D⁻.ν∇u + 2 * τ

    #   D⁺.κ∇θ = -D⁻.κ∇θ

    #-  new diffusive flux BC:
    Q⁺.u = Q⁻.u
    A⁺.u_d = A⁻.u_d
    A⁺.w = A⁻.w
    Q⁺.θ = Q⁻.θ
    D⁺.ν∇u = n⁻ * (@SVector [τᶻ, -0])'
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceNoStressForcing, ::CentralNumericalFluxSecondOrder)

apply no flux boundary condition for velocity
apply forcing boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    m::AbstractOceanModel,
    ::OceanSurfaceNoStressForcing,
    ::CentralNumericalFluxSecondOrder,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    #   D⁺.ν∇u = -D⁻.ν∇u

    σᶻ = temperature_flux(m.problem, A⁻.y, Q⁻.θ)
    #   σ = @SVector [-0, -0, σᶻ]
    #   D⁺.κ∇θ = -D⁻.κ∇θ + 2 * σ

    #-  new diffusive flux BC:
    Q⁺.u = Q⁻.u
    A⁺.u_d = A⁻.u_d
    A⁺.w = A⁻.w
    Q⁺.θ = Q⁻.θ
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'
    D⁺.κ∇θ = n⁻ * σᶻ

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceStressForcing, ::CentralNumericalFluxSecondOrder)

apply wind-stress boundary condition for velocity
apply forcing boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    m::AbstractOceanModel,
    ::OceanSurfaceStressForcing,
    ::CentralNumericalFluxSecondOrder,
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
    #   τ = @SMatrix [-0 -0; -0 -0; τᶻ -0]
    #   D⁺.ν∇u = -D⁻.ν∇u + 2 * τ

    σᶻ = temperature_flux(m.problem, A⁻.y, Q⁻.θ)
    #   σ = @SVector [-0, -0, σᶻ]
    #   D⁺.κ∇θ = -D⁻.κ∇θ + 2 * σ

    #-  new diffusive flux BC:
    Q⁺.u = Q⁻.u
    A⁺.u_d = A⁻.u_d
    A⁺.w = A⁻.w
    Q⁺.θ = Q⁻.θ
    D⁺.ν∇u = n⁻ * (@SVector [τᶻ, -0])'
    D⁺.κ∇θ = n⁻ * σᶻ

    return nothing
end

@inline velocity_flux(p::AbstractOceanProblem, y, ρ) =
    -(p.τₒ / ρ) * cos(y * π / p.Lʸ)

@inline function temperature_flux(p::AbstractOceanProblem, y, θ)
    θʳ = p.θᴱ * (1 - y / p.Lʸ)
    return p.λʳ * (θʳ - θ)
end
