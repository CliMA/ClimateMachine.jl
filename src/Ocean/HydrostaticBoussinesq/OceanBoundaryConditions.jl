using ..DGmethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder

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

applies boundary condition ν∇u = 0 and κ∇θ = 0
"""

"""
    ocean_boundary_state!(::HBModel, ::CoastlineFreeSlip, ::Union{RusanovNumericalFlux, CentralNumericalFluxGradient})

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature

# Arguments
- `Q⁺`: state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::RusanovNumericalFlux,
    ::CoastlineFreeSlip,
    ::HBModel,
    states,
    n⁻,
    t,
)
    u⁻ = states.conservative⁻.u
    u⁺ = states.conservative⁺.u
    n = @SVector [n⁻[1], n⁻[2]]

    u⁺ = u⁻ - 2 * (n ⋅ u⁻) * n

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::CoastlineFreeSlip, ::CentralNumericalFluxGradient)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature

# Arguments
- `Q⁺`: state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxGradient,
    ::CoastlineFreeSlip,
    ::HBModel,
    states,
    n⁻,
    t,
)
    u⁻ = states.conservative⁻.u
    u⁺ = states.conservative⁺.u
    n = @SVector [n⁻[1], n⁻[2]]

    u⁺ = u⁻ - (n ⋅ u⁻) * n

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::CoastlineFreeSlip, ::CentralNumericalFluxSecondOrder)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
sets ghost point to have no numerical flux on the boundary for ν∇u and κ∇θ

# Arguments
- `Q⁺`: state vector at ghost point
- `D⁺`: diffusive state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `D⁻`: diffusive state vector at interior point
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    ::CoastlineFreeSlip,
    ::HBModel,
    states,
    n⁻,
    t,
)
    ν∇u⁻ = states.gradient_flux⁻.ν∇u
    κ∇θ⁻ = states.gradient_flux⁻.κ∇θ
    ν∇u⁺ = states.gradient_flux⁺.ν∇u
    κ∇θ⁺ = states.gradient_flux⁺.κ∇θ

    ν∇u⁺ = -ν∇u⁻
    κ∇θ⁺ = -κ∇θ⁻

    return nothing
end

"""
    CoastlineNoSlip

applies boundary condition u = 0 and κ∇θ = 0
"""

"""
    ocean_boundary_state!(::HBModel, ::CoastlineNoSlip, ::RusanovNumericalFlux)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
set sets ghost point to have no numerical flux on the boundary for u

# Arguments
- `Q⁺`: state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::RusanovNumericalFlux,
    ::CoastlineNoSlip,
    ::HBModel,
    states,
    n⁻,
    t,
)
    u⁻ = states.conservative⁻.u
    u⁺ = states.conservative⁺.u

    u⁺ = -u⁻

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::CoastlineNoSlip, ::CentralNumericalFluxGradient)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
set numerical flux to zero for u

# Arguments
- `Q⁺`: state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxGradient,
    ::CoastlineNoSlip,
    ::HBModel,
    states,
    n⁻,
    t,
)
    u⁺ = states.conservative⁺.u
    FT = eltype(u⁺)

    u⁺ = SVector(-zero(FT), -zero(FT))

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::CoastlineNoSlip, ::CentralNumericalFluxSecondOrder)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
sets ghost point to have no numerical flux on the boundary for u and κ∇θ

# Arguments
- `Q⁺`: state vector at ghost point
- `D⁺`: diffusive state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `D⁻`: diffusive state vector at interior point
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    ::CoastlineNoSlip,
    ::HBModel,
    states,
    n⁻,
    t,
)
    u⁻ = states.conservative⁻.u
    u⁺ = states.conservative⁺.u
    κ∇θ⁻ = states.gradient_flux⁻.κ∇θ
    κ∇θ⁺ = states.gradient_flux⁺.κ∇θ

    u⁺ = -u⁻
    κ∇θ⁺ = -κ∇θ⁻

    return nothing
end

"""
    OceanFloorFreeSlip

applies boundary condition ν∇u = 0 and κ∇θ = 0
"""

"""
    ocean_boundary_state!(::HBModel, ::OceanFloorFreeSlip, ::RusanovNumericalFlux)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
set ghost point to have no numerical flux on the boundary for w

# Arguments
- `Q⁺`: state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::RusanovNumericalFlux,
    ::OceanFloorFreeSlip,
    ::HBModel,
    states,
    n⁻,
    t,
)
    w⁻ = states.auxiliary⁻.w
    w⁺ = states.auxiliary⁺.w

    w⁺ = -w⁻

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::OceanFloorFreeSlip, ::CentralNumericalFluxGradient)

apply free slip boundary condition for velocity
apply no penetration boundary for temperature
set numerical flux to zero for w

# Arguments
- `Q⁺`: state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxGradient,
    ::OceanFloorFreeSlip,
    ::HBModel,
    states,
    n⁻,
    t,
)
    w⁺ = states.auxiliary⁺.w
    FT = eltype(w⁺)

    w⁺ = -zero(FT)

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::OceanFloorFreeSlip, ::CentralNumericalFluxSecondOrder)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
sets ghost point to have no numerical flux on the boundary for ν∇u and κ∇θ

# Arguments
- `Q⁺`: state vector at ghost point
- `D⁺`: diffusive state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `D⁻`: diffusive state vector at interior point
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    ::OceanFloorFreeSlip,
    ::HBModel,
    states,
    n⁻,
    t,
)
    w⁻ = states.auxiliary⁻.w
    w⁺ = states.auxiliary⁺.w
    ν∇u⁻ = states.gradient_flux⁻.ν∇u
    κ∇θ⁻ = states.gradient_flux⁻.κ∇θ
    ν∇u⁺ = states.gradient_flux⁺.ν∇u
    κ∇θ⁺ = states.gradient_flux⁺.κ∇θ

    w⁺ = -w⁻
    ν∇u⁺ = -ν∇u⁻
    κ∇θ⁺ = -κ∇θ⁻

    return nothing
end

"""
    OceanFloorNoSlip

applies boundary condition u = 0 and κ∇θ = 0
"""

"""
    ocean_boundary_state!(::HBModel, ::OceanFloorNoSlip, ::RusanovNumericalFlux)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
set sets ghost point to have no numerical flux on the boundary for u and w

# Arguments
- `Q⁺`: state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::RusanovNumericalFlux,
    ::OceanFloorNoSlip,
    ::HBModel,
    states,
    n⁻,
    t,
)
    u⁻ = states.conservative⁻.u
    u⁺ = states.conservative⁺.u
    w⁻ = states.auxiliary⁻.w
    w⁺ = states.auxiliary⁺.w

    u⁺ = -u⁻
    w⁺ = -w⁻

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::OceanFloorNoSlip, ::CentralNumericalFluxGradient)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
set numerical flux to zero for u and w

# Arguments
- `Q⁺`: state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxGradient,
    ::OceanFloorNoSlip,
    ::HBModel,
    states,
    n⁻,
    t,
)
    u⁺ = states.conservative⁺.u
    w⁺ = states.auxiliary⁺.w
    FT = eltype(u⁺)

    u⁺ = SVector(-zero(FT), -zero(FT))
    w⁺ = -zero(FT)

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::OceanFloorNoSlip, ::CentralNumericalFluxSecondOrder)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
sets ghost point to have no numerical flux on the boundary for u,w and κ∇θ

# Arguments
- `Q⁺`: state vector at ghost point
- `D⁺`: diffusive state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `D⁻`: diffusive state vector at interior point
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    ::OceanFloorNoSlip,
    ::HBModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    u⁻ = states.conservative⁻.u
    u⁺ = states.conservative⁺.u
    w⁻ = states.auxiliary⁻.w
    w⁺ = states.auxiliary⁺.w
    κ∇θ⁻ = states.gradient_flux⁻.κ∇θ
    κ∇θ⁺ = states.gradient_flux⁺.κ∇θ

    u⁺ = -u⁻
    w⁺ = -w⁻
    κ∇θ⁺ = -κ∇θ⁻

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::Union{OceanSurface*}, ::Union{RusanovNumericalFlux, CentralNumericalFluxGradient})

applying neumann boundary conditions, so don't need to do anything for these numerical fluxes
"""
@inline function ocean_boundary_state!(
    ::Union{RusanovNumericalFlux, CentralNumericalFluxGradient},
    ::Union{
        OceanSurfaceNoStressNoForcing,
        OceanSurfaceStressNoForcing,
        OceanSurfaceNoStressForcing,
        OceanSurfaceStressForcing,
    },
    ::HBModel,
    states,
    n⁻,
    t,
)
    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::OceanSurfaceNoStressNoForcing, ::CentralNumericalFluxSecondOrder)

apply no flux boundary condition for velocity
apply no flux boundary condition for temperature
set ghost point to have no numerical flux on the boundary for ν∇u and κ∇θ

# Arguments
- `Q⁺`: state vector at ghost point
- `D⁺`: diffusive state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `D⁻`: diffusive state vector at interior point
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    ::OceanSurfaceNoStressNoForcing,
    ::HBModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    ν∇u⁻ = states.gradient_flux⁻.ν∇u
    κ∇θ⁻ = states.gradient_flux⁻.κ∇θ
    ν∇u⁺ = states.gradient_flux⁺.ν∇u
    κ∇θ⁺ = states.gradient_flux⁺.κ∇θ

    ν∇u⁺ = -ν∇u⁻
    κ∇θ⁺ = -κ∇θ⁻

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::OceanSurfaceStressNoForcing, ::CentralNumericalFluxSecondOrder)

apply wind-stress boundary condition for velocity
apply no flux boundary condition for temperature
set ghost point for numerical flux on the boundary for ν∇u and κ∇θ

# Arguments
- `Q⁺`: state vector at ghost point
- `D⁺`: diffusive state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `D⁻`: diffusive state vector at interior point
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    ::OceanSurfaceStressNoForcing,
    m::HBModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    ν∇u⁻ = states.gradient_flux⁻.ν∇u
    κ∇θ⁻ = states.gradient_flux⁻.κ∇θ
    ν∇u⁺ = states.gradient_flux⁺.ν∇u
    κ∇θ⁺ = states.gradient_flux⁺.κ∇θ

    τᶻ = kinematic_stress(m.problem, A⁻.y, m.ρₒ)
    τ = @SMatrix [-0 -0; -0 -0; τᶻ -0]

    ν∇u⁺ = -ν∇u⁻ + 2 * τ
    κ∇θ⁺ = -κ∇θ⁻

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::OceanSurfaceNoStressForcing, ::CentralNumericalFluxSecondOrder)

apply no flux boundary condition for velocity
apply forcing boundary condition for temperature
set ghost point for numerical flux on the boundary for ν∇u and κ∇θ

# Arguments
- `Q⁺`: state vector at ghost point
- `D⁺`: diffusive state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `D⁻`: diffusive state vector at interior point
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    ::OceanSurfaceNoStressForcing,
    m::HBModel,
    states,
    n⁻,
    t,
)
    ν∇u⁻ = states.gradient_flux⁻.ν∇u
    κ∇θ⁻ = states.gradient_flux⁻.κ∇θ
    ν∇u⁺ = states.gradient_flux⁺.ν∇u
    κ∇θ⁺ = states.gradient_flux⁺.κ∇θ

    σᶻ = temperature_flux(m.problem, A⁻.y, Q⁻.θ)
    σ = @SVector [-0, -0, σᶻ]

    ν∇u⁺ = -ν∇u⁻
    κ∇θ⁺ = -κ∇θ⁻ + 2 * σ

    return nothing
end

"""
    ocean_boundary_state!(::HBModel, ::OceanSurfaceStressForcing, ::CentralNumericalFluxSecondOrder)

apply wind-stress boundary condition for velocity
apply forcing boundary condition for temperature
set ghost point for numerical flux on the boundary for ν∇u and κ∇θ

# Arguments
- `Q⁺`: state vector at ghost point
- `D⁺`: diffusive state vector at ghost point
- `A⁺`: auxiliary state vector at ghost point
- `n⁻`: normal vector, not used
- `Q⁻`: state vector at interior
- `D⁻`: diffusive state vector at interior point
- `A⁻`: auxiliary state vector at interior point
- `t`:  time, not used
"""
@inline function ocean_boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    ::OceanSurfaceStressForcing,
    m::HBModel,
    states,
    n⁻,
    t,
)
    ν∇u⁻ = states.gradient_flux⁻.ν∇u
    κ∇θ⁻ = states.gradient_flux⁻.κ∇θ
    ν∇u⁺ = states.gradient_flux⁺.ν∇u
    κ∇θ⁺ = states.gradient_flux⁺.κ∇θ

    τᶻ = kinematic_stress(m.problem, A⁻.y, m.ρₒ)
    τ = @SMatrix [-0 -0; -0 -0; τᶻ -0]

    σᶻ = temperature_flux(m.problem, A⁻.y, Q⁻.θ)
    σ = @SVector [-0, -0, σᶻ]

    ν∇u⁺ = -ν∇u⁻ + 2 * τ
    κ∇θ⁺ = -κ∇θ⁻ + 2 * σ

    return nothing
end
