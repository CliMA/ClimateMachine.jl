export OceanBC,
    VelocityBC,
    VelocityDragBC,
    TemperatureBC,
    Impenetrable,
    Penetrable,
    NoSlip,
    FreeSlip,
    KinematicStress,
    Insulating,
    TemperatureFlux

using StaticArrays

using ..BalanceLaws
using ..DGMethods.NumericalFluxes

"""
    OceanBC(velocity    = Impenetrable(NoSlip())
            temperature = Insulating())

The standard boundary condition for OceanModel. The default options imply a "no flux" boundary condition.
"""
Base.@kwdef struct OceanBC{M, T}
    velocity::M = Impenetrable(NoSlip())
    temperature::T = Insulating()
end

abstract type VelocityBC end
abstract type VelocityDragBC end
abstract type TemperatureBC end

"""
    Impenetrable(drag::VelocityDragBC) :: VelocityBC

Defines an impenetrable wall model for velocity. This implies:
  - no flow in the direction normal to the boundary, and
  - flow parallel to the boundary is subject to the `drag` condition.
"""
struct Impenetrable{D <: VelocityDragBC} <: VelocityBC
    drag::D
end

"""
    Penetrable(drag::VelocityDragBC) :: VelocityBC

Defines an penetrable wall model for velocity. This implies:
  - no constraint on flow in the direction normal to the boundary, and
  - flow parallel to the boundary is subject to the `drag` condition.
"""
struct Penetrable{D <: VelocityDragBC} <: VelocityBC
    drag::D
end

"""
    NoSlip() :: VelocityDragBC

Zero velocity at the boundary.
"""
struct NoSlip <: VelocityDragBC end

"""
    FreeSlip() :: VelocityDragBC

No surface drag on velocity parallel to the boundary.
"""
struct FreeSlip <: VelocityDragBC end

"""
    KinematicStress(stress) :: VelocityDragBC

Applies the specified kinematic stress on velocity normal to the boundary.
Prescribe the net inward kinematic stress across the boundary by `stress`,
a function with signature `stress(problem, state, aux, t)`, returning the flux (in m²/s²).
"""
struct KinematicStress{S} <: VelocityDragBC
    stress::S

    function KinematicStress(stress::S = nothing) where {S}
        new{S}(stress)
    end
end

kinematic_stress(problem, y, ρ₀) = @SVector [0, 0] # fallback for generic problems
kinematic_stress(problem, y, ρ₀, ::Nothing) = kinematic_stress(problem, y, ρ₀)
kinematic_stress(problem, y, ρ₀, stress) = stress(y)

"""
    Insulating() :: TemperatureBC

No temperature flux across the boundary
"""
struct Insulating <: TemperatureBC end

"""
    TemperatureFlux(flux) :: TemperatureBC

Prescribe the net inward temperature flux across the boundary by `flux`,
a function with signature `flux(problem, state, aux, t)`, returning the flux (in m⋅K/s).
"""
struct TemperatureFlux <: TemperatureBC end

# these functions just trim off the extra arguments
function _ocean_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    bc,
    ocean,
    Q⁺,
    A⁺,
    n,
    Q⁻,
    A⁻,
    t,
    _...,
)
    return ocean_boundary_state!(nf, bc, ocean, Q⁺, A⁺, n, Q⁻, A⁻, t)
end

function _ocean_boundary_state!(
    nf::NumericalFluxSecondOrder,
    bc,
    ocean,
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
    return ocean_boundary_state!(nf, bc, ocean, Q⁺, D⁺, A⁺, n, Q⁻, D⁻, A⁻, t)
end
