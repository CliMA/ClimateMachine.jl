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
struct KinematicStress <: VelocityDragBC end

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

"""
    ocean_boundary_state!(nf, boundaries::Tuple, ::HBModel,
                          Q⁺, A⁺, n, Q⁻, A⁻, bctype)
applies boundary conditions for the first-order and gradient fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@generated function ocean_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    boundaries::Tuple,
    ocean,
    Q⁺,
    A⁺,
    n,
    Q⁻,
    A⁻,
    bctype,
    t,
    args...,
)
    N = fieldcount(boundaries)
    return quote
        Base.Cartesian.@nif(
            $(N + 1),
            i -> bctype == i, # conditionexpr
            i -> ocean_boundary_state!(
                nf,
                boundaries[i],
                ocean,
                Q⁺,
                A⁺,
                n,
                Q⁻,
                A⁻,
                t,
            ), # expr
            i -> error("Invalid boundary tag")
        ) # elseexpr
        return nothing
    end
end

"""
    ocean_boundary_state!(nf, boundaries::Tuple, ::HBModel,
                          Q⁺, A⁺, D⁺, n, Q⁻, A⁻, D⁻, bctype)
applies boundary conditions for the second-order fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@generated function ocean_boundary_state!(
    nf::NumericalFluxSecondOrder,
    boundaries::Tuple,
    ocean,
    Q⁺,
    D⁺,
    A⁺,
    n,
    Q⁻,
    D⁻,
    A⁻,
    bctype,
    t,
    args...,
)
    N = fieldcount(boundaries)
    return quote
        Base.Cartesian.@nif(
            $(N + 1),
            i -> bctype == i, # conditionexpr
            i -> ocean_boundary_state!(
                nf,
                boundaries[i],
                ocean,
                Q⁺,
                D⁺,
                A⁺,
                n,
                Q⁻,
                D⁻,
                A⁻,
                t,
            ), # expr
            i -> error("Invalid boundary tag")
        ) # elseexpr
        return nothing
    end
end
