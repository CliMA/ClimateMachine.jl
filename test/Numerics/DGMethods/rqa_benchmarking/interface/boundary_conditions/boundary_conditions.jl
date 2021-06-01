abstract type AbstractBoundaryCondition end

"""
    FluidBC(momentum    = Impenetrable(NoSlip())
            temperature = Insulating())

The standard boundary condition for CNSEModel. The default options imply a "no flux" boundary condition.
"""
Base.@kwdef struct FluidBC{M, T} <: AbstractBoundaryCondition
    ρu::M = Impenetrable(NoSlip())
    ρθ::T = Insulating()
end

abstract type StateBC end
abstract type MomentumBC <: StateBC end
abstract type MomentumDragBC <: StateBC end
abstract type TemperatureBC <: StateBC end

(bc::StateBC)(state, aux, t) = bc.flux(bc.params, state, aux, t)

"""
    Impenetrable(drag::MomentumDragBC) :: MomentumBC

Defines an impenetrable wall model for momentum. This implies:
  - no flow in the direction normal to the boundary, and
  - flow parallel to the boundary is subject to the `drag` condition.
"""
struct Impenetrable{D <: MomentumDragBC} <: MomentumBC
    drag::D
end

"""
    Penetrable(drag::MomentumDragBC) :: MomentumBC

Defines an penetrable wall model for momentum. This implies:
  - no constraint on flow in the direction normal to the boundary, and
  - flow parallel to the boundary is subject to the `drag` condition.
"""
struct Penetrable{D <: MomentumDragBC} <: MomentumBC
    drag::D
end

"""
    NoSlip() :: MomentumDragBC

Zero momentum at the boundary.
"""
struct NoSlip <: MomentumDragBC end

"""
    FreeSlip() :: MomentumDragBC

No surface drag on momentum parallel to the boundary.
"""
struct FreeSlip <: MomentumDragBC end

"""
    MomentumFlux(stress) :: MomentumDragBC

Applies the specified kinematic stress on momentum normal to the boundary.
Prescribe the net inward kinematic stress across the boundary by `stress`,
a function with signature `stress(problem, state, aux, t)`, returning the flux (in m²/s²).
"""
Base.@kwdef struct MomentumFlux{𝒯, 𝒫} <: MomentumDragBC
    flux::𝒯 = nothing
    params::𝒫 = nothing
end

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
struct TemperatureFlux{T} <: TemperatureBC
    flux::T

    function TemperatureFlux(flux::T = nothing) where {T}
        new{T}(flux)
    end
end

# Smart defaults
#=

"""
    FluidBC(momentum    = Impenetrable(NoSlip())
            temperature = Insulating())
The standard boundary condition for CNSEModel. The default options imply a "no flux" boundary condition.
"""
Base.@kwdef struct FluidBC{ℳ, ℰ, 𝒬} <: BoundaryCondition
    momentum::ℳ = FreeSlip()
    temperature::𝒯 = NoFlux()
    moisture::𝒬
end

function check_bc(bcs, label)
    bctype = FluidBC

    bc_ρu = check_bc(bcs, Val(:ρu), label)
    bc_ρθ = check_bc(bcs, Val(:ρθ), label)

    return bctype(bc_ρu, bc_ρθ)
end

function check_bc(bcs, ::Val{:ρe}, label)
    if haskey(bcs, :ρe)
        if haskey(bcs[:ρe], label)
            return bcs[:ρe][label]
        end
    end

    return NoFlux()
end

function check_bc(bcs, ::Val{:ρq}, label)
    if haskey(bcs, :ρq)
        if haskey(bcs[:ρq], label)
            return bcs[:ρq][label]
        end
    end

    return NoFlux()
end

function check_bc(bcs, ::Val{:ρu}, label)
    if haskey(bcs, :ρu)
        if haskey(bcs[:ρu], label)
            return bcs[:ρu][label]
        end
    end

    return FreeSlip()
end

function get_boundary_conditions(
    model::SpatialModel{BL},
) where {BL <: AbstractFluid3D}
    bcs = model.boundary_conditions

    west_east = (check_bc(bcs, :west), check_bc(bcs, :east))
    south_north = (check_bc(bcs, :south), check_bc(bcs, :north))
    bottom_top = (check_bc(bcs, :bottom), check_bc(bcs, :top))

    return (west_east..., south_north..., bottom_top...)
end

=#
