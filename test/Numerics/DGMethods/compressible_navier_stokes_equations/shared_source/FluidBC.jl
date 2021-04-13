abstract type BoundaryCondition end

"""
    FluidBC(momentum    = Impenetrable(NoSlip())
            temperature = Insulating())

The standard boundary condition for CNSEModel. The default options imply a "no flux" boundary condition.
"""
Base.@kwdef struct FluidBC{M, T} <: BoundaryCondition
    momentum::M = Impenetrable(NoSlip())
    temperature::T = Insulating()
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
Base.@kwdef struct TemperatureFlux{𝒯, 𝒫} <: TemperatureBC
    flux::𝒯 = nothing
    params::𝒫 = nothing
end

function check_bc(bcs, label)
    bctype = FluidBC

    bc_ρu = check_bc(bcs, Val(:ρu), label)
    bc_ρθ = check_bc(bcs, Val(:ρθ), label)

    return bctype(bc_ρu, bc_ρθ)
end

function check_bc(bcs, ::Val{:ρθ}, label)
    if haskey(bcs, :ρθ)
        if haskey(bcs[:ρθ], label)
            return bcs[:ρθ][label]
        end
    end

    return Insulating()
end

function check_bc(bcs, ::Val{:ρu}, label)
    if haskey(bcs, :ρu)
        if haskey(bcs[:ρu], label)
            return bcs[:ρu][label]
        end
    end

    return Impenetrable(FreeSlip())
end

# these functions just trim off the extra arguments
function _cnse_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    bc,
    model,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    t,
    _...,
)
    return cnse_boundary_state!(nf, bc, model, state⁺, aux⁺, n, state⁻, aux⁻, t)
end

function _cnse_boundary_state!(
    nf::NumericalFluxSecondOrder,
    bc,
    model,
    state⁺,
    gradflux⁺,
    hyperflux⁺,
    aux⁺,
    n,
    state⁻,
    gradflux⁻,
    hyperflux⁻,
    aux⁻,
    t,
    _...,
)
    return cnse_boundary_state!(
        nf,
        bc,
        model,
        state⁺,
        gradflux⁺,
        aux⁺,
        n,
        state⁻,
        gradflux⁻,
        aux⁻,
        t,
    )
end
