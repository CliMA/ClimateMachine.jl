using ClimateMachine.DGMethods: DGModel
using ClimateMachine.MPIStateArrays: MPIStateArray
using ClimateMachine.DGMethods.NumericalFluxes: NumericalFluxSecondOrder
using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.Mesh.Grids: Direction

import ClimateMachine.BalanceLaws:
    vars_state_gradient,
    vars_state_gradient_flux,
    vars_integrals,
    vars_reverse_integrals,
    flux_second_order!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

@inline function linearized_pressure(ρ, ρe, Φ)
    FT = eltype(ρ)
    γ = FT(gamma(param_set))
    (γ - 1) * (ρe - ρ * Φ)
end

abstract type DryAtmosLinearModel <: BalanceLaw end

function vars_state_conservative(lm::DryAtmosLinearModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
    end
end
vars_state_gradient(lm::DryAtmosLinearModel, FT) = @vars()
vars_state_gradient_flux(lm::DryAtmosLinearModel, FT) = @vars()
vars_state_auxiliary(lm::DryAtmosLinearModel, FT) =
    vars_state_auxiliary(lm.atmos, FT)
vars_integrals(lm::DryAtmosLinearModel, FT) = @vars()
vars_reverse_integrals(lm::DryAtmosLinearModel, FT) = @vars()

function update_auxiliary_state!(
    dg::DGModel,
    lm::DryAtmosLinearModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    return false
end
function flux_second_order!(
    lm::DryAtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
integral_load_auxiliary_state!(
    lm::DryAtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
integral_set_auxiliary_state!(lm::DryAtmosLinearModel, aux::Vars, integ::Vars) =
    nothing
reverse_integral_load_auxiliary_state!(
    lm::DryAtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
reverse_integral_set_auxiliary_state!(
    lm::DryAtmosLinearModel,
    aux::Vars,
    integ::Vars,
) = nothing
flux_second_order!(
    lm::DryAtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
) = nothing
function wavespeed(
    lm::DryAtmosLinearModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ref = aux.ref_state
    return soundspeed(ref.ρ, ref.p)
end

function boundary_state!(
    nf::NumericalFluxFirstOrder,
    lm::DryAtmosLinearModel,
    args...,
)
    boundary_state!(nf, lm.atmos, args...)
end
function boundary_state!(
    nf::NumericalFluxSecondOrder,
    lm::DryAtmosLinearModel,
    args...,
)
    nothing
end
init_state_auxiliary!(lm::DryAtmosLinearModel, aux::Vars, geom::LocalGeometry) =
    nothing
init_state_conservative!(
    lm::DryAtmosLinearModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
) = nothing

struct DryAtmosAcousticGravityLinearModel{M} <: DryAtmosLinearModel
    atmos::M
    function DryAtmosAcousticGravityLinearModel(atmos::M) where {M}
        if atmos.ref_state === NoReferenceState()
            error("DryAtmosAcousticGravityLinearModel needs a model with a reference state")
        end
        new{M}(atmos)
    end
end
function flux_first_order!(
    lm::DryAtmosAcousticGravityLinearModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)
    ref = aux.ref_state

    flux.ρ = state.ρu
    pL = linearized_pressure(state.ρ, state.ρe, aux.Φ)
    flux.ρu += pL * I
    flux.ρe = ((ref.ρe + ref.p) / ref.ρ) * state.ρu
    nothing
end
function source!(
    lm::DryAtmosAcousticGravityLinearModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    ::NTuple{1, Dir},
) where {Dir <: Direction}
    if Dir === VerticalDirection || Dir === EveryDirection
        ∇Φ = aux.∇Φ
        source.ρu -= state.ρ * ∇Φ
    end
    nothing
end
