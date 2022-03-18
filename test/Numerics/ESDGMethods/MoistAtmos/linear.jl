using ClimateMachine.DGMethods: DGModel
using ClimateMachine.MPIStateArrays: MPIStateArray
using ClimateMachine.DGMethods.NumericalFluxes: NumericalFluxSecondOrder
using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.Mesh.Grids: Direction
using Thermodynamics

import ClimateMachine.BalanceLaws:
    flux_second_order!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

@inline function linearized_pressure(ρ, ρe, Φ)
    # param_set is currently a defined constant
    FT = eltype(ρ)
    γ = FT(gamma(param_set))
    if total_energy
        (γ - 1) * (ρe - ρ * Φ)
    else
        (γ - 1) * ρe
    end
end

"""
    linearized_air_pressure(ρ, ρe_tot, ρe_pot, ρq_tot=0, ρq_liq=0, ρq_ice=0)
The air pressure, linearized around a dry rest state, from the equation of state
(ideal gas law) where:
 - `ρ` (moist-)air density
 - `ρe_tot` total energy density
 - `ρe_pot` potential energy density
and, optionally,
 - `ρq_tot` total water density
 - `ρq_liq` liquid water density
 - `ρq_ice` ice density
"""
function linearized_air_pressure(
    param_set::AbstractParameterSet,
    ρ::FT,
    ρe_tot::FT,
    ρe_pot::FT,
    ρq_tot::FT = FT(0),
    ρq_liq::FT = FT(0),
    ρq_ice::FT = FT(0),
) where {FT <: Real, PS}
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    return ρ * _R_d * _T_0 +
           _R_d / _cv_d * (
        ρe_tot - ρe_pot - (ρq_tot - ρq_liq) * _e_int_v0 +
        ρq_ice * (_e_int_i0 + _e_int_v0)
    )
end

abstract type MoistAtmosLinearModel <: BalanceLaw end

function vars_state(lm::MoistAtmosLinearModel, ::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        ρq_tot::FT
    end
end
vars_state(lm::MoistAtmosLinearModel, st::Auxiliary, FT) =
    vars_state(lm.atmos, st, FT)

function update_auxiliary_state!(
    dg::DGModel,
    lm::MoistAtmosLinearModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    return false
end
function flux_second_order!(
    lm::MoistAtmosLinearModel,
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
    lm::MoistAtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
integral_set_auxiliary_state!(lm::MoistAtmosLinearModel, aux::Vars, integ::Vars) =
    nothing
reverse_integral_load_auxiliary_state!(
    lm::MoistAtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
reverse_integral_set_auxiliary_state!(
    lm::MoistAtmosLinearModel,
    aux::Vars,
    integ::Vars,
) = nothing
flux_second_order!(
    lm::MoistAtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
) = nothing
function wavespeed(
    lm::MoistAtmosLinearModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ref = aux.ref_state
    return soundspeed(ref.ρ, ref.p)
end

boundary_conditions(lm::MoistAtmosLinearModel) = (1, 2)
function boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc,
    lm::MoistAtmosLinearModel,
    args...,
)
    boundary_state!(nf, bc, lm.atmos, args...)
end
function boundary_state!(
    nf::NumericalFluxSecondOrder,
    bc,
    lm::MoistAtmosLinearModel,
    args...,
)
    nothing
end
init_state_auxiliary!(lm::MoistAtmosLinearModel, aux::Vars, geom::LocalGeometry) =
    nothing
init_state_prognostic!(
    lm::MoistAtmosLinearModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
) = nothing

struct MoistAtmosAcousticGravityLinearModel{M} <: MoistAtmosLinearModel
    atmos::M
    function MoistAtmosAcousticGravityLinearModel(atmos::M) where {M}
        if atmos.ref_state === NoReferenceState()
            error("MoistAtmosAcousticGravityLinearModel needs a model with a reference state")
        end
        new{M}(atmos)
    end
end
function flux_first_order!(
    lm::MoistAtmosAcousticGravityLinearModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)
    ref = aux.ref_state

    flux.ρ = state.ρu
    pL = linearized_air_pressure(param_set, state.ρ, state.ρe, state.ρ*aux.Φ, state.ρq_tot)
    flux.ρu += pL * I
    flux.ρe = ((ref.ρe + ref.p) / ref.ρ) * state.ρu
    nothing
end
function source!(
    lm::MoistAtmosAcousticGravityLinearModel,
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
        if !total_energy
            source.ρe -= state.ρu' * ∇Φ
        end
    end
    nothing
end
