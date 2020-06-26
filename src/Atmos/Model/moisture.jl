export DryModel, EquilMoist, NonEquilMoist

#### Moisture component in atmosphere model
abstract type MoistureModel end

vars_state_conservative(::MoistureModel, FT) = @vars()
vars_state_gradient(::MoistureModel, FT) = @vars()
vars_state_gradient_flux(::MoistureModel, FT) = @vars()
vars_state_auxiliary(::MoistureModel, FT) = @vars()

function atmos_nodal_update_auxiliary_state!(
    ::MoistureModel,
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function flux_moisture!(
    ::MoistureModel,
    ::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function compute_gradient_flux!(
    ::MoistureModel,
    diffusive,
    ∇transform,
    state,
    aux,
    t,
) end
function flux_second_order!(
    ::MoistureModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
) end
function flux_first_order!(
    ::MoistureModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function compute_gradient_argument!(
    ::MoistureModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) end

internal_energy(atmos::AtmosModel, state::Vars, aux::Vars) =
    internal_energy(atmos.moisture, atmos.orientation, state, aux)

@inline function internal_energy(
    moist::MoistureModel,
    orientation::Orientation,
    state::Vars,
    aux::Vars,
)
    Thermodynamics.internal_energy(
        state.ρ,
        state.ρe,
        state.ρu,
        gravitational_potential(orientation, aux),
    )
end

temperature(atmos::AtmosModel, ::MoistureModel, state::Vars, aux::Vars) =
    air_temperature(thermo_state(atmos, state, aux))
pressure(atmos::AtmosModel, ::MoistureModel, state::Vars, aux::Vars) =
    air_pressure(thermo_state(atmos, state, aux))
soundspeed(atmos::AtmosModel, ::MoistureModel, state::Vars, aux::Vars) =
    soundspeed_air(thermo_state(atmos, state, aux))

@inline function total_specific_enthalpy(
    atmos::AtmosModel,
    moist::MoistureModel,
    state::Vars,
    aux::Vars,
)
    phase = thermo_state(atmos, state, aux)
    R_m = gas_constant_air(phase)
    T = air_temperature(phase)
    e_tot = state.ρe * (1 / state.ρ)
    return e_tot + R_m * T
end

"""
    DryModel

Assumes the moisture components is in the dry limit.
"""
struct DryModel{FT} <: MoistureModel end

vars_state_auxiliary(::DryModel, FT) = @vars(θ_v::FT, air_T::FT, enthalpy::FT)
@inline function atmos_nodal_update_auxiliary_state!(
    moist::DryModel,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    e_int = internal_energy(atmos, state, aux)
    ts = PhaseDry(atmos.param_set, e_int, state.ρ)
    aux.moisture.θ_v = virtual_pottemp(ts)
    aux.moisture.air_T = air_temperature(ts)
    aux.moisture.enthalpy = total_specific_enthalpy(atmos, atmos.moisture, state, aux)
    nothing
end

thermo_state(atmos::AtmosModel, state::Vars, aux::Vars) =
    thermo_state(atmos, atmos.moisture, state, aux)

function thermo_state(
    atmos::AtmosModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
)
    return PhaseDry(
        atmos.param_set,
        internal_energy(atmos, state, aux),
        state.ρ,
    )
end

"""
    EquilMoist

Assumes the moisture components are computed via thermodynamic equilibrium.
"""
struct EquilMoist{FT} <: MoistureModel
    maxiter::Int
    tolerance::FT
end
EquilMoist{FT}(;
    maxiter::IT = 3,
    tolerance::FT = FT(1e-1),
) where {FT <: AbstractFloat, IT <: Int} = EquilMoist{FT}(maxiter, tolerance)


vars_state_conservative(::EquilMoist, FT) = @vars(ρq_tot::FT)
vars_state_gradient(::EquilMoist, FT) = @vars(q_tot::FT)
vars_state_gradient_flux(::EquilMoist, FT) = @vars(∇q_tot::SVector{3, FT})
vars_state_auxiliary(::EquilMoist, FT) =
    @vars(temperature::FT, θ_v::FT, q_liq::FT, pressure::FT, enthalpy::FT)

@inline function atmos_nodal_update_auxiliary_state!(
    moist::EquilMoist,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ps = atmos.param_set
    e_int = internal_energy(atmos, state, aux)
    ts = PhaseEquil(
        ps,
        e_int,
        state.ρ,
        state.moisture.ρq_tot / state.ρ,
        moist.maxiter,
        moist.tolerance,
    )
    aux.moisture.temperature = air_temperature(ts)
    aux.moisture.pressure = air_pressure(ts)
    aux.moisture.θ_v = virtual_pottemp(ts)
    aux.moisture.q_liq = PhasePartition(ts).liq
    aux.moisture.enthalpy = total_specific_enthalpy(atmos, atmos.moisture, state, aux)
    nothing
end

function thermo_state(
    atmos::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    e_int = internal_energy(atmos, state, aux)
    ps = atmos.param_set
    PS = typeof(ps)
    FT = eltype(state)
    return PhaseEquil{FT, PS}(
        ps,
        e_int,
        state.ρ,
        state.moisture.ρq_tot / state.ρ,
        aux.moisture.temperature,
    )
end

function compute_gradient_argument!(
    moist::EquilMoist,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.moisture.q_tot = state.moisture.ρq_tot * ρinv
end

function compute_gradient_flux!(
    moist::EquilMoist,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    # diffusive flux of q_tot
    diffusive.moisture.∇q_tot = ∇transform.moisture.q_tot
end

function flux_moisture!(
    moist::EquilMoist,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρ = state.ρ
    u = state.ρu / ρ
    flux.moisture.ρq_tot += u * state.moisture.ρq_tot
end

function flux_second_order!(
    moist::EquilMoist,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    d_q_tot = (-D_t) .* diffusive.moisture.∇q_tot
    flux_second_order!(moist, flux, state, d_q_tot)
end
#TODO: Consider whether to not pass ρ and ρu (not state), foc BCs reasons
function flux_second_order!(moist::EquilMoist, flux::Grad, state::Vars, d_q_tot)
    flux.ρ += d_q_tot * state.ρ
    flux.ρu += d_q_tot .* state.ρu'
    flux.moisture.ρq_tot += d_q_tot * state.ρ
end

"""
  NonEquilMoist
  Assumes that there isn't any thermodynamic equilibrium

  Adds addtional moisture variable for cloud water ρq_liq and cloud ice ρq_ice::FT that are transported in the same way as ρq_tot
  The equilibrium phase partition is calculated and used as a reference to compute the source term for the cloud phase.
"""

struct NonEquilMoist{FT} <: MoistureModel end

vars_state(::NonEquilMoist, FT) = @vars(ρq_tot::FT, ρq_liq::FT, ρq_ice::FT)
vars_gradient(::NonEquilMoist, FT) =
    @vars(q_tot::FT, q_liq::FT, q_ice::FT, h_tot::FT)
vars_diffusive(::NonEquilMoist, FT) = @vars(
    ∇q_tot::SVector{3, FT},
    ∇q_liq::SVector{3, FT},
    ∇q_ice::SVector{3, FT}
)
vars_aux(::NonEquilMoist, FT) =
    @vars(temperature::FT, θ_v::FT, q_liq::FT, q_ice::FT, p::FT, src_qliq::FT)

@inline function atmos_nodal_update_aux!(
    moist::NonEquilMoist,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    TS = thermo_state(atmos, moist, state, aux)
    aux.moisture.temperature = air_temperature(TS)
    q_p_sat = q_vap_saturation(TS)#PhasePartition_equil(
        #air_temperature(TS),
       #state.ρ,
      #state.moisture.ρq_tot,
    #)
    aux.moisture.src_qliq = conv_q_vap_to_q_liq(PhasePartition(q_p_sat), PhasePartition(TS))
    aux.moisture.θ_v = virtual_pottemp(TS)
    aux.moisture.q_liq = PhasePartition(TS).liq
    aux.moisture.q_ice = PhasePartition(TS).ice
    aux.moisture.p = air_pressure(TS)
    return nothing
end

function thermo_state(
    atmos::AtmosModel,
    moist::NonEquilMoist,
    state::Vars,
    aux::Vars,
)
    FT = eltype(state)
    ρinv = 1 / state.ρ
    e_int = internal_energy(atmos, state, aux)

    q_pt = PhasePartition{FT}(
        state.moisture.ρq_tot * ρinv,
        state.moisture.ρq_liq * ρinv,
        state.moisture.ρq_ice * ρinv,
    )
    return PhaseNonEquil(e_int, state.ρ, q_pt)
end

function gradvariables!(
    moist::NonEquilMoist,
    atmos::AtmosModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.moisture.q_tot = state.moisture.ρq_tot * ρinv
    phase = thermo_state(moist, atmos.orientation, state, aux)
    transform.moisture.q_liq = state.moisture.ρq_liq * ρinv
    transform.moisture.q_ice = state.moisture.ρq_ice * ρinv
end

function diffusive!(
    moist::NonEquilMoist,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.moisture.∇q_tot = ∇transform.moisture.q_tot
    diffusive.moisture.∇q_liq = ∇transform.moisture.q_liq
    diffusive.moisture.∇q_ice = ∇transform.moisture.q_ice
end

function flux_moisture!(
    moist::NonEquilMoist,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρ = state.ρ
    u = state.ρu / ρ
    flux.moisture.ρq_tot += u * state.moisture.ρq_tot
    flux.moisture.ρq_liq += u * state.moisture.ρq_liq
    flux.moisture.ρq_ice += u * state.moisture.ρq_ice
end

function flux_diffusive!(
    moist::NonEquilMoist,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    d_q_tot = (-D_t) .* diffusive.moisture.∇q_tot
    d_q_liq = (-D_t) .* diffusive.moisture.∇q_liq
    d_q_ice = (-D_t) .* diffusive.moisture.∇q_ice
    u = state.ρu / state.ρ
    flux.ρ += d_q_tot * state.ρ
    flux.ρu += d_q_tot .* u' * state.ρ
    flux.moisture.ρq_tot += d_q_tot * state.ρ
    flux.moisture.ρq_liq += d_q_liq * state.ρ
    flux.moisture.ρq_ice += d_q_ice * state.ρ
end
