export DryModel, EquilMoist

#### Moisture component in atmosphere model
abstract type MoistureModel end

vars_state(::MoistureModel, FT) = @vars()
vars_gradient(::MoistureModel, FT) = @vars()
vars_diffusive(::MoistureModel, FT) = @vars()
vars_aux(::MoistureModel, FT) = @vars()

function atmos_nodal_update_aux!(::MoistureModel, m::AtmosModel, state::Vars,
                                 aux::Vars, t::Real)
end
function diffusive!(::MoistureModel, diffusive, ∇transform, state, aux, t, ρD_t)
end
function flux_diffusive!(::MoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function flux_nondiffusive!(::MoistureModel, flux::Grad, state::Vars, aux::Vars, t::Real)
end
function gradvariables!(::MoistureModel, transform::Vars, state::Vars, aux::Vars, t::Real)
end

@inline function internal_energy(moist::MoistureModel, orientation::Orientation, state::Vars, aux::Vars)
  MoistThermodynamics.internal_energy(state.ρ, state.ρe, state.ρu, gravitational_potential(orientation, aux))
end
@inline temperature(moist::MoistureModel, orientation::Orientation, state::Vars, aux::Vars) = air_temperature(thermo_state(moist, orientation, state, aux))
@inline pressure(moist::MoistureModel, orientation::Orientation, state::Vars, aux::Vars) = air_pressure(thermo_state(moist, orientation, state, aux))
@inline soundspeed(moist::MoistureModel, orientation::Orientation, state::Vars, aux::Vars) = soundspeed_air(thermo_state(moist, orientation, state, aux))

@inline function total_specific_enthalpy(moist::MoistureModel, orientation::Orientation, state::Vars, aux::Vars)
  phase = thermo_state(moist, orientation, state, aux)
  R_m = gas_constant_air(phase)
  T = air_temperature(phase)
  e_tot = state.ρe * (1/state.ρ)
  e_tot + R_m*T
end



"""
    DryModel

Assumes the moisture components is in the dry limit.
"""
struct DryModel <: MoistureModel
end

vars_aux(::DryModel,FT) = @vars(θ_v::FT)
@inline function atmos_nodal_update_aux!(moist::DryModel, atmos::AtmosModel,
                                         state::Vars, aux::Vars, t::Real)
  e_int = internal_energy(moist, atmos.orientation, state, aux)
  TS = PhaseDry(e_int, state.ρ)
  aux.moisture.θ_v = virtual_pottemp(TS)
  nothing
end

thermo_state(moist::DryModel, orientation::Orientation, state::Vars, aux::Vars) = PhaseDry(internal_energy(moist, orientation, state, aux), state.ρ)

"""
    EquilMoist

Assumes the moisture components are computed via thermodynamic equilibrium.
"""
struct EquilMoist <: MoistureModel
end
vars_state(::EquilMoist,FT) = @vars(ρq_tot::FT)
vars_gradient(::EquilMoist,FT) = @vars(q_tot::FT, h_tot::FT)
vars_diffusive(::EquilMoist,FT) = @vars(ρd_q_tot::SVector{3,FT}, ρd_h_tot::SVector{3,FT})
vars_aux(::EquilMoist,FT) = @vars(temperature::FT, θ_v::FT, q_liq::FT)

@inline function atmos_nodal_update_aux!(moist::EquilMoist, atmos::AtmosModel,
                                         state::Vars, aux::Vars, t::Real)
  e_int = internal_energy(moist, atmos.orientation, state, aux)
  TS = PhaseEquil(e_int, state.moisture.ρq_tot/state.ρ, state.ρ)
  aux.moisture.temperature = air_temperature(TS)
  aux.moisture.θ_v = virtual_pottemp(TS)
  aux.moisture.q_liq = PhasePartition(TS).liq
  nothing
end

function thermo_state(moist::EquilMoist, orientation::Orientation, state::Vars, aux::Vars)
  e_int = internal_energy(moist, orientation, state, aux)
  PhaseEquil(e_int, state.moisture.ρq_tot/state.ρ, state.ρ, aux.moisture.temperature)
end

function gradvariables!(moist::EquilMoist, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  transform.moisture.q_tot = state.moisture.ρq_tot * ρinv
end

function diffusive!(moist::EquilMoist, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, ρD_t)
  # diffusive flux of q_tot
  diffusive.moisture.ρd_q_tot = (-ρD_t) .* ∇transform.moisture.q_tot
end

function flux_diffusive!(moist::EquilMoist, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  u = state.ρu / state.ρ
  flux.ρ += diffusive.moisture.ρd_q_tot
  flux.ρu += diffusive.moisture.ρd_q_tot .* u'
  flux.moisture.ρq_tot += diffusive.moisture.ρd_q_tot
end
