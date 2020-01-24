export DryModel, EquilMoist

#### Moisture component in atmosphere model
abstract type MoistureModel end

vars_state(::MoistureModel, FT) = @vars()
vars_gradient(::MoistureModel, FT) = @vars()
vars_diffusive(::MoistureModel, FT) = @vars()
vars_aux(::MoistureModel, FT) = @vars()

function atmos_nodal_update_aux!(::MoistureModel, ::AtmosModel, state::Vars, aux::Vars, t::Real) end
function flux_moisture!(         ::MoistureModel, ::AtmosModel, state::Vars, aux::Vars, t::Real, flux::Grad) end
function diffusive!(             ::MoistureModel, state::Vars, aux::Vars, t::Real, diffusive, ∇transform, ρD_t) end
function flux_diffusive!(        ::MoistureModel, state::Vars, aux::Vars, t::Real, flux::Grad, diffusive::Vars) end
function flux_nondiffusive!(     ::MoistureModel, state::Vars, aux::Vars, t::Real, flux::Grad) end
function gradvariables!(         ::MoistureModel, state::Vars, aux::Vars, t::Real, transform::Vars) end

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
Base.@kwdef struct EquilMoist <: MoistureModel
  maxiter::Int = 3
end
vars_state(::EquilMoist,FT) = @vars(ρq_tot::FT)
vars_gradient(::EquilMoist,FT) = @vars(q_tot::FT, h_tot::FT)
vars_diffusive(::EquilMoist,FT) = @vars(ρd_q_tot::SVector{3,FT}, ρd_h_tot::SVector{3,FT})
vars_aux(::EquilMoist,FT) = @vars(temperature::FT, θ_v::FT, q_liq::FT)

@inline function atmos_nodal_update_aux!(moist::EquilMoist, atmos::AtmosModel,
                                         state::Vars, aux::Vars, t::Real)
  e_int = internal_energy(moist, atmos.orientation, state, aux)
  TS = PhaseEquil(e_int, state.ρ, state.moisture.ρq_tot/state.ρ, moist.maxiter)
  aux.moisture.temperature = air_temperature(TS)
  aux.moisture.θ_v = virtual_pottemp(TS)
  aux.moisture.q_liq = PhasePartition(TS).liq
  nothing
end

function thermo_state(moist::EquilMoist, orientation::Orientation, state::Vars, aux::Vars)
  e_int = internal_energy(moist, orientation, state, aux)
  FT = eltype(state)
  return PhaseEquil{FT}(e_int, state.ρ, state.moisture.ρq_tot/state.ρ, aux.moisture.temperature)
end

function gradvariables!(moist::EquilMoist, state::Vars, aux::Vars, t::Real, transform::Vars)
  ρinv = 1/state.ρ
  transform.moisture.q_tot = state.moisture.ρq_tot * ρinv
end

function diffusive!(moist::EquilMoist, state::Vars, aux::Vars, t::Real, diffusive::Vars, ∇transform::Grad, ρD_t)
  # diffusive flux of q_tot
  diffusive.moisture.ρd_q_tot = (-ρD_t) .* ∇transform.moisture.q_tot
end

function flux_moisture!(moist::EquilMoist, atmos::AtmosModel, state::Vars, aux::Vars, t::Real, flux::Grad)
  ρ = state.ρ
  u = state.ρu / ρ
  z = altitude(atmos.orientation, aux)
  usub = subsidence_velocity(atmos.subsidence, z)
  ẑ = vertical_unit_vector(atmos.orientation, aux)
  u_tot = u .- usub * ẑ
  flux.moisture.ρq_tot += u_tot * state.moisture.ρq_tot
end

function flux_diffusive!(moist::EquilMoist, state::Vars, aux::Vars, t::Real, flux::Grad, diffusive::Vars)
  u = state.ρu / state.ρ
  flux.ρ += diffusive.moisture.ρd_q_tot
  flux.ρu += diffusive.moisture.ρd_q_tot .* u'
  flux.moisture.ρq_tot += diffusive.moisture.ρd_q_tot
end
