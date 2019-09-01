#### Moisture component in atmosphere model
abstract type MoistureModel end

vars_state(::MoistureModel, T) = @vars()
vars_gradient(::MoistureModel, T) = @vars()
vars_diffusive(::MoistureModel, T) = @vars()
vars_aux(::MoistureModel, T) = @vars()

function update_aux!(::MoistureModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function diffusive!(::MoistureModel, diffusive, ∇transform, state, aux, t, ν)
end
function flux_diffusive!(::MoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function flux_nondiffusive!(::MoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function gradvariables!(::MoistureModel, transform::Vars, state::Vars, aux::Vars, t::Real)
end

function internal_energy(m::MoistureModel, state::Vars, aux::Vars)
  T = eltype(state)
  ρinv = 1 / state.ρ
  ρe_kin = ρinv*sum(abs2, state.ρu)/2
  ρe_pot = state.ρ * aux.orientation.Φ
  ρe_int = state.ρe - ρe_kin - ρe_pot
  e_int = ρinv*ρe_int
  return e_int
end

temperature(m::MoistureModel, state::Vars, aux::Vars) = air_temperature(thermo_state(m, state, aux))
pressure(m::MoistureModel, state::Vars, aux::Vars) = air_pressure(thermo_state(m, state, aux))
soundspeed(m::MoistureModel, state::Vars, aux::Vars) = soundspeed_air(thermo_state(m, state, aux))

"""
    DryModel

Assumes the moisture components is in the dry limit.
"""
struct DryModel <: MoistureModel
end

vars_aux(::DryModel,T) = @vars(θ_v::T)
function update_aux!(m::DryModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  e_int = internal_energy(m, state, aux)
  TS = PhaseDry(e_int, state.ρ)
  aux.moisture.θ_v = virtual_pottemp(TS)
  nothing
end

get_phase_partition(::DryModel, state::Vars) = PhasePartition(eltype(state)(0))
thermo_state(::DryModel, state::Vars, aux::Vars) = PhaseDry(internal_energy(m, state, aux), state.ρ)

"""
    EquilMoist

Assumes the moisture components are computed via thermodynamic equilibrium.
"""
struct EquilMoist <: MoistureModel
end
vars_state(::EquilMoist,T) = @vars(ρq_tot::T)
vars_gradient(::EquilMoist,T) = @vars(q_tot::T)
vars_diffusive(::EquilMoist,T) = @vars(ρd_q_tot::SVector{3,T})
vars_aux(::EquilMoist,T) = @vars(temperature::T, θ_v::T)

function update_aux!(m::EquilMoist, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  e_int = internal_energy(m, state, aux)
  TS = PhaseEquil(e_int, get_phase_partition(m, state).tot, state.ρ)
  aux.moisture.temperature = air_temperature(TS)
  aux.moisture.θ_v = virtual_pottemp(TS)
  nothing
end

get_phase_partition(::EquilMoist, state::Vars) = PhasePartition(state.moisture.ρq_tot/state.ρ)
function thermo_state(m::EquilMoist, state::Vars, aux::Vars)
  e_int = internal_energy(m, state, aux)
  PhaseEquil(e_int, state.moisture.ρq_tot/state.ρ, state.ρ, aux.moisture.temperature)
end

function gradvariables!(m::EquilMoist, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  transform.moisture.q_tot = state.moisture.ρq_tot * ρinv
end

function diffusive!(m::EquilMoist, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, ρD_t::Union{Real,AbstractMatrix})
  # diffusive flux of q_tot
  diffusive.moisture.ρd_q_tot = -ρD_t .* ∇transform.moisture.q_tot
end

function flux_diffusive!(m::EquilMoist, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  u = state.ρu / state.ρ
  flux.ρ += diffusive.moisture.ρd_q_tot
  flux.ρu += diffusive.moisture.ρd_q_tot .* u'
  flux.moisture.ρq_tot = diffusive.moisture.ρd_q_tot
end
