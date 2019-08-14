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
function gradvariables!(::MoistureModel, transform::Vars, state::Vars, aux::Vars, t::Real)
end

function internal_energy(m::MoistureModel, state::Vars, aux::Vars)
  T = eltype(state)
  ρinv = 1 / state.ρ
  q_pt = get_phase_partition(m, state)
  
  ρe_kin = ρinv*sum(abs2, state.ρu)/2
  ρe_pot = state.ρ * grav * aux.coord.z
  ρe_int = state.ρe - ρe_kin - ρe_pot
  
  e_int = ρinv*ρe_int
  return e_int
end

temperature(m::MoistureModel, state::Vars, aux::Vars) = air_temperature(thermo_state(m, state, aux))
pressure(m::MoistureModel, state::Vars, aux::Vars) = air_pressure(thermo_state(m, state, aux))
soundspeed(m::MoistureModel, state::Vars, aux::Vars) = soundspeed_air(thermo_state(m, state, aux))
gas_constant(m::MoistureModel, state::Vars, aux::Vars) = gas_constant_air(thermo_state(m, state, aux))

"""
    DryModel

Assumes the moisture components is in the dry limit.
"""
struct DryModel <: MoistureModel
end

vars_aux(::DryModel,T) = @vars(e_int::T, temperature::T, R_m::T)
function update_aux!(m::DryModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  aux.moisture.e_int = internal_energy(m, state, aux)
  TS = PhaseDry(aux.moisture.e_int, state.ρ)
  aux.moisture.temperature = air_temperature(TS)
  aux.moisture.R_m = gas_constant_air(TS)
  nothing
end

get_phase_partition(::DryModel, state::Vars) = PhasePartition(eltype(state)(0))
thermo_state(::DryModel, state::Vars, aux::Vars) = PhaseDry(aux.moisture.e_int, state.ρ)

"""
    EquilMoist

Assumes the moisture components are computed via thermodynamic equilibrium.
"""
struct EquilMoist <: MoistureModel
end
vars_state(::EquilMoist,T) = @vars(ρq_tot::T)
vars_gradient(::EquilMoist,T) = @vars(q_tot::T)
vars_diffusive(::EquilMoist,T) = @vars(ρd_q_tot::SVector{3,T})
vars_aux(::EquilMoist,T) = @vars(e_int::T, temperature::T)

function update_aux!(m::EquilMoist, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  aux.moisture.e_int = internal_energy(m, state, aux)
  TS = PhaseEquil(aux.moisture.e_int, get_phase_partition(m, state).tot, state.ρ)

  aux.moisture.temperature = air_temperature(TS)
  nothing
end

get_phase_partition(::EquilMoist, state::Vars) = PhasePartition(state.moisture.ρq_tot/state.ρ)
thermo_state(::EquilMoist, state::Vars, aux::Vars) = PhaseEquil(aux.moisture.e_int, state.moisture.ρq_tot/state.ρ, state.ρ, aux.moisture.temperature)

function gradvariables!(m::EquilMoist, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  phase = thermo_state(m, state, aux)
  q = PhasePartition(phase)
  R_m = gas_constant_air(q)
  T = aux.moisture.temperature
  e_tot = state.ρe * ρinv
  
  transform.moisture.q_tot          = state.moisture.ρq_tot * ρinv
end


function diffusive!(m::EquilMoist, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, ν::Union{Real,AbstractMatrix})
  # turbulent Prandtl number
  T = eltype(state)
  diag_ν = ν isa Real ? ν : diag(ν) # either a scalar or vector
  Prandtl_t = T(1/3)
  D_T = diag_ν / Prandtl_t

  diffusive.moisture.ρd_q_tot = state.ρ * (-D_T) .* ∇transform.moisture.q_tot # diffusive flux of q_tot
end

function flux_diffusive!(m::EquilMoist, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  T = eltype(state)
  
  ρu = state.ρu
  ρinv = 1/state.ρ
  u = ρinv * ρu
  
  flux.ρ               += diffusive.moisture.ρd_q_tot
  flux.ρu              += diffusive.moisture.ρd_q_tot .* u'
  flux.moisture.ρq_tot += diffusive.moisture.ρd_q_tot
end
