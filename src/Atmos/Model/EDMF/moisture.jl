#### Moisture component in atmosphere model
abstract type EDMFMoistureModel end

export EDMFDryModel, EDMFEquilMoist

vars_state(    ::EDMFMoistureModel, T, N) = @vars()
vars_gradient( ::EDMFMoistureModel, T, N) = @vars()
vars_diffusive(::EDMFMoistureModel, T, N) = @vars()
vars_aux(      ::EDMFMoistureModel, T, N) = @vars()

function update_aux!(       edmf::EDMF{N}, m::EDMFMoistureModel, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function gradvariables!(    edmf::EDMF{N}, m::EDMFMoistureModel, transform::Vars, state::Vars, aux::Vars, t::Real) where N; end
function flux_diffusive!(   edmf::EDMF{N}, m::EDMFMoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_nondiffusive!(edmf::EDMF{N}, m::EDMFMoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_advective!(   edmf::EDMF{N}, m::EDMFMoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function source!(           edmf::EDMF{N}, m::EDMFMoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function boundarycondition!(edmf::EDMF{N}, m::EDMFMoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end

function internal_energy(edmf::EDMF{N}, m::EDMFMoistureModel, state::Vars, aux::Vars) where N
  T = eltype(state)
  id = idomains(N)
  e_int = similar(state.turbconv.ρ)
  @inbounds for i in (id.env, id.up...)
    ρinv = 1 / state.ρ
    ρe_kin = ρinv*sum(abs2, state.turbconv.momentum.ρu[i])/2
    ρe_pot = state.turbconv.area_frac.ρa[i] * grav * aux.coord.z
    ρe_int = state.turbconv.energy.ρe[i] - ρe_kin - ρe_pot
    e_int[i] = ρinv*ρe_int
  end
  return e_int
end

temperature(edmf::EDMF, state::Vars, aux::Vars, i) = air_temperature(thermo_state(edmf, state, aux, i))
pressure(edmf::EDMF, state::Vars, aux::Vars, i) = air_pressure(thermo_state(edmf, state, aux, i))
soundspeed(edmf::EDMF, state::Vars, aux::Vars, i) = soundspeed_air(thermo_state(edmf, state, aux, i))

"""
    EDMFDryModel

Assumes the moisture components is in the dry limit.
"""
struct EDMFDryModel <: EDMFMoistureModel end

vars_aux(::EDMFDryModel, T, N) = @vars(e_int::T)
function update_aux!(edmf::EDMF, m::EDMFDryModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  aux.turbconv.energy.e_int = internal_energy(edmf, m, state, aux)
  nothing
end

function thermo_state(edmf::EDMF{N,AF,EDMFDryModel,TKE,SM,ML,ED,P,B}, state::Vars, aux::Vars, i) where {N,AF,TKE,SM,ML,ED,P,B}
  if i==idomains(N).gm
    PhaseDry(aux.moisture.e_int, state.ρ)
  else
    PhaseDry(aux.turbconv.energy.e_int[i], state.ρ)
  end
end

"""
    EDMFEquilMoist

Assumes the moisture components are computed via thermodynamic equilibrium.
"""
struct EDMFEquilMoist <: EDMFMoistureModel
end
vars_state(::EDMFEquilMoist,T, N) = @vars(ρq_tot::T)
vars_gradient(::EDMFEquilMoist,T, N) = @vars(q_tot::T, total_enthalpy::T)
vars_diffusive(::EDMFEquilMoist,T, N) = @vars(ρd_q_tot::SVector{3,T}, ρ_SGS_enthalpyflux::SVector{3,T})
vars_aux(::EDMFEquilMoist,T, N) = @vars(e_int::T, temperature::T)

function update_aux!(edmf::EDMF{N}, m::EDMFEquilMoist, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  aux.turbconv.energy.e_int = internal_energy(edmf, m, state, aux)
  id = idomains(N)
  @inbounds for i in (id.env, id.up...)
    TS = PhaseEquil(aux.turbconv.energy.e_int[i], get_phase_partition(m, state, i).tot, state.ρ)
    aux.turbconv.moisture.temperature[i] = air_temperature(TS)
  end
  nothing
end

function get_phase_partition(edmf::EDMF{N}, m::EDMFEquilMoist, state::Vars, i) where N
  if i==idomains(N).gm
    PhasePartition(state.moisture.ρq_tot/state.ρ)
  else
    PhasePartition(state.turbconv.moisture.ρq_tot[i]/state.ρ)
  end
end

function thermo_state(edmf::EDMF{N,AF,EDMFEquilMoist,TKE,SM,ML,ED,P,B}, state::Vars, aux::Vars, i) where {N,AF,TKE,SM,ML,ED,P,B}
  if i==idomains(N).gm
    PhaseEquil(aux.moisture.e_int, state.ρ, state.moisture.ρq_tot/state.ρ, aux.moisture.temperature)
  else
    PhaseEquil(aux.turbconv.energy.e_int[i], state.ρ, state.turbconv.moisture.ρq_tot[i]/state.ρ, aux.turbconv.moisture.temperature[i])
  end
end
