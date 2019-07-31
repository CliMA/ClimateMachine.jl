#### Moisture component in atmosphere model
abstract type MoistureModel end

vars_state(::MoistureModel, T) = Tuple{}
vars_gradient(::MoistureModel, T) = Tuple{}
vars_diffusive(::MoistureModel, T) = Tuple{}
vars_aux(::MoistureModel, T) = Tuple{}

function diffusive!(::MoistureModel, diffusive, ∇transform, state, aux, t, ν)
end
function flux_diffusive!(::MoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function update_aux!(::MoistureModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function gradvariables!(::MoistureModel, transform::Vars, state::Vars, aux::Vars, t::Real)
end

function internal_energy(m::MoistureModel, state::Vars, aux::Vars)
  T = eltype(state)
  q_pt = get_phase_partition(m, state)
  ρinv = 1 / state.ρ
  ρe_kin = ρinv*sum(abs2, state.ρu)/2
  ρe_pot = state.ρ * grav * aux.coord.z
  ρe_int = state.ρe - ρe_kin
  # ρe_int = state.ρe - ρe_kin - ρe_pot # FIXME: Should we always include/exclude ρe_pot?
  e_int = ρinv*ρe_int
  return e_int
end

pressure(m::MoistureModel  , state::Vars, aux::Vars) = air_pressure(thermo_state(m, state, aux))
soundspeed(m::MoistureModel, state::Vars, aux::Vars) = soundspeed_air(thermo_state(m, state, aux))

function update_aux!(m::MoistureModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  aux.moisture.e_int = internal_energy(m, state, aux)
  TS = PhaseEquil(aux.moisture.e_int, get_phase_partition(m, state).tot, state.ρ)
  aux.moisture.temperature = air_temperature(TS)
  nothing
end

"""
    DryModel

Assumes the moisture components is in the dry limit.
"""
struct DryModel <: MoistureModel
end

vars_aux(::DryModel,T) = NamedTuple{(:e_int, :temperature),Tuple{T,T}}

get_phase_partition(::DryModel, state::Vars) = PhasePartition(eltype(state)(0))
thermo_state(::DryModel, state::Vars, aux::Vars) = PhaseEquil(aux.moisture.e_int, eltype(state.ρ)(0), state.ρ, aux.moisture.temperature)

"""
    EquilMoist

Assumes the moisture components are computed via thermodynamic equilibrium.
"""
struct EquilMoist <: MoistureModel
end
vars_state(::EquilMoist,T) = NamedTuple{(:ρq_tot,),Tuple{T}}
vars_gradient(::EquilMoist,T) = NamedTuple{(:q_vap, :q_liq, :q_ice, :temperature),Tuple{T,T,T,T}}
vars_diffusive(::EquilMoist,T) = NamedTuple{(:ρd_q_tot, :ρJ_ρD), Tuple{SVector{3,T},SVector{3,T}}}
vars_aux(::EquilMoist,T) = NamedTuple{(:e_int, :temperature),Tuple{T,T}}

get_phase_partition(::EquilMoist, state::Vars) = PhasePartition(state.ρq_tot/state.ρ)
thermo_state(::EquilMoist, state::Vars, aux::Vars) = PhaseEquil(aux.moisture.e_int, state.ρq_tot/state.ρ, state.ρ, aux.moisture.temperature)

function gradvariables!(m::EquilMoist, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρ = state.ρ
  q_tot = state.ρq_tot / ρ
  phase = thermo_state(m, state, aux)
  q = PhasePartition(phase)

  transform.moisture.q_vap = q.tot - q.liq - q.ice
  transform.moisture.q_liq = q.liq
  transform.moisture.q_ice = q.ice
  transform.moisture.temperature = aux.moisture.temperature
end


function diffusive!(m::EquilMoist, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, ν::Union{Real,AbstractMatrix})
  # turbulent Prandtl number
  diag_ν = ν isa Real ? ν : diag(ν) # either a scalar or vector
  D_q_vap = D_q_liq = D_q_ice = D_q_tot = diag_ν / Prandtl_t # either a scalar or vector

  # diffusive flux of q_tot
  # FIXME
  ρd_q_vap = state.ρ * (-D_q_vap) .* ∇transform.moisture.q_vap # a vector
  ρd_q_liq = state.ρ * (-D_q_liq) .* ∇transform.moisture.q_liq # a vector
  ρd_q_ice = state.ρ * (-D_q_ice) .* ∇transform.moisture.q_ice # a vector

  diffusive.moisture.ρd_q_tot = ρd_q_vap + ρd_q_liq + ρd_q_ice

  D_T = diag_ν / Prandtl_t
  phase = thermo_state(m, state, aux)

  # J is the conductive or SGS turbulent flux of sensible heat per unit mass
  ρJ = state.ρ * cv_m(phase) * D_T * ∇transform.moisture.temperature

  # D is the total specific energy flux
  T = aux.moisture.temperature
  u = state.ρu / state.ρ

  # FIXME
  I_vap = cv_v * (T - T_0) + e_int_v0
  I_liq = cv_l * (T - T_0)
  I_ice = cv_v * (T - T_0) - e_int_i0
  e_kin = 0.5 * sum(abs2,u)
  e_pot = grav * aux.coord.z
  e_tot_vap = e_kin + e_pot + I_vap
  e_tot_liq = e_kin + e_pot + I_liq
  e_tot_ice = e_kin + e_pot + I_ice
  ρD = state.ρ * ((e_tot_vap + R_v*T)*diffusive.moisture.ρd_q_vap + e_tot_liq*diffusive.moisture.ρd_q_liq + e_tot_ice*diffusive.moisture.ρd_q_ice)

  diffusive.moisture.ρJ_ρD = ρJ + ρD
end

function flux_diffusive!(m::EquilMoist, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  flux.ρ += diffusive.moisture.ρd_q_tot
  flux.ρu += diffusive.moisture.ρd_q_tot .* u'
  flux.ρe += diffusive.moisture.ρJ_ρD

  flux.moisture.ρq_tot = diffusive.moisture.ρd_q_tot
end
