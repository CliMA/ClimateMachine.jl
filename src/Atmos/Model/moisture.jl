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
  q_pt = get_phase_partition(m, state)
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

vars_aux(::DryModel,T) = @vars()
function update_aux!(m::DryModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  e_int = internal_energy(m, state, aux)
  TS = PhaseDry(e_int, state.ρ)
  nothing
end

thermo_state(::DryModel, state::Vars, aux::Vars) = PhaseDry(internal_energy(m, state, aux), state.ρ)

"""
    EquilMoist

Assumes the moisture components are computed via thermodynamic equilibrium.
"""
struct EquilMoist <: MoistureModel
end
vars_state(::EquilMoist,T) = @vars(ρq_tot::T)
vars_gradient(::EquilMoist,T) = @vars(q_tot::T, h_tot::T)
vars_diffusive(::EquilMoist,T) = @vars(ρd_q_tot::SVector{3,T}, ρd_h_tot::SVector{3,T})
vars_aux(::EquilMoist,T) = @vars(temperature::T)

function update_aux!(m::EquilMoist, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  e_int = internal_energy(m, state, aux)
  TS = PhaseEquil(aux.moisture.e_int, get_phase_partition(m, state).tot, state.ρ)
  aux.moisture.temperature = air_temperature(TS)
  nothing
end


get_phase_partition(::EquilMoist, state::Vars) = PhasePartition(state.moisture.ρq_tot/state.ρ)
function thermo_state(::EquilMoist, state::Vars, aux::Vars)
  e_int = internal_energy(m, state, aux)
  PhaseEquil(e_int, state.moisture.ρq_tot/state.ρ, state.ρ, aux.moisture.temperature)
end

function gradvariables!(m::EquilMoist, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  transform.moisture.q_tot = state.moisture.ρq_tot * ρinv

  phase = thermo_state(m, state, aux)
  R_m = gas_constant_air(phase)
  T = air_temperature(phase)
  e_tot = state.ρe * ρinv
  transform.moisture.h_tot = e_tot + R_m*T
end


function diffusive!(m::EquilMoist, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, ρν::Union{Real,AbstractMatrix})
  # turbulent Prandtl number
  diag_ρν = ρν isa Real ? ρν : diag(ρν) # either a scalar or matrix
  D_T = diag_ρν / Prandtl_t

  # diffusive flux of q_tot
  diffusive.moisture.ρd_q_tot = (-D_T) .* ∇transform.moisture.q_tot

  # diffusive flux of total energy
  diffusive.moisture.ρd_h_tot = (-D_T) .* ∇transform.transform.moisture.h_tot
end

function flux_diffusive!(m::EquilMoist, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  flux.ρ += diffusive.moisture.ρd_q_tot
  flux.ρu += diffusive.moisture.ρd_q_tot .* u'

  flux.moisture.ρq_tot = diffusive.moisture.ρd_q_tot
end

"""
    NonEquilMoist

Assumes the moisture components are computed via thermodynamic non-equilibrium.
"""
struct NonEquilMoist <: MoistureModel end
vars_state(::NonEquilMoist    ,T) = @vars(ρq_tot::T, ρq_liq::T, ρq_ice::T)
vars_gradient(::NonEquilMoist ,T) = @vars(q_tot::T,q_liq::T,q_ice::T, h_tot::T)
vars_diffusive(::NonEquilMoist,T) = @vars(ρd_q_tot::SVector{3,T},ρd_q_liq::SVector{3,T},ρd_q_ice::SVector{3,T})
vars_aux(::NonEquilMoist      ,T) = @vars()

function update_aux!(m::NonEquilMoist, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end

function get_phase_partition(::NonEquilMoist, state::Vars)
  ρ = state.ρ
  PhasePartition(state.moisture.ρq_tot/ρ,
                 state.moisture.ρq_liq/ρ,
                 state.moisture.ρq_ice/ρ)
end
function thermo_state(m::NonEquilMoist, state::Vars, aux::Vars)
  ρ = state.ρ
  q_pt = PhasePartition(state.moisture.ρq_tot/ρ, state.moisture.ρq_liq/ρ, state.moisture.ρq_ice/ρ)
  PhaseNonEquil(internal_energy(m, state, aux), q_pt, ρ)
end

function gradvariables!(m::NonEquilMoist, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  transform.moisture.q_tot = state.moisture.ρq_tot * ρinv

  phase = thermo_state(m, state, aux)
  R_m = gas_constant_air(phase)
  T = air_temperature(phase)
  e_tot = state.ρe * ρinv
  transform.moisture.h_tot = e_tot + R_m*T
end

function diffusive!(m::NonEquilMoist, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, ρν::Union{Real,AbstractMatrix}, D_T)
  # TODO: handle D_T better
  # diffusive flux
  diffusive.moisture.ρd_q_tot = state.ρ .* (-D_T) .* ∇transform.moisture.q_tot
  diffusive.moisture.ρd_q_liq = state.ρ .* (-D_T) .* ∇transform.moisture.q_liq
  diffusive.moisture.ρd_q_ice = state.ρ .* (-D_T) .* ∇transform.moisture.q_ice
end

function flux_diffusive!(m::NonEquilMoist, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  # TODO: handle D_T better
  flux.moisture.ρq_tot = diffusive.moisture.ρd_q_tot
  flux.moisture.ρq_liq = diffusive.moisture.ρd_q_liq
  flux.moisture.ρq_ice = diffusive.moisture.ρd_q_ice
end

function source_microphysics!(m::NonEquilMoist, source::Vars, state::Vars, aux::Vars, t::Real)
  source.moisture.ρq_tot -= aux.precipitation.src_q_rai_tot
  source.moisture.ρq_liq -= aux.precipitation.src_q_rai_tot
end

function boundarycondition_moisture!(m::NonEquilMoist, stateP::Vars, diffP::Vars, auxP::Vars,
    nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t)
  stateP.moisture.ρq_tot = stateM.moisture.ρq_tot
  stateP.moisture.ρq_liq = stateM.moisture.ρq_liq
  stateP.moisture.ρq_ice = stateM.moisture.ρq_ice
  nothing
end
