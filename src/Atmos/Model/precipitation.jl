#### Precipitation component in atmosphere model
abstract type PrecipitationModel end

export NoPrecipitation, Rain

using ..Microphysics

vars_state(    ::PrecipitationModel, T) = @vars()
vars_gradient( ::PrecipitationModel, T) = @vars()
vars_diffusive(::PrecipitationModel, T) = @vars()
vars_aux(      ::PrecipitationModel, T) = @vars()

function update_aux!(::PrecipitationModel, state::Vars, diffusive::Vars, aux::Vars, t::Real);end
function diffusive!(::PrecipitationModel, diffusive, ∇transform, state, aux, t, ν);end
function flux_diffusive!(::PrecipitationModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real);end
function flux_nondiffusive!(::PrecipitationModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real);end
function gradvariables!(::PrecipitationModel, transform::Vars, state::Vars, aux::Vars, t::Real);end

"""
    NoPrecipitation <: PrecipitationModel

No precipitation.
"""
struct NoPrecipitation <: PrecipitationModel end


"""
    Rain <: PrecipitationModel

Precipitation model with rain only.
"""
struct Rain <: PrecipitationModel end

vars_state(    ::Rain, T) = @vars(ρq_rain::T)
vars_gradient( ::Rain, T) = @vars(q_rain::T)
vars_diffusive(::Rain, T) = @vars(ρd_q_rain::SVector{3,T})
vars_aux(      ::Rain, T) = @vars(terminal_velocity::T, src_q_rai_tot::T)

function update_aux!(m::Rain, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  DT = eltype(state)
  q_rain = state.precipitation.ρq_rain/state.ρ
  if q_rain > DT(0) # TODO - need a way to prevent negative values
    aux.precipitation.terminal_velocity = terminal_velocity(q_rain, state.ρ)
  else
    aux.precipitation.terminal_velocity = DT(0)
  end

  ρ = state.ρ
  p = aux.pressure
  # TODO - ensure positive definite
  q_tot = max(DT(0), state.ρq_tot/ρ)
  q_rai = max(DT(0), state.ρq_rain/ρ)

  # current state
  ts    = thermo_state(m, state, aux)
  # q     = PhasePartition(q_tot, q_liq, q_ice)
  q     = PhasePartition(ts)
  # T     = air_temperature(e_int, q)
  T     = air_temperature(ts)
  # equilibrium state at current T
  # q_eq = PhasePartition_equil(T, ρ, q_tot)
  q_eq = PhasePartition_equil(ts)

  # tendency from cloud water condensation/evaporation
  # src_q_liq = conv_q_vap_to_q_liq(q_eq, q)# TODO - temporary handling ice

  # tendencies from rain
  src_q_rai_evap = conv_q_rai_to_q_vap(q_rai, q, T , p, ρ)
  src_q_rai_acnv = conv_q_liq_to_q_rai_acnv(q.liq)
  src_q_rai_accr = conv_q_liq_to_q_rai_accr(q.liq, q_rai, ρ)

  aux.precipitation.src_q_rai_tot = src_q_rai_acnv + src_q_rai_accr + src_q_rai_evap
end

function gradvariables!(m::Rain, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.precipitation.q_rain = state.precipitation.ρq_rain/state.ρ
end

function source_microphysics!(m::Rain, source::Vars, state::Vars, aux::Vars, t::Real)
  source.precipitation.ρq_rain += aux.precipitation.src_q_rai_tot
end

function diffusive!(m::Rain, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, ρν::Union{Real,AbstractMatrix}, D_T)
  # diffusive flux
  diffusive.precipitation.ρd_q_rain = state.ρ .* (-D_T) .* ∇transform.precipitation.q_rain
end


function flux_diffusive!(m::Rain, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real, D_T)
  flux.precipitation.ρq_rain = diffusive.precipitation.ρd_q_rain
end

function boundarycondition_moisture!(m::Rain, stateP::Vars, diffP::Vars, auxP::Vars,
    nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype::BC, t) where {BC}
  stateP.precipitation.ρq_rain = eltype(stateP)(0)
end

