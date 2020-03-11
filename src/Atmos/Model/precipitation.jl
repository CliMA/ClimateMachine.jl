#### Precipitation component in atmosphere model
abstract type PrecipitationModel end

export NoPrecipitation, Rain

using ..Microphysics

vars_state(    ::PrecipitationModel, FT) = @vars()
vars_gradient( ::PrecipitationModel, FT) = @vars()
vars_diffusive(::PrecipitationModel, FT) = @vars()
vars_aux(      ::PrecipitationModel, FT) = @vars()

function atmos_nodal_update_aux!(::PrecipitationModel, m::AtmosModel, state::Vars, aux::Vars, t::Real);end
function flux_precipitation!(::PrecipitationModel, atmos::AtmosModel, flux::Grad, state::Vars, aux::Vars, t::Real);end
function diffusive!(::PrecipitationModel, diffusive, ∇transform, state, aux, t, ρD_t);end
function flux_diffusive!(::PrecipitationModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real);end
function flux_nondiffusive!(::PrecipitationModel, flux::Grad, state::Vars, aux::Vars, t::Real);end
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

vars_state(    ::Rain, FT) = @vars(ρq_rain::FT)
vars_gradient( ::Rain, FT) = @vars(q_rain::FT)
vars_diffusive(::Rain, FT) = @vars(ρd_q_rain::SVector{3,FT})
vars_aux(      ::Rain, FT) = @vars(terminal_velocity::FT, src_q_rai_tot::FT, src_accr::FT, src_acnv::FT)
function atmos_nodal_update_aux!(rain::Rain, atmos::AtmosModel,
                                         state::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  q_rain = state.precipitation.ρq_rain/state.ρ
  if q_rain > FT(0) # TODO - need a way to prevent negative values
    aux.precipitation.terminal_velocity = terminal_velocity(q_rain, state.ρ)
  else
    aux.precipitation.terminal_velocity = FT(0)
  end

  ρ = state.ρ
  ρinv = 1/state.ρ
  # TODO - ensure positive definite
  q_tot = max(FT(0), state.moisture.ρq_tot*ρinv)
  q_rai = max(FT(0), state.precipitation.ρq_rain*ρinv)

  # current state
  ts    = thermo_state(atmos.moisture, atmos.orientation, state, aux)
  # q     = PhasePartition(q_tot, q_liq, q_ice)
  q     = PhasePartition(ts)
  # T     = air_temperature(e_int, q)
  T     = air_temperature(ts)
  # equilibrium state at current T
  q_eq = q#PhasePartition_equi/(ts)

  # tendency from cloud water condensation/evaporation
  # src_q_liq = conv_q_vap_to_q_liq(q_eq, q)# TODO - temporary handling ice
  p = air_pressure(ts)
  # tendencies from rain
  src_q_rai_evap = conv_q_rai_to_q_vap(q_rai, q, T , p, ρ)
  src_q_rai_acnv = conv_q_liq_to_q_rai_acnv(q.liq)
  src_q_rai_accr = conv_q_liq_to_q_rai_accr(q.liq, q_rai, ρ)
  aux.precipitation.src_accr = src_q_rai_accr
  aux.precipitation.src_acnv = src_q_rai_acnv
  aux.precipitation.src_q_rai_tot = src_q_rai_acnv + src_q_rai_accr + src_q_rai_evap
end


function gradvariables!(rain::Rain, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.precipitation.q_rain = state.precipitation.ρq_rain/state.ρ
end

function diffusive!(rain::Rain, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  # diffusive flux
  diffusive.precipitation.ρd_q_rain = state.ρ  .* ∇transform.precipitation.q_rain
end

function flux_precipitation!(rain::Rain, atmos::AtmosModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  u = state.ρu / state.ρ
  k̂ = aux.orientation.∇Φ / norm(aux.orientation.∇Φ)
  flux.precipitation.ρq_rain += state.precipitation.ρq_rain * (u - aux.precipitation.terminal_velocity * k̂)
  flux.ρ -= aux.precipitation.terminal_velocity * state.precipitation.ρq_rain * k̂
  flux.moisture.ρq_tot -= aux.precipitation.terminal_velocity * state.precipitation.ρq_rain * k̂
  flux.moisture.ρq_liq -= aux.precipitation.terminal_velocity * state.precipitation.ρq_rain * k̂
  flux.ρu -= aux.precipitation.terminal_velocity * state.precipitation.ρq_rain * k̂ .* state.ρu'
end

function flux_diffusive!(rain::Rain, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real, D_T)
  flux.precipitation.ρq_rain += (-D_T) .* diffusive.precipitation.ρd_q_rain
end
#=
function boundarycondition_precipitation!(rain::Rain, stateP::Vars, diffP::Vars, auxP::Vars,
    nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype::BC, t) where {BC}
  stateP.precipitation.ρq_rain = eltype(stateP)(0)
end
=#
