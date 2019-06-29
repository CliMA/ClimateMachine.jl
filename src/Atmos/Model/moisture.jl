#### Moisture component in atmosphere model
abstract type MoistureModel end 

struct DryModel <: MoistureModel
end
vars_state(::DryModel) = ()
function flux!(m::DryModel, flux::Grad, state::State, diffusive::State, auxstate::State, t::Real)
end
function preodefun_elem!(m::DryModel, auxstate::State, state::State, t::Real)
end

function pressure(::DryModel, state, diffusive, auxstate, t)
  T = eltype(state)
  γ = T(γ_exact)
  ρinv = 1 / state.ρ
  P = (γ-1)*(state.ρe - ρinv * (state.ρu^2 + state.ρv^2 + state.ρw^2) / 2)
end



struct EquilMoist <: MoistureModel
end
vars_state(::EquilMoist) = (:ρqt,)
function flux!(m::EquilMoist, flux::Grad, state::State, diffusive::State, auxstate::State, t::Real)
  flux_tracer!(:ρqt, flux, state)
end
function preodefun_elem!(m::EquilMoist, auxstate::State, state::State, t::Real)
  T = eltype(auxstate)

  ρinv = 1 / state.ρ
  e_int = (state.ρe - ρinv * (state.ρu^2 + state.ρv^2 + state.ρw^2) / 2) * ρinv - T(grav) * aux.z
  qt = state.ρqt * ρinv

  TS = PhaseEquil(e_int, q_tot, ρ)

  aux.T = air_temperature(TS)
  aux.P = air_pressure(TS) # Test with dry atmosphere
  aux.ql = PhasePartition(TS).liq
  aux.soundspeed_air = soundspeed_air(TS)
end

