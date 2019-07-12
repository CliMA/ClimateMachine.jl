#### Moisture component in atmosphere model
abstract type MoistureModel end

vars_state(::MoistureModel, T) = Tuple{}
vars_transform(::MoistureModel, T) = Tuple{}
vars_diffusive(::MoistureModel, T) = Tuple{}
vars_aux(::MoistureModel, T) = Tuple{}

function preodefun_elem!(::MoistureModel, aux::Vars, state::Vars, t::Real)
end


struct DryModel <: MoistureModel
end
using CLIMA.PlanetParameters: γ_exact
function pressure(::DryModel, state::Vars, aux::Vars, t::Real)
  T = eltype(state)
  γ = T(γ_exact)
  ρinv = 1 / state.ρ
  P = (γ-1)*(state.ρe - ρinv/2 * sum(abs2, state.ρu))
end

function soundspeed(::DryModel, state::Vars, aux::Vars, t::Real)
  T = eltype(state)
  γ = T(γ_exact)

  ρinv = 1 / state.ρ
  P = pressure(m.moisture, state, aux, t)
  sqrt(ρinv * γ * P)
end

function diffusive!(::DryModel, diffusive, ∇transform, state, aux, t, ν)
end
function flux_diffusive!(m::MoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end



struct EquilMoist <: MoistureModel
end
vars_state(::EquilMoist,T) = NamedTuple{(:ρq_tot,),Tuple{T}}
vars_transform(::EquilMoist,T) = NamedTuple{(:q_tot),Tuple{T}}
vars_diffusive(::EquilMoist,T) = NamedTuple{(:d_q_tot, :J),Tuple{SVector{3,T},SVector{3,T}}}
vars_aux(::EquilMoist,T) = NamedTuple{(:T,:P,:q_liq),Tuple{T,T,T}}


function pressure(::MoistureModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  aux.moisture.P
end
function soundspeed(::MoistureModel, state::Vars, aux::Vars, t::Real)
  aux.moisture.T
end

function gradtransform!(::MoistureModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1 / state.ρ
  transform.moisture.q_tot = ρinv * state.moisture.ρq_tot
  transform.moisture.T = aux.moisture.T
end


function diffusive!(::EquilMoist, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, ν::Union{Real,AbstractMatrix})
  # turbulent Prandtl number
  diag_ν = ν isa Real ? ν : diag(ν) # either a scalar or vector
  D_q_tot = diag_ν / Prandtl_t # either a scalar or vector
  
  # diffusive flux of q_tot
  diffusive.moisture.d_q_tot = (-D_q_tot) .* ∇transform.moisture.q_tot # a vector

  # J is the conductive or SGS turbulent flux of sensible heat per unit mass
  diffusive.moisture.J = (cp_d / Prandtl_t) * ν * ∇transform.moisture.T
end

function flux_diffusive!(m::EquilMoist, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  ρd_q_tot = state.ρ * diffusive.moisture.d_q_tot

  # TODO: this violates mass preservation?
  # flux.ρ += ρd_q_tot 
  flux.ρu += ρd_q_tot .* u' 
  flux.ρe += state.ρ * diffusive.moisture.J

  
  flux.moisture.ρq_tot = ρd_q_tot
end




function preodefun_elem!(m::EquilMoist, aux::Vars, state::Vars, t::Real)
  T = eltype(aux)

  ρinv = 1 / state.ρ
  e_int = (state.ρe - ρinv * (state.ρu^2 + state.ρv^2 + state.ρw^2) / 2) * ρinv - T(grav) * aux.z
  qt = state.ρqt * ρinv

  TS = PhaseEquil(e_int, q_tot, ρ)

  aux.T = air_temperature(TS)
  aux.P = air_pressure(TS) # Test with dry atmosphere
  aux.ql = PhasePartition(TS).liq
  aux.soundspeed_air = soundspeed_air(TS)
end
