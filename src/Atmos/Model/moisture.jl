#### Moisture component in atmosphere model
abstract type MoistureModel end

vars_state(::MoistureModel, T) = Tuple{}
vars_transform(::MoistureModel, T) = Tuple{}
vars_diffusive(::MoistureModel, T) = Tuple{}
vars_aux(::MoistureModel, T) = Tuple{}

function diffusive!(::MoistureModel, diffusive, ∇transform, state, aux, t, ν)
end
function flux_diffusive!(::MoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function preodefun_elem!(::MoistureModel, aux::Vars, state::Vars, t::Real)
end
function gradtransform!(::MoistureModel, transform::Vars, state::Vars, aux::Vars, t::Real)
end



struct EquilMoist <: MoistureModel
end
vars_state(::EquilMoist,T) = NamedTuple{(:ρq_tot,),Tuple{T}}
vars_transform(::EquilMoist,T) = NamedTuple{(:q_tot),Tuple{T}}
vars_diffusive(::EquilMoist,T) = NamedTuple{(:ρd_q_tot, :ρJ_ρD), 
        Tuple{SVector{3,T},SVector{3,T}}
vars_aux(::EquilMoist,T) = NamedTuple{(:T,:P,:q_liq),Tuple{T,T,T}}


function pressure(::EquilMoist, state::Vars, aux::Vars, t::Real)
  aux.moisture.P
end
function soundspeed(::EquilMoist, state::Vars, aux::Vars, t::Real)
  aux.moisture.T
end

function gradtransform!(::EquilMoist, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1 / state.ρ
  transform.moisture.q_tot = ρinv * state.moisture.ρq_tot
  transform.moisture.T = aux.moisture.T
end


function diffusive!(::EquilMoist, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, ν::Union{Real,AbstractMatrix})
  # turbulent Prandtl number
  diag_ν = ν isa Real ? ν : diag(ν) # either a scalar or vector
  D_q_tot = diag_ν / Prandtl_t # either a scalar or vector
  
  # diffusive flux of q_tot
  diffusive.moisture.ρd_q_tot = state.ρ * (-D_q_tot) .* ∇transform.moisture.q_tot # a vector

  # J is the conductive or SGS turbulent flux of sensible heat per unit mass
  ρJ = state.ρ * (cp_d / Prandtl_t) * ν * ∇transform.moisture.T

  T = aux.moisture.T
  u = state.ρu / state.ρ

  I_vap = cv_v * (T - T_0) + e_int_v0
  I_liq = cv_l * (T - T_0)
  I_ice = cv_v * (T - T_0) - e_int_i0
  e_kin = 0.5 * sum(abs2,u)
  e_pot = grav * aux.coord.z
  e_tot_vap = e_kin + e_pot + I_vap
  e_tot_liq = e_kin + e_pot + I_liq
  e_tot_ice = e_kin + e_pot + I_ice
  
  # total specific energy flux
  ρD = state.ρ * ((e_tot_v + R_v*T)*q_vap + e_tot_l*q_liq + e_tot_ice*q_ice) * u

  diffusive.moisture.ρJ_ρD = ρJ + ρD
end

function flux_diffusive!(m::EquilMoist, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  flux.ρ += diffusive.moisture.ρd_q_tot 
  flux.ρu += diffusive.moisture.ρd_q_tot .* u' 
  flux.ρe += diffusive.moisture.ρJ_ρD
  
  flux.moisture.ρq_tot = diffusive.moisture.ρd_q_tot
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
