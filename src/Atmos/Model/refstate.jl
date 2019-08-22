### Reference state

abstract type AbstractReferenceState{T} end
struct ReferenceState{T} <: AbstractReferenceState{T}
  T_min::T
  T_surface::T
  lapse_rate::T
end

vars_state(m::AbstractReferenceState    , DT) = @vars()
vars_gradient(m::AbstractReferenceState , DT) = @vars()
vars_diffusive(m::AbstractReferenceState, DT) = @vars()
vars_aux(m::AbstractReferenceState      , DT) = @vars(ρ::DT, p::DT, T::DT)

"""
    init_aux!(m::ReferenceState, am::AtmosModel, state::Vars, aux::Vars)

Initialize the reference state fields:
  - T temperature
  - p presure
  - ρ density
"""
function init_aux!(m::ReferenceState, am::AtmosModel, state::Vars, aux::Vars)
  aux.refstate.T = max(m.T_surface - m.lapse_rate*z, m.T_min)
  H = R_d*aux.refstate.T/grav
  z = aux.coord[3]
  aux.refstate.p = MSLP*exp(-1/H*(log(m.T_surface) -log(m.T_min) + m.lapse_rate*z - m.T_surface + m.T_min))
  aux.refstate.ρ = aux.refstate.p/(R_d*aux.refstate.T)
end
