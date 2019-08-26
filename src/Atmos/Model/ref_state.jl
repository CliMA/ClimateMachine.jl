### Reference state

export IsothermalHydrostaticState, HydrostaticStateNonZeroLapseRate

"""
    ReferenceState

Reference state, for example, used as initial
condition or for linearization.
"""
abstract type ReferenceState end

vars_state(m::ReferenceState    , DT) = @vars()
vars_gradient(m::ReferenceState , DT) = @vars()
vars_diffusive(m::ReferenceState, DT) = @vars()
vars_aux(m::ReferenceState      , DT) = @vars(ρ::DT, p::DT, T::DT, ρe::DT, ρq_tot::DT)

function init_aux_hydrostatic!(m::ReferenceState, aux::Vars)
  DT = eltype(aux)
  T = aux.ref_state.T
  ρ = aux.ref_state.p/(R_d*T)
  aux.ref_state.ρ = ρ
  q_vap_sat = q_vap_saturation(T, ρ)
  ρq_tot = ρ*m.RH*q_vap_sat
  aux.ref_state.ρq_tot = ρq_tot
  q_pt = PhasePartition(ρq_tot, DT(0), DT(0))
  aux.ref_state.ρe = ρ*MoistThermodynamics.internal_energy(T, q_pt)
  e_kin = DT(0)
  e_pot = aux.orientation.Φ
  aux.ref_state.ρe = ρ*MoistThermodynamics.total_energy(e_kin, e_pot, T, q_pt)
end


"""
    IsothermalHydrostaticState{DT} <: ReferenceState

A hydrostatic state assuming a uniform temperature profile,
given
 - `T_surface` surface temperature
 - `RH` relative humidity
"""
struct IsothermalHydrostaticState{DT} <: ReferenceState
  T_surface::DT
  RH::DT
end

function init_aux!(m::IsothermalHydrostaticState, aux::Vars)
  T_s = m.T_surface
  e_pot = aux.orientation.Φ

  aux.ref_state.T = T_s
  aux.ref_state.p = MSLP*exp(-e_pot/(R_d*T_s))
  init_aux_hydrostatic!(m, aux)
end

"""
    HydrostaticStateNonZeroLapseRate{DT} <: ReferenceState

A hydrostatic state assuming a non-zero lapse rate, given
 - `T_min` minimum temperature
 - `T_surface` surface temperature
 - `Γ` lapse rate
 - `RH` relative humidity
"""
struct HydrostaticStateNonZeroLapseRate{DT} <: ReferenceState
  T_min::DT
  T_surface::DT
  Γ::DT
  RH::DT
end

function init_aux!(m::HydrostaticStateNonZeroLapseRate, aux::Vars)
  Γ = m.Γ
  T_s = m.T_surface
  T_min = m.T_min
  e_pot = aux.orientation.Φ
  z = e_pot/grav

  T = max(T_s - Γ*z, T_min)
  H = R_d*T/grav
  z_t = (T_s-T_min)/Γ
  H_min = R_d*T_min/grav
  if z<z_t
    p = (1-Γ*z/T_s)^(grav/(R_d*Γ))
  else
    p = (T_min/T_s)^(grav/(R_d*Γ))*exp(-(z-z_t)/H_min)
  end
  aux.ref_state.T = T
  aux.ref_state.p = p
  init_aux_hydrostatic!(m, aux)
end

