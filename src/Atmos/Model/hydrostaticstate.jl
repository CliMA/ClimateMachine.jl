### Hydrostatic state

"""
    AbstractHydrostaticState{DT}

Hydrostatic state, potentially used as a reference state.
"""
abstract type AbstractHydrostaticState{DT} end

vars_state(m::AbstractHydrostaticState    , DT) = @vars()
vars_gradient(m::AbstractHydrostaticState , DT) = @vars()
vars_diffusive(m::AbstractHydrostaticState, DT) = @vars()
vars_aux(m::AbstractHydrostaticState      , DT) = @vars(ρ::DT, p::DT, T::DT, ρe::DT, ρq_tot::DT)

function compute_additional_fields!(m::AbstractHydrostaticState{DT}, aux::Vars) where DT
  T = aux.hydrostatic.T
  ρ = aux.hydrostatic.p/(R_d*T)
  aux.hydrostatic.ρ = ρ
  q_vap_sat = q_vap_saturation(T, ρ)
  ρq_tot = ρ*m.relative_humidity*q_vap_sat
  aux.hydrostatic.ρq_tot = ρq_tot
  q_pt = PhasePartition(ρq_tot, DT(0), DT(0))
  aux.hydrostatic.ρe = ρ*MoistThermodynamics.internal_energy(T, q_pt)
end


"""
    HydrostaticStateIsothermalAtmos{DT} <: AbstractHydrostaticState{DT}

"""
struct HydrostaticStateIsothermalAtmos{DT} <: AbstractHydrostaticState{DT}
  T_surface::DT
  relative_humidity::DT
end
export HydrostaticStateIsothermalAtmos

"""
    init_aux!(m::HydrostaticStateIsothermalAtmos, aux::Vars)

Initialize the isothermal hydrostatic state fields
"""
function init_aux!(m::HydrostaticStateIsothermalAtmos, aux::Vars)
  T_s = m.T_surface
  z = aux.coord[3]

  aux.hydrostatic.T = T_s
  H_sfc = R_d*T_s/grav
  aux.hydrostatic.p = MSLP*exp(-z/H_sfc)
  compute_additional_fields!(m, aux)
end

"""
    HydrostaticStateNonZeroLapseRate{DT} <: AbstractHydrostaticState{DT}

"""
struct HydrostaticStateNonZeroLapseRate{DT} <: AbstractHydrostaticState{DT}
  T_min::DT
  T_surface::DT
  lapse_rate::DT
  relative_humidity::DT
end
export HydrostaticStateNonZeroLapseRate

"""
    init_aux!(m::HydrostaticStateNonZeroLapseRate, aux::Vars)

Initialize the hydrostatic non-zero lapse rate state fields
"""
function init_aux!(m::HydrostaticStateNonZeroLapseRate, aux::Vars)
  Γ = m.lapse_rate
  T_s = m.T_surface
  T_min = m.T_min
  z = aux.coord[3]

  T = max(T_s - Γ*z, T_min)
  H = R_d*T/grav
  z_t = (T_s-T_min)/Γ
  H_min = R_d*T_min/grav
  if z<z_t
    p = (1-Γ*z/T_s)^(grav/(R_d*Γ))
  else
    p = (T_min/T_s)^(grav/(R_d*Γ))*exp(-(z-z_t)/H_min)
  end
  aux.hydrostatic.T = T
  aux.hydrostatic.p = p
  compute_additional_fields!(m, aux)
end

