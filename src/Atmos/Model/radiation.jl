using CLIMA.PlanetParameters
export NoRadiation, StevensRadiation

abstract type RadiationModel end

vars_state(::RadiationModel, DT) = @vars()
vars_gradient(::RadiationModel, DT) = @vars()
vars_diffusive(::RadiationModel, DT) = @vars()
vars_aux(::RadiationModel, DT) = @vars()
vars_integrals(::RadiationModel, DT) = @vars()

function atmos_nodal_update_aux!(::RadiationModel, ::AtmosModel, state::Vars, diffusive::Vars,
                                 aux::Vars, t::Real)
end
function preodefun!(::RadiationModel, aux::Vars, state::Vars, t::Real)
end
function integrate_aux!(::RadiationModel, integ::Vars, state::Vars, aux::Vars)
end
function flux_radiation!(::RadiationModel, flux::Grad, state::Vars,
                         aux::Vars, t::Real)
end

struct NoRadiation <: RadiationModel
end

"""
  StevensRadiation <: RadiationModel

Stevens et. al (2005) version of the δ-four stream model used to represent radiative transfer. 
Analytical description as a function of the liquid water path and inversion height zᵢ
"""
struct StevensRadiation{DT} <: RadiationModel
  "κ [m^2/s] "
  κ::DT
  "α_z Troposphere cooling parameter [m^(-4/3)]"
  α_z::DT
  "z_i Inversion height [m]"
  z_i::DT
  "ρ_i Density"
  ρ_i::DT
  "D_subsidence Large scale divergence [s^(-1)]"
  D_subsidence::DT
  "F₀ Radiative flux parameter [W/m^2]"
  F_0::DT
  "F₁ Radiative flux parameter [W/m^2]"
  F_1::DT
end
vars_integrals(m::StevensRadiation, DT) = @vars(∂κLWP::DT)
function integrate_aux!(m::StevensRadiation, integrand::Vars, state::Vars, aux::Vars)
  DT = eltype(state)
  integrand.radiation.∂κLWP = state.ρ * m.κ * aux.moisture.q_liq
end
function flux_radiation!(m::StevensRadiation, flux::Grad, state::Vars,
                         aux::Vars, t::Real)
  DT = eltype(flux)
  z = aux.orientation.Φ/grav
  Δz_i = max(z - m.z_i, -zero(DT))
  # Constants
  cloud_top_cooling = m.F_0 * exp(-aux.∫dnz.radiation.∂κLWP)
  cloud_base_warming = m.F_1 * exp(-aux.∫dz.radiation.∂κLWP)
  free_troposphere_cooling = m.ρ_i * DT(cp_d) * m.D_subsidence * m.α_z * ((cbrt(Δz_i))^4 / 4 + m.z_i * cbrt(Δz_i))
  F_rad = cloud_base_warming + cloud_base_warming + free_troposphere_cooling
  flux.ρe += SVector(DT(0), 
                     DT(0), 
                     F_rad)
end
function preodefun!(m::StevensRadiation, aux::Vars, state::Vars, t::Real)
end
