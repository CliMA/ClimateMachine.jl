using CLIMA.PlanetParameters
export NoRadiation, StevensRadiation

abstract type RadiationModel end

vars_state(::RadiationModel, FT) = @vars()
vars_aux(::RadiationModel, FT) = @vars()
vars_integrals(::RadiationModel, FT) = @vars()

function atmos_nodal_update_aux!(::RadiationModel, ::AtmosModel, state::Vars,
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
struct StevensRadiation{FT} <: RadiationModel
  "κ [m^2/s] "
  κ::FT
  "α_z Troposphere cooling parameter [m^(-4/3)]"
  α_z::FT
  "z_i Inversion height [m]"
  z_i::FT
  "ρ_i Density"
  ρ_i::FT
  "D_subsidence Large scale divergence [s^(-1)]"
  D_subsidence::FT
  "F₀ Radiative flux parameter [W/m^2]"
  F_0::FT
  "F₁ Radiative flux parameter [W/m^2]"
  F_1::FT
end
vars_integrals(m::StevensRadiation, FT) = @vars(∂κLWP::FT)
function integrate_aux!(m::StevensRadiation, integrand::Vars, state::Vars, aux::Vars)
  FT = eltype(state)
  integrand.radiation.∂κLWP = state.ρ * m.κ * aux.moisture.q_liq
end
function flux_radiation!(m::StevensRadiation, flux::Grad, state::Vars,
                         aux::Vars, t::Real)
  FT = eltype(flux)
  z = aux.orientation.Φ/grav
  Δz_i = max(z - m.z_i, -zero(FT))
  # Constants
  cloud_top_cooling = m.F_0 * exp(-aux.∫dnz.radiation.∂κLWP)
  cloud_base_warming = m.F_1 * exp(-aux.∫dz.radiation.∂κLWP)
  free_troposphere_cooling = m.ρ_i * FT(cp_d) * m.D_subsidence * m.α_z * ((cbrt(Δz_i))^4 / 4 + m.z_i * cbrt(Δz_i))
  F_rad = cloud_base_warming + cloud_base_warming + free_troposphere_cooling
  flux.ρe += SVector(FT(0), 
                     FT(0), 
                     F_rad)
end
function preodefun!(m::StevensRadiation, aux::Vars, state::Vars, t::Real)
end
