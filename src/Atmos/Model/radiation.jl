using CLIMA.PlanetParameters
export NoRadiation, DYCOMSRadiation

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
function flux_radiation!(::RadiationModel, atmos::AtmosModel, flux::Grad, state::Vars,
                         aux::Vars, t::Real)
end

struct NoRadiation <: RadiationModel
end

"""
  DYCOMSRadiation <: RadiationModel

Stevens et. al (2005) approximation of longwave radiative fluxes in DYCOMS.
Analytical description as a function of the liquid water path and inversion height zᵢ

* Stevens, B. et. al. (2005) "Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus". Mon. Wea. Rev., 133, 1443–1462, https://doi.org/10.1175/MWR2930.1

# Fields

$(DocStringExtensions.FIELDS)
"""
struct DYCOMSRadiation{FT} <: RadiationModel
  "mass absorption coefficient `[m^2/kg]`"
  κ::FT
  "Troposphere cooling parameter `[m^(-4/3)]`"
  α_z::FT
  "Inversion height `[m]`"
  z_i::FT
  "Density"
  ρ_i::FT
  "Large scale divergence `[s^(-1)]`"
  D_subsidence::FT
  "Radiative flux parameter `[W/m^2]`"
  F_0::FT
  "Radiative flux parameter `[W/m^2]`"
  F_1::FT
end
vars_integrals(m::DYCOMSRadiation, FT) = @vars(attenuation_coeff::FT)
vars_aux(m::DYCOMSRadiation, FT) = @vars(Rad_flux::FT)
function integrate_aux!(m::DYCOMSRadiation, integrand::Vars, state::Vars, aux::Vars)
  FT = eltype(state)
  integrand.radiation.attenuation_coeff = state.ρ * m.κ * aux.moisture.q_liq
end
function flux_radiation!(m::DYCOMSRadiation, atmos::AtmosModel, flux::Grad, state::Vars,
                         aux::Vars, t::Real)
  FT = eltype(flux)
  z = altitude(atmos.orientation, aux)
  Δz_i = max(z - m.z_i, -zero(FT))
  # Constants
  upward_flux_from_cloud  = m.F_0 * exp(-aux.∫dnz.radiation.attenuation_coeff)
  upward_flux_from_sfc = m.F_1 * exp(-aux.∫dz.radiation.attenuation_coeff)
  free_troposphere_flux = m.ρ_i * FT(cp_d) * m.D_subsidence * m.α_z * cbrt(Δz_i) * (Δz_i/4 + m.z_i)
  F_rad = upward_flux_from_sfc + upward_flux_from_cloud + free_troposphere_flux
  ẑ = vertical_unit_vector(atmos.orientation, aux)
  flux.ρe += F_rad * ẑ
end
function preodefun!(m::DYCOMSRadiation, aux::Vars, state::Vars, t::Real)
end
