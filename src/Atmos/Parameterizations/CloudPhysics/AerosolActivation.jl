"""
    Aerosol activation module, which includes:

  - mean hygroscopicity for each mode of an aerosol model
  - critical supersaturation for each mode of an aerosol model
  - maximum supersaturation for an entire aerosol model
  - total number of particles actived in a system given an aerosol model
  - a number of helper functions
"""
module AerosolActivation

using SpecialFunctions

using Thermodynamics

using ClimateMachine.AerosolModel
using ClimateMachine.Microphysics: G_func

using CLIMAParameters
using CLIMAParameters: gas_constant
using CLIMAParameters.Planet:
    ρ_cloud_liq, R_v, grav, molmass_water, molmass_dryair, cp_d#, surface_tension_coeff

using CLIMAParameters.Atmos.Microphysics: K_therm, D_vapor

const APS = AbstractParameterSet

export mean_hygroscopicity
export max_supersaturation
export total_N_activated

"""
    coeff_of_curvature(param_set, T)

  - `param_set` - abstract set with Earth's parameters
  - `T` - air temperature

Returns a curvature coefficient.
"""
function coeff_of_curvature(param_set::APS, T::FT) where {FT <: Real}

    _molmass_water::FT = molmass_water(param_set)
    _gas_constant::FT = gas_constant()
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)
    #_surface_tension::FT = surface_tension_coeff(param_set)

    _surface_tension::FT = 0.072

    return 2 * _surface_tension * _molmass_water / _ρ_cloud_liq /
           _gas_constant / T
end

"""
    mean_hygroscopicity(param_set, am)

  - `param_set` - abstract set with Earth's parameters
  - `am` - aerosol model struct

Returns a tuple of mean hygroscopicities
(one tuple element for each aerosol size distribution mode).
"""
function mean_hygroscopicity(param_set::APS, am::aerosol_model)

    _molmass_water = molmass_water(param_set)
    _ρ_cloud_liq = ρ_cloud_liq(param_set)

    return ntuple(length(am.modes)) do i

        mode_i = am.modes[i]

        nom = sum(1:(mode_i.n_components)) do j
            mode_i.mass_mix_ratio[j] *
            mode_i.dissoc[j] *
            mode_i.osmotic_coeff[j] *
            mode_i.soluble_mass_frac[j] / mode_i.molar_mass[j]
        end

        den = sum(1:(mode_i.n_components)) do j
            mode_i.mass_mix_ratio[j] / mode_i.aerosol_density[j]
        end

        nom / den * _molmass_water / _ρ_cloud_liq
    end
end

"""
    critical_supersaturation(param_set, am, T)

  - `param_set` - abstract set with Earth's parameters
  - `am` - aerosol model struct
  - `T` - air temperature

Returns a tuple of critical supersaturations
(one tuple element for each aerosol size distribution mode).
"""
function critical_supersaturation(
    param_set::APS,
    am::aerosol_model,
    T::FT,
) where {FT <: Real}

    A::FT = coeff_of_curvature(param_set, T)
    B = mean_hygroscopicity(param_set, am) #TODO - how to specify a type here

    return ntuple(length(am.modes)) do i
        2 / sqrt(B[i]) * (A / 3 / am.modes[i].r_dry)^(3 / 2)
    end
end

"""
    max_supersaturation(param_set, am, T, p, w)

  - `param_set` - abstract set with Earth's parameters
  - `am` - aerosol model struct
  - `T` - air temperature
  - `p` - air pressure
  - `w` - vertical velocity

Returns the maximum supersaturation.
"""
function max_supersaturation(
    param_set::APS,
    am::aerosol_model,
    T::FT,
    p::FT,
    w::FT,
) where {FT <: Real}

    _grav::FT = grav(param_set)
    _molmass_water::FT = molmass_water(param_set)
    _molmass_dryair::FT = molmass_dryair(param_set)
    _gas_constant::FT = gas_constant()
    _cp_d::FT = cp_d(param_set)
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)

    L::FT = latent_heat_vapor(param_set, T)
    p_vs::FT = saturation_vapor_pressure(param_set, T, Liquid())
    G::FT = G_func(param_set, T, Liquid())

    # eq 11, 12 in Razzak et al 1998
    α::FT =
        _grav * _molmass_water * L / _cp_d / _gas_constant / T^2 -
        _grav * _molmass_dryair / _gas_constant / T
    γ::FT =
        _gas_constant * T / p_vs / _molmass_water +
        _molmass_water * L^2 / _cp_d / p / _molmass_dryair / T

    A::FT = coeff_of_curvature(param_set, T)
    ζ::FT = 2 * A / 3 * sqrt(α * w / G)

    Sm = critical_supersaturation(param_set, am, T) # TODO how to specify the type here?

    tmp::FT = sum(1:length(am.modes)) do i

        mode_i = am.modes[i]

        f::FT = 0.5 * exp(2.5 * (log(mode_i.stdev))^2)
        g::FT = 1 + 0.25 * log(mode_i.stdev)
        η::FT = (α * w / G)^(3 / 2) / (2 * pi * _ρ_cloud_liq * γ * mode_i.N)

        1 / (Sm[i])^2 *
        (f * (ζ / η)^(3 / 2) + g * (Sm[i]^2 / (η + 3 * ζ))^(3 / 4))
    end

    return FT(1) / sqrt(tmp)
end

"""
    total_N_activated(param_set, am, T, p, w)

  - `param_set` - abstract set with Earth's parameters
  - `am` - aerosol model struct
  - `T` - air temperature
  - `p` - air pressure
  - `w` - vertical velocity

Returns the total number of activated aerosol particles.
"""
function total_N_activated(
    param_set::APS,
    am::aerosol_model,
    T::FT,
    p::FT,
    w::FT,
) where {FT <: Real}

    smax::FT = max_supersaturation(param_set, am, T, p, w)
    sm = critical_supersaturation(param_set, am, T) # TODO how to specify a type here?

    return sum(1:length(am.modes)) do i

        mode_i = am.modes[i]
        u_i::FT = 2 * log(sm[i] / smax) / 3 / sqrt(2) / log(mode_i.stdev)

        mode_i.N * (1 / 2) * (1 - erf(u_i))
    end
end

end # module AerosolActivation.jl
