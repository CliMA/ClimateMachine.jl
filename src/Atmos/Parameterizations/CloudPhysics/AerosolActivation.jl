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
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, molmass_water, molmass_dryair, LH_v0, cp_d
using CLIMAParameters.Atmos.Microphysics: K_therm, D_vapor

const APS = AbstractParameterSet

export mean_hygroscopicity
export critical_supersaturation
export max_supersaturation
export total_N_activated

"""
    coeff_of_curvature(param_set, T)

  - `param_set` - abstract set with Earth's parameters
  - `T` - air temperature

Returns a curvature coefficient.
"""
function coeff_of_curvature(param_set::APS, T::FT) where {FT <: Real}

    _molmass_water = molmass_water(param_set)
    _gas_constant = gas_constant()
    _ρ_cloud_liq = ρ_cloud_liq(param_set)

    surface_tension = 0.072 #TODO - take it from CLIMAParameters

    return 2 * surface_tension * _molmass_water / _ρ_cloud_liq / _gas_constant / T
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
        n_comps = length(mode_i.particle_density)

        nom = sum(1:n_comps) do j
            mode_i.mass_mix_ratio[j] * mode_i.dissoc[j] *
            mode_i.osmotic_coeff[j] * mode_i.soluble_mass_frac[j] / mode_i.molar_mass[j]
        end

        den = sum(1:n_comps) do j
            mode_i.mass_mix_ratio[j] / mode_i.aerosol_density[j]
        end

        nom / den * _molmass_water / _ρ_cloud_liq
    end
end

"""
TO DO: DOCSTRING

  - `param_set` - abstract set with Earth's parameters
  - `am` - aerosol model struct
  - `T` - air temperature
"""
function critical_supersaturation(param_set::APS, am::aerosol_model, T::FT) where {FT <: Real}

    coeff_of_curve = coeff_of_curvature(param_set, T)
    mh = mean_hygroscopicity(param_set, am)

    return ntuple(length(am.modes)) do i
        mode_i = am.modes[i]
        # weighted average of mode radius
        n_comps = length(mode_i.particle_density)
        numerator = sum(1:n_comps) do j
            mode_i.dry_radius[j]*mode_i.particle_density[j]
        end
        denominator = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_radius = numerator/denominator
        exp1 = 2 / (mh[i])^(.5)
        exp2 = (coeff_of_curve / 3 * avg_radius)^(3/2)
        exp1*exp2
    end
end

"""
TO DO: DOCSTRING

  - `param_set` - abstract set with Earth's parameters
  - `am` - aerosol model struct
  - `T` - air temperature
  - `p` - air pressure
  - `w` - vertical velocity
"""
function max_supersaturation(param_set::APS, am::aerosol_model, T::FT, p::FT, w::FT) where {FT <: Real}

    _grav::FT = grav(param_set)
    _molmass_water::FT = molmass_water(param_set)
    _molmass_dryair::FT = molmass_dryair(param_set)
    _gas_constant::FT = gas_constant()
    _R_v::FT = R_v(param_set)
    _cp_d::FT = cp_d(param_set)
    _K_therm::FT = K_therm(param_set)
    _D_vapor::FT = D_vapor(param_set)
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)

    L::FT = latent_heat_vapor(param_set, T)
    p_vs::FT = saturation_vapor_pressure(param_set, T, Liquid())
    _G::FT = G_func(param_set, T, Liquid())

    # eq 11, 12 in Razzak et al 1998
    α::FT = _grav * _molmass_water * L / _cp_d / _gas_constant / T^2 -
            _grav * _molmass_dryair / _gas_constant / T
    γ::FT = _gas_constant * T / p_vs / _molmass_water +
            _molmass_water * L^2 / _cp_d / p / _molmass_dryair / T

    A = coeff_of_curvature(param_set, T)
    Sm = critical_supersaturation(param_set, am, T)

    X = sum(1:length(am.modes)) do i

        mode_i = am.modes[i]

        # weighted avgs of diff params:
        n_comps = length(mode_i.particle_density)
        # radius_stdev
        num = sum(1:n_comps) do j
            mode_i.particle_density[j]  *  mode_i.radius_stdev[j]
        end
        den = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_radius_stdev = num/den

        total_particles = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end

        f = 0.5  *  exp(2.5  *  (log(avg_radius_stdev))^2 )
        g = 1 + 0.25  *  log(avg_radius_stdev)

        zeta = (2 * A * (1/3)) * ((α * w) / _G)^(.5)
        eta = (((α * w) / (_G))^(3/2)) / (2 * pi * _ρ_cloud_liq * γ * total_particles)

        exp1 = 1/(Sm[i])^2
        exp2 = f*(zeta/eta)^(3/2)
        exp3 = g*((Sm[i]^2)/(eta+3*zeta))^(3/4)

        exp1*(exp2+exp3)
    end
    return (X)^(1/2)

end

"""
TO DO: DOCSTRING

  - `param_set` - abstract set with Earth's parameters
  - `am` - aerosol model struct
  - `T` - air temperature
  - `p` - air pressure
  - `w` - vertical velocity
"""
function total_N_activated(param_set::APS, am::aerosol_model, T::FT, p::FT, w::FT) where {FT <: Real}

    smax = max_supersaturation(param_set, am, T, p, w)
    sm = critical_supersaturation(param_set, am, T)

    return sum(1:length(am.modes)) do i
        mode_i = am.modes[i]
        # weighted avgs of diff params:
        n_comps = length(mode_i.particle_density)
        # radius_stdev
        num = sum(1:n_comps) do j
            mode_i.particle_density[j]  *  mode_i.radius_stdev[j]
        end
        den = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_radius_stdev = num/den

        total_particles = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end

        utop = 2*log(sm[i]/smax)
        ubottom = 3*(2^.5)*log(avg_radius_stdev)
        u = utop/ubottom
        total_particles*(1/2)*(1-erf(u))
    end
end

end # module AerosolActivation.jl
