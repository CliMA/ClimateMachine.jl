using SpecialFunctions
using Test

using Thermodynamics

using ClimateMachine.AerosolModel: mode, aerosol_model
using ClimateMachine.AerosolActivation

using CLIMAParameters
using CLIMAParameters: gas_constant
using CLIMAParameters.Planet: molmass_water, ρ_cloud_liq, grav, cp_d#, surface_tension_coeff
using CLIMAParameters.Atmos.Microphysics

struct EarthParameterSet <: AbstractEarthParameterSet end
const EPS = EarthParameterSet
const param_set = EarthParameterSet()

T = 283.15     # air temperature
p = 100000.0   # air pressure
w = 5.0        # vertical velocity

# TODO - move areosol properties to CLIMAParameters

# Sea Salt--universal parameters
osmotic_coeff_seasalt = 0.9
dissoc_seasalt = 2.0
molar_mass_seasalt = 0.058443
rho_seasalt = 2170.0
soluble_mass_frac_seasalt = 1.0

# TODO: Dust parameters (just copy and pasted seasalt values rn)
# Dust--universal parameters
osmotic_coeff_dust = 0.9
dissoc_dust = 2.0
molar_mass_dust = 0.058443
rho_dust = 2170.0
soluble_mass_frac_dust = 1.0

# Accumulation mode
r_dry_accum = 0.243 * 1e-6 # μm
stdev_accum = 1.4          # -
N_accum = 100.0 * 1e6      # 1/m3

# Coarse Mode
r_dry_coarse = 1.5 * 1e-6  # μm
stdev_coarse = 2.1         # -
N_coarse = 1.0 * 1e6       # 1/m3

# Building the test structures:

# 1) create aerosol modes
accum_mode_seasalt = mode(
    r_dry_accum,
    stdev_accum,
    N_accum,
    (1.0,), # mass mix ratio TODO
    (soluble_mass_frac_seasalt,),
    (osmotic_coeff_seasalt,),
    (molar_mass_seasalt,),
    (dissoc_seasalt,),
    (rho_seasalt,),
    1,
)

coarse_mode_seasalt = mode(
    r_dry_coarse,
    stdev_coarse,
    N_coarse,
    (1.0,), # mass mix ratio TODO
    (soluble_mass_frac_seasalt,),
    (osmotic_coeff_seasalt,),
    (molar_mass_seasalt,),
    (dissoc_seasalt,),
    (rho_seasalt,),
    1,
)

accum_mode_seasalt_dust = mode(
    r_dry_accum,
    stdev_accum,
    N_accum,
    (0.5, 0.5), # mass mix ratio TODO
    (soluble_mass_frac_seasalt, soluble_mass_frac_dust),
    (osmotic_coeff_seasalt, osmotic_coeff_dust),
    (molar_mass_seasalt, molar_mass_dust),
    (dissoc_seasalt, dissoc_dust),
    (rho_seasalt, rho_dust),
    2,
)

coarse_mode_seasalt_dust = mode(
    r_dry_coarse,
    stdev_coarse,
    N_coarse,
    (0.25, 0.75), # mass mix ratio TODO
    (soluble_mass_frac_seasalt, soluble_mass_frac_dust),
    (osmotic_coeff_seasalt, osmotic_coeff_dust),
    (molar_mass_seasalt, molar_mass_dust),
    (dissoc_seasalt, dissoc_dust),
    (rho_seasalt, rho_dust),
    2,
)

# 2) create aerosol models
AM_1 = aerosol_model((accum_mode_seasalt,))
AM_2 = aerosol_model((coarse_mode_seasalt,))
AM_3 = aerosol_model((accum_mode_seasalt, coarse_mode_seasalt))
AM_4 = aerosol_model((accum_mode_seasalt_dust,))
AM_5 = aerosol_model((accum_mode_seasalt_dust, coarse_mode_seasalt_dust))

# 3) bundle them together
AM_test_cases = [AM_1, AM_2, AM_3, AM_4, AM_5]



function tp_coeff_of_curve(param_set::EPS, T::FT) where {FT <: Real}

    _gas_const::FT = gas_constant()
    _molmass_water::FT = molmass_water(param_set)
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)
    #_surface_tension::FT  = surface_tension_coeff(param_set)

    _surface_tension::FT = 0.072

    return 2 * _surface_tension * _molmass_water / _ρ_cloud_liq / _gas_const / T
end

function tp_mean_hygroscopicity(param_set::EPS, am::aerosol_model)

    _molmass_water = molmass_water(param_set)
    _ρ_cloud_liq = ρ_cloud_liq(param_set)

    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        numerator = sum(num_of_comp) do j
            mode_i.osmotic_coeff[j] *
            mode_i.mass_mix_ratio[j] *
            mode_i.dissoc[j] *
            mode_i.soluble_mass_frac[j] *
            1 / mode_i.molar_mass[j]
        end
        denominator = sum(num_of_comp) do j
            mode_i.mass_mix_ratio[j] / mode_i.aerosol_density[j]
        end
        (numerator / denominator) * (_molmass_water / _ρ_cloud_liq)
    end
end

# questions about temp,
# need to fill equations: , alpha --> 1.0, eta() --> 2.0
# Key:
# surface tension == A
# surface_tension_effects(zeta) --> 3.0

function alpha(param_set::EPS, T::FT, aerosol_mass::FT) where {FT <: Real}

    _molmass_water::FT = molmass_water(param_set)
    _grav::FT = grav(param_set)
    _gas_constant::FT = gas_constant()
    _cp_d::FT = cp_d(param_set)

    L::FT = latent_heat_vapor(param_set, T)

    return _grav * _molmass_water * L / (_cp_d * _gas_constant * T^2) -
           _grav * aerosol_mass / (_gas_constant * T)
end

function gamma(
    param_set::EPS,
    T::FT,
    aerosol_mass::FT,
    press::FT,
) where {FT <: Real}

    _molmass_water::FT = molmass_water(param_set)
    _gas_constant::FT = gas_constant()
    _cp_d::FT = cp_d(param_set)

    L::FT = latent_heat_vapor(param_set, T)
    p_vs::FT = saturation_vapor_pressure(param_set, T, Liquid())

    return _gas_constant * T / (p_vs * _molmass_water) +
           _molmass_water * L^2 / (_cp_d * press * aerosol_mass * T)
end

function zeta(
    param_set::EPS,
    T::FT,
    aerosol_mass::FT,
    updraft_velocity::FT,
    G_diff::FT,
) where {FT <: Real}
    return 2 * tp_coeff_of_curve(param_set, T) / 3 *
           (
        alpha(param_set, T, aerosol_mass) * updraft_velocity / G_diff
    )^(1 / 2)
end

function eta(
    param_set::EPS,
    temp::Float64,
    aerosol_mass::Float64,
    number_concentration::Float64,
    G_diff::Float64,
    updraft_velocity::Float64,
    press::Float64,
)

    _ρ_cloud_liq = ρ_cloud_liq(param_set)

    return alpha(param_set, temp, aerosol_mass) * updraft_velocity /
           G_diff^(3 / 2) / (
        2 *
        pi *
        _ρ_cloud_liq *
        gamma(param_set, temp, aerosol_mass, press) *
        number_concentration
    )
end

function tp_max_super_sat(
    param_set::EPS,
    am::aerosol_model,
    temp::Float64,
    updraft_velocity::Float64,
    G_diff::Float64,
    press::Float64,
)
    mean_hygro = tp_mean_hygroscopicity(param_set, am)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        a = sum(num_of_comp) do j
            f = 0.5 * exp(2.5 * log(mode_i.stdev)^2)
            g = 1 + 0.25 * log(mode_i.stdev)
            coeff_of_curve = tp_coeff_of_curve(param_set, temp)
            surface_tension_effects = zeta(
                param_set,
                temp,
                mode_i.molar_mass[j],
                updraft_velocity,
                G_diff,
            )
            critsat =
                2 / sqrt(mean_hygro[i]) *
                (coeff_of_curve / (3 * mode_i.r_dry))^(3 / 2) # FILL

            eta_value = eta(
                param_set,
                temp,
                mode_i.molar_mass[j],
                mode_i.N,
                G_diff,
                updraft_velocity,
                press,
            )

            1 / (critsat^2) * (
                f * (surface_tension_effects / eta_value)^(3 / 2) +
                g * (critsat^2) /
                (eta_value + 3 * surface_tension_effects)^(3 / 4)
            )

        end
        a^(1 / 2)

    end
end

function tp_critical_supersaturation(
    param_set::EPS,
    am::aerosol_model,
    temp::Float64,
)
    mean_hygro = tp_mean_hygroscopicity(param_set, am)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        a = sum(num_of_comp) do j
            2 / sqrt(mean_hygro[i]) *
            (tp_coeff_of_curve(param_set, temp) / (3 * mode_i.r_dry))^(3 / 2)
        end
        a
    end

end

function tp_total_n_act(
    param_set::EPS,
    am::aerosol_model,
    temp::Float64,
    updraft_velocity::Float64,
    G_diff::Float64,
    press::Float64,
)
    critical_supersaturation = tp_critical_supersaturation(param_set, am, temp)
    max_supersat =
        tp_max_super_sat(param_set, am, temp, updraft_velocity, G_diff, press)
    values = ntuple(am.N) do i
        mode_i = am.modes[i]

        sigma = mode_i.stdev
        u_top = 2 * log(critical_supersaturation[i] / max_supersat[i])
        u_bottom = 3 * sqrt(2) * log(sigma)
        u = u_top / u_bottom
        mode_i.N * 1 / 2 * (1 - erf(u))
    end
    summation = 0.0
    for i in range(1, length = length(values))
        summation += values[i]
    end
    return summation
end



@testset "mean_hygroscopicity" begin

    println("----------")
    println("mean_hygroscopicity: ")
    println(tp_mean_hygroscopicity(param_set, AM_1))
    println(mean_hygroscopicity(param_set, AM_1))

    for AM in AM_test_cases
        @test all(
            tp_mean_hygroscopicity(param_set, AM) .≈
            mean_hygroscopicity(param_set, AM),
        )
    end
    println(" ")
end

@testset "max_supersaturation" begin

    println("----------")
    println("max_supersaturation: ")
    println(tp_max_super_sat(param_set, AM_1, 2.0, 3.0, 4.0, 1.0))
    println(max_supersaturation(param_set, AM_1, T, p, w))

    # TODO
    #for AM in AM_test_cases
    #    @test all(
    #        tp_max_super_sat(param_set, AM, 2.0, 3.0, 4.0, 1.0) .≈
    #        max_supersaturation(param_set, AM, T, p, w)
    #    )
    #end

    println(" ")
end

@testset "total_n_act" begin

    println("----------")
    println("total_N_act: ")
    println(tp_total_n_act(param_set, AM_1, 2.0, 3.0, 4.0, 1.0))
    println(total_N_activated(param_set, AM_1, T, p, w))

    # TODO
    #for AM in AM_test_cases
    #    @test all(
    #        tp_total_n_act(param_set, AM, 2.0, 3.0, 4.0, 1.0) .≈
    #        total_N_activated(param_set, AM, T, p, w)
    #    )
    #end

    println(" ")
end
