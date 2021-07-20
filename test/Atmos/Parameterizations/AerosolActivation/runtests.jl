
using SpecialFunctions
using Test

using Thermodynamics

using ClimateMachine.AerosolModel: mode, aerosol_model
using ClimateMachine.AerosolActivation

using ClimateMachine.Microphysics: G_func

using CLIMAParameters
using CLIMAParameters: gas_constant
using CLIMAParameters.Planet: molmass_water, ρ_cloud_liq, grav, cp_d, molmass_dryair
using CLIMAParameters.Atmos.Microphysics


struct EarthParameterSet <: AbstractEarthParameterSet end
const EPS = EarthParameterSet
const param_set = EarthParameterSet()

# include("/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Mode_creation.jl")
include("/home/idularaz/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Mode_creation.jl")
# prinln("length of AM", length(AM.N))
# CONSTANTS FOR TEST
T = 283.15     # air temperature
p = 100000.0   # air pressure
w = 5.0        # vertical velocity

# TODO - move areosol properties to CLIMAParameters


# FUNCTIONS FOR TESTS

"""    
    functionality: calculates the coefficient of curvature
    parameters: parameter set, temperature
    returns: scalar coefficeint of curvature
"""
function tp_coeff_of_curve(param_set::EPS, T::FT) where {FT <: Real}

    _molmass_water::FT = molmass_water(param_set)
    _gas_const::FT = gas_constant()
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)

    _surface_tension::FT  = 0.072   # TODO - use from CLIMAParameters

    return 2 * _surface_tension * _molmass_water / _ρ_cloud_liq / _gas_const / T
end

"""    
    functionality: calculates the mean hygroscopicity for all the modes
    parameters: parameter set, aerosol model
    returns: tuple of the mean hygroscopicities for each mode
"""
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

"""    
    functionality: calculates alpha (size-invariant coefficient)
    parameters: parameter set, temperature, aerosol particle mass
    returns: scalar size-invariant coefficient
"""
function α(param_set::EPS, T::FT, aerosol_mass::FT) where {FT <: Real}

    _molmass_water::FT = molmass_water(param_set)
    _grav::FT = grav(param_set)
    _gas_constant::FT = gas_constant()
    _cp_d::FT = cp_d(param_set)
    _molmass_dryair::FT = molmass_dryair(param_set)
    L::FT = latent_heat_vapor(param_set, T)

    return _grav * _molmass_water * L / (_cp_d * _gas_constant * T^2) -
           _grav * _molmass_dryair / (_gas_constant * T)
end

"""    
    functionality: calculates gamma (size-invariant coefficient)
    parameters: parameter set, temperature, aerosol particle mass, and pressure
    returns: scalar coefficeint of curvature
"""
function γ(
    param_set::EPS,
    T::FT,
    aerosol_mass::FT,
    press::FT,
) where {FT <: Real}

    _molmass_water::FT = molmass_water(param_set)
    _gas_constant::FT = gas_constant()
    _molmass_dryair::FT = molmass_dryair(param_set)
    _cp_d::FT = cp_d(param_set)

    L::FT = latent_heat_vapor(param_set, T)
    p_vs::FT = saturation_vapor_pressure(param_set, T, Liquid())
    println("pressure2: ", press)
    return _gas_constant * T / (p_vs * _molmass_water) +
           _molmass_water * L^2  / (_cp_d * press * _molmass_dryair * T)
end
"""    
    functionality: calculates zeta
    parameters: parameter set, temperature, aerosol particle mass, 
                updraft velocity, and G_diff
    returns: scalar zeta value
"""
function ζ(
    param_set::EPS,
    T::FT,
    aerosol_mass::FT,
    updraft_velocity::FT,
    G_diff::FT,
) where {FT <: Real}
    α_var = α(param_set, T, aerosol_mass)
    return 2 * tp_coeff_of_curve(param_set, T) / 3 *
           sqrt(α_var * updraft_velocity / G_diff)
end

"""    
    functionality: calculates eta
    parameters: parameter set, temerpature, aerosol particle mass, 
                aerosol particle number concentration, G_diff, 
                updraft velocity, and pressure
    returns: scalar eta value
"""
function η(
    param_set::EPS,
    temp::Float64,
    aerosol_mass::Float64,
    number_concentration::Float64,
    G_diff::Float64,
    updraft_velocity::Float64,
    press::Float64,
)

    _ρ_cloud_liq = ρ_cloud_liq(param_set)
    α_var = α(param_set, temp, aerosol_mass)
    γ_var = γ(param_set, temp, aerosol_mass, press)
    # println("this is α")
    # println(α_var)
    println("this is γ test: ", γ_var)
    return (α_var * updraft_velocity /
           G_diff)^(3 / 2) / (
        2 *
        pi *
        _ρ_cloud_liq * γ_var
         *
        number_concentration
    )
end

"""    
    functionality: calculates the maximum super saturation for each mode
    parameters: parameter set, aerosol model, temperature, pressure, and
                updraft velocity
    returns: a tuple with the max supersaturations for each mode
"""
function tp_max_super_sat(
    param_set::EPS,
    am::aerosol_model,
    temp::FT,
    press::FT,
    updraft_velocity::FT) where {FT <: Real}

    _grav::FT = grav(param_set)
    _molmass_water::FT = molmass_water(param_set)
    _molmass_dryair::FT = molmass_dryair(param_set)
    _gas_constant::FT = gas_constant()
    _cp_d::FT = cp_d(param_set)
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)

    L::FT = latent_heat_vapor(param_set, T)
    p_vs::FT = saturation_vapor_pressure(param_set, T, Liquid())
    G_diff::FT = G_func(param_set, T, Liquid())

    mean_hygro = tp_mean_hygroscopicity(param_set, am)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        a = sum(num_of_comp) do j
            f = 0.5 * exp(2.5 * (log(mode_i.stdev))^2)
            g = 1 + 0.25 * log(mode_i.stdev)
            coeff_of_curve = tp_coeff_of_curve(param_set, temp)
            surface_tension_effects = ζ(param_set,
                                           temp,
                                           mode_i.molar_mass[j],
                                           updraft_velocity,
                                           G_diff)
                                
            println("this is ζ test: ", surface_tension_effects)
            critsat =
                2 / sqrt(mean_hygro[i]) *
                (coeff_of_curve / (3 * mode_i.r_dry))^(3 / 2)
            η_value = η(param_set, temp, mode_i.molar_mass[j], mode_i.N, G_diff, updraft_velocity, press)
            println("this is η_value test: ", η_value)
            1 / (critsat^2) * (
                f * (surface_tension_effects / η_value)^(3 / 2) +
                g * (critsat^2 /
                (η_value + 3 * surface_tension_effects))^(3 / 4)
            )

        end
        a^(1 / 2)

    end
end

"""    
    functionality: calculates the critical supersaturation
    parameters: parameter set, aerosol model, and temperature
    returns: a tuple of the critical supersaturations of each mode
"""
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
            2 /
            sqrt(mean_hygro[i]) *
            (
                tp_coeff_of_curve(param_set, temp) / (3 * mode_i.r_dry)
            )^(3 / 2)
        end
        a
    end
end

"""    
    functionality: calculates the total number of particles activated across all 
                   modes and components
    parameters: parameters set, aerosol model, temperature, updraft velocity,
                G_diff, and pressure
    returns: a scalar of the total number of particles activated across all modes 
             and components
"""
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
        tp_max_super_sat(param_set, am, temp, updraft_velocity, press)
    values = ntuple(am.N) do i
        mode_i = am.modes[i]

        sigma = mode_i.stdev
        u_top = 2 * log(critical_supersaturation[i] / max_supersat[i])
        u_bottom = 3 * sqrt(2) * log(sigma)
        u = u_top / u_bottom
        mode_i.N *
        1 / 2 * (1 - erf(u))
    end
    summation = 0.0
    for i in range(1, length = length(values))
        summation += values[i]
    end
    return summation
end


# TESTS

@testset "mean_hygroscopicity" begin

    println("----------")
    println("mean_hygroscopicity: ")
    println(tp_mean_hygroscopicity(param_set, AM_1))
    println(mean_hygroscopicity(param_set, AM_1))

    for AM in AM_test_cases
        @test all(
            tp_mean_hygroscopicity(param_set, AM) .≈
            mean_hygroscopicity(param_set, AM)
        )
    end
    println(" ")
end

@testset "max_supersaturation" begin

    println("----------")
    println("max_supersaturation: ")
    println(tp_max_super_sat(param_set, AM_1,  T, p, w))
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

# @testset "total_n_act" begin

#     println("----------")
#     println("total_N_act: ")
#     println(tp_total_n_act(param_set, AM_1, 2.0, 3.0, 4.0, 1.0))
#     println(total_N_activated(param_set, AM_1, T, p, w))

#     # TODO
#     #for AM in AM_test_cases
#     #    @test all(
#     #        tp_total_n_act(param_set, AM, 2.0, 3.0, 4.0, 1.0) .≈
#     #        total_N_activated(param_set, AM, T, p, w)
#     #    )
#     #end

#     println(" ")
# end


# println(tp_max_super_sat_prac(param_set, AM_5, 2.0, 3.0, 4.0, 1.0,))
# println(max_supersaturation(param_set, AM_1, T, p, w))



@testset "Zero Verification" begin
    println(total_N_activated(param_set, AM_6, T, p, w))
    @test(total_N_activated(param_set, AM_6, T, p, w)≈0.0)
    @test(total_N_activated(param_set, AM_1, T, p, 0.0000000000000001)≈0.0)
end