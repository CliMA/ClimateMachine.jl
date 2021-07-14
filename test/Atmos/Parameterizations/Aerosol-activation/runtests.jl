using SpecialFunctions
using Test
# using ClimateMachine.AerosolModel: mode, aerosol_model
# using ClimateMachine.AerosolActivationV2
include("/home/idularaz/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/AerosolActivationV2/AerosolActivationV2.jl")
using CLIMAParameters
using CLIMAParameters: gas_constant
using CLIMAParameters.Planet: molmass_water, ρ_cloud_liq, grav, LH_v0, cp_v
using CLIMAParameters.Atmos.Microphysics: K_therm, D_vapor

struct EarthParameterSet <: AbstractEarthParameterSet end
const EPS = EarthParameterSet
const param_set = EarthParameterSet()
# individual aerosol mode struct
struct mode{T}
    particle_density::T
    osmotic_coeff::T
    molar_mass::T
    dissoc::T
    mass_frac::T
    mass_mix_ratio::T
    dry_radius::T
    radius_stdev::T
    aerosol_density::T
    n_components::Int64
end

# complete aerosol model struct
struct aerosol_model{T}
    modes::T
    N::Int
    function aerosol_model(modes::T) where {T}
        return new{T}(modes, length(modes)) #modes new{T}
    end
end

SURFACE_TENSION = 0.0757   # TODO
P_SAT = 1.0 # need to fix  # TODO
#TODO LATENT_HEAT = 1000.0  is it this one? _LH_v0 = LH_v0(param_set)
#TODO SPECIFIC_HEAT = 1 is it this one?: _cp_v = cp_v(param_set)


# Building the test structures
# 1. Set Aerosol parameters:

# Sea Salt--universal parameters
osmotic_coeff_seasalt = 0.9
molar_mass_seasalt = 0.058443
rho_seasalt = 2170.0
dissoc_seasalt = 2.0
mass_frac_seasalt = 1.0
mass_mix_ratio_seasalt = 1.0

# Sea Salt -- Accumulation mode
dry_radius_seasalt_accum = 0.000000243
radius_stdev_seasalt_accum = 0.0000014
particle_density_seasalt_accum = 100.0

# Sea Salt -- Coarse Mode
dry_radius_seasalt_coarse = 0.0000015
radius_stdev_seasalt_coarse = 0.0000021
particle_density_seasalt_coarse = 100.0

# TODO: Dust parameters (just copy and pasted seasalt values rn)
# Dust--universal parameters
osmotic_coeff_dust = 0.9
molar_mass_dust = 0.058443
rho_dust = 2170.0
dissoc_dust = 2.0
mass_frac_dust = 1.0
mass_mix_ratio_dust = 1.0

# Dust -- Accumulation mode
dry_radius_dust_accum = 0.000000243
radius_stdev_dust_accum = 0.0000014
particle_density_dust_accum = 100.0

# Dust -- Coarse Mode
dry_radius_dust_coarse = 0.0000015
radius_stdev_dust_coarse = 0.0000021
particle_density_dust_coarse = 100.0

# 3. Populate structs to pass into functions/run calculations
# Test cases 1-3 (Just Sea Salt)
accum_mode_seasalt = mode((particle_density_seasalt_accum,),
                          (osmotic_coeff_seasalt,),
                          (molar_mass_seasalt,),
                          (dissoc_seasalt,),
                          (mass_frac_seasalt,),
                          (mass_mix_ratio_seasalt,),
                          (dry_radius_seasalt_accum,),
                          (radius_stdev_seasalt_accum,),
                          (rho_seasalt,),
                          1)

coarse_mode_seasalt = mode((particle_density_seasalt_coarse,),
                           (osmotic_coeff_seasalt,),
                           (molar_mass_seasalt,),
                           (dissoc_seasalt,),
                           (mass_frac_seasalt,),
                           (mass_mix_ratio_seasalt,),
                           (dry_radius_seasalt_coarse,),
                           (radius_stdev_seasalt_coarse,),
                           (rho_seasalt,),
                           1)

aerosolmodel_testcase1 = aerosol_model((accum_mode_seasalt,))
aerosolmodel_testcase2 = aerosol_model((coarse_mode_seasalt,))
aerosolmodel_testcase3 = aerosol_model((accum_mode_seasalt, coarse_mode_seasalt))

# Test cases 4-5 (Sea Salt and Dust)
accum_mode_seasalt_dust = mode((particle_density_seasalt_accum,
                                particle_density_dust_accum),
                               (osmotic_coeff_seasalt,
                                osmotic_coeff_dust),
                               (molar_mass_seasalt,
                                molar_mass_dust),
                               (dissoc_seasalt,
                                dissoc_dust),
                               (mass_frac_seasalt,
                                mass_frac_dust),
                               (mass_mix_ratio_seasalt,
                                mass_mix_ratio_dust),
                               (dry_radius_seasalt_accum,
                                dry_radius_dust_accum),
                               (radius_stdev_seasalt_accum,
                                radius_stdev_dust_accum),
                               (rho_seasalt,
                                rho_dust),
                                2)

coarse_mode_seasalt_dust = mode((particle_density_seasalt_coarse,
                                 particle_density_dust_coarse),
                                (osmotic_coeff_seasalt,
                                 osmotic_coeff_dust),
                                (molar_mass_seasalt,
                                 molar_mass_dust),
                                (dissoc_seasalt,
                                 dissoc_dust),
                                (mass_frac_seasalt,
                                 mass_frac_dust),
                                (mass_mix_ratio_seasalt,
                                 mass_mix_ratio_dust),
                                (dry_radius_seasalt_coarse,
                                 dry_radius_dust_coarse),
                                (radius_stdev_seasalt_coarse,
                                 radius_stdev_dust_coarse),
                                (rho_seasalt,
                                 rho_dust),
                                 2)

aerosolmodel_testcase4 = aerosol_model((accum_mode_seasalt_dust,))
aerosolmodel_testcase5 = aerosol_model((accum_mode_seasalt_dust,
                                        coarse_mode_seasalt_dust))

function tp_coeff_of_curve(param_set::EPS, T::FT) where{FT <: Real}

    _gas_constant::FT = gas_constant()
    _molmass_water::FT = molmass_water(param_set)
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)

    value = 2 * SURFACE_TENSION * _molmass_water / (_ρ_cloud_liq * _gas_constant * T)
    return value
end

function total_mass(m::mode)
    num_of_comp = m.n_components
    total_mass = sum(num_of_comp) do j
            m.particle_density[j]
    end
    return total_mass
end

function tp_mean_hygroscopicity(param_set::EPS, am::aerosol_model)

    _molmass_water = molmass_water(param_set)
    _ρ_cloud_liq = ρ_cloud_liq(param_set)

    return ntuple(am.N) do i
        mode_i = am.modes[i]
        total_mass_value = total_mass(mode_i)
        num_of_comp = mode_i.n_components # mode_i.n_components
        numerator = sum(num_of_comp) do j
            mode_i.osmotic_coeff[j] * mode_i.mass_mix_ratio[j] * mode_i.dissoc[j] * mode_i.mass_frac[j] * 1/mode_i.molar_mass[j] # mode_i.particle_density[j]/total_mass_value *
        end
        denominator = sum(num_of_comp) do j
            mode_i.particle_density[j]/total_mass_value * mode_i.mass_mix_ratio[j] / mode_i.aerosol_density[j]
        end
        (numerator/denominator) * (_molmass_water / _ρ_cloud_liq)
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
    _LH_v0::FT = LH_v0(param_set)
    _cp_v::FT = cp_v(param_set)

    return _grav * _molmass_water * _LH_v0 / (_cp_v * _gas_constant * T^2) - _grav * aerosol_mass/(_gas_constant * T)
end

function gamma(param_set::EPS, T::FT, aerosol_mass::FT, press::FT) where {FT <: Real}

    _molmass_water::FT = molmass_water(param_set)
    _gas_constant::FT = gas_constant()
    _LH_v0::FT = LH_v0(param_set)
    _cp_v::FT = cp_v(param_set)

    return _gas_constant * T / (P_SAT * _molmass_water) + _molmass_water * _LH_v0 ^ 2 / (_cp_v * press * aerosol_mass * T)
end

function zeta(param_set::EPS, T::FT, aerosol_mass::FT, updraft_velocity::FT, G_diff::FT) where {FT <: Real}
    return 2 * tp_coeff_of_curve(param_set, T) / 3 * (alpha(param_set, T, aerosol_mass) * updraft_velocity / G_diff)^(1/2)
end

function eta(param_set::EPS,
             temp::Float64,
             aerosol_mass::Float64,
             particle_density::Float64,
             G_diff::Float64,
             updraft_velocity::Float64,
             press::Float64)

    _ρ_cloud_liq = ρ_cloud_liq(param_set)

    return alpha(param_set, temp, aerosol_mass) * updraft_velocity / G_diff^(3/2) /
           (2 * pi * _ρ_cloud_liq * gamma(param_set, temp, aerosol_mass, press) * particle_density)
end

function tp_max_super_sat(param_set::EPS,
                          am::aerosol_model,
                          temp::Float64,
                          updraft_velocity::Float64,
                          press::Float64)
    _R_v = R_v(param_set)
    _LH_v0 = LH_v0(param_set)
    _K_therm = K_therm(param_set)
    _D_vapor = D_vapor(param_set)
    mean_hygro = tp_mean_hygroscopicity(param_set, am)
    G_diff = ((_LH_v0/(_K_therm*TEMP))*(((_LH_v0/TEMP/_R_v)-1))+((_R_v*TEMP)/(P_SAT*_D_vapor)))^(-1)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        total_mass_value = total_mass(mode_i)
        num_of_comp = mode_i.n_components
        a = sum(num_of_comp) do j
            f = 0.5 * exp(2.5 * log(mode_i.radius_stdev[j])^2)
            g = 1 + 0.25 * log(mode_i.radius_stdev[j])
            coeff_of_curve = tp_coeff_of_curve(param_set, temp)
            surface_tension_effects = zeta(param_set, temp, mode_i.molar_mass[j], updraft_velocity, G_diff)
            critsat = 2/sqrt(mean_hygro[i]) * (coeff_of_curve / (3 * mode_i.dry_radius[j])) ^ (3/2) # FILL
            eta_value = eta(param_set, temp, mode_i.molar_mass[j], mode_i.particle_density[j], G_diff, updraft_velocity, press)
            mode_i.particle_density[j]/total_mass_value * (1/(critsat ^ 2) * (f * (surface_tension_effects/eta_value) ^(3/2) + g * (critsat ^ 2)/ (eta_value + 3 * surface_tension_effects)^(3/4)))
        end
        a ^ (1/2)
    end
end

function tp_critical_supersaturation(param_set::EPS,
                                     am::aerosol_model,
                                     temp::Float64)
    mean_hygro = tp_mean_hygroscopicity(param_set, am)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        total_mass_value = total_mass(mode_i)
        a = sum(num_of_comp) do j
            mode_i.particle_density[j]/total_mass_value * 2 / sqrt(mean_hygro[i]) * (tp_coeff_of_curve(param_set, temp) / (3 * mode_i.dry_radius[j])) ^ (3/2)
        end
        a
    end

end

function tp_total_n_act(param_set::EPS,
                        am::aerosol_model,
                        temp::Float64,
                        updraft_velocity::Float64,
                        G_diff::Float64,
                        press::Float64)
    critical_supersaturation = tp_critical_supersaturation(param_set, am, temp)
    max_supersat = tp_max_super_sat(param_set, am, temp, updraft_velocity, G_diff, press)
    values = ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        total_mass_value = total_mass(mode_i)
        a = sum(num_of_comp) do j
            sigma = mode_i.radius_stdev[j]
            u_top = 2 * log(critical_supersaturation[i] / max_supersat[i])
            u_bottom = 3 * sqrt(2) * log(sigma)
            u = u_top / u_bottom
            mode_i.particle_density[j]/total_mass_value * mode_i.particle_density[j] * 1/2 * (1 - erf(u))
        end
    end
    summation = 0.0
    for i in range(1, length=length(values))
        summation += values[i]
    end
    return summation
end
# test constants:
TEMP = 273.15
UPDFT_VELO = 5.0
PRESS = 100000.0

@testset "mean_hygroscopicity" begin
    @test all(tp_mean_hygroscopicity(param_set, aerosolmodel_testcase1) .≈ mean_hygroscopicity(param_set, aerosolmodel_testcase1))
    @test all(tp_mean_hygroscopicity(param_set, aerosolmodel_testcase2) .≈ mean_hygroscopicity(param_set, aerosolmodel_testcase2))
    @test all(tp_mean_hygroscopicity(param_set, aerosolmodel_testcase3) .≈ mean_hygroscopicity(param_set, aerosolmodel_testcase3))
    @test all(tp_mean_hygroscopicity(param_set, aerosolmodel_testcase4) .≈ mean_hygroscopicity(param_set, aerosolmodel_testcase4))
    @test all(tp_mean_hygroscopicity(param_set, aerosolmodel_testcase5) .≈ mean_hygroscopicity(param_set, aerosolmodel_testcase5))
end

@testset "max_supersaturation" begin
    # println(tp_max_super_sat(param_set, aerosolmodel_testcase1, 2.0, UPDFT_VELO, PRESS))
    # println(max_supersaturation(param_set, aerosolmodel_testcase1, P_SAT))
    @test all(tp_max_super_sat(param_set, aerosolmodel_testcase1, TEMP, UPDFT_VELO, PRESS) .≈ max_supersaturation(param_set, aerosolmodel_testcase1, TEMP, UPDFT_VELO, PRESS, P_SAT))
    @test all(tp_max_super_sat(param_set, aerosolmodel_testcase2, TEMP, UPDFT_VELO, PRESS) .≈ max_supersaturation(param_set, aerosolmodel_testcase2, TEMP, UPDFT_VELO, PRESS, P_SAT))
    @test all(tp_max_super_sat(param_set, aerosolmodel_testcase3, TEMP, UPDFT_VELO, PRESS) .≈ max_supersaturation(param_set, aerosolmodel_testcase3, TEMP, UPDFT_VELO, PRESS, P_SAT))
    @test all(tp_max_super_sat(param_set, aerosolmodel_testcase4, TEMP, UPDFT_VELO, PRESS) .≈ max_supersaturation(param_set, aerosolmodel_testcase4, TEMP, UPDFT_VELO, PRESS, P_SAT))
    @test all(tp_max_super_sat(param_set, aerosolmodel_testcase5, TEMP, UPDFT_VELO, PRESS) .≈ max_supersaturation(param_set, aerosolmodel_testcase5, TEMP, UPDFT_VELO, PRESS, P_SAT))
end

@testset "total_n_act" begin
    @test all(tp_total_n_act(param_set, aerosolmodel_testcase1, TEMP, UPDFT_VELO, PRESS) .≈ total_N_activated(param_set, aerosolmodel_testcase1, TEMP, UPDFT_VELO, PRESS, P_SAT))
    @test all(tp_total_n_act(param_set, aerosolmodel_testcase1, TEMP, UPDFT_VELO, PRESS) .≈ total_N_activated(param_set, aerosolmodel_testcase2, TEMP, UPDFT_VELO, PRESS, P_SAT))
    @test all(tp_total_n_act(param_set, aerosolmodel_testcase1, TEMP, UPDFT_VELO, PRESS) .≈ total_N_activated(param_set, aerosolmodel_testcase3, TEMP, UPDFT_VELO, PRESS, P_SAT))
    @test all(tp_total_n_act(param_set, aerosolmodel_testcase1, TEMP, UPDFT_VELO, PRESS) .≈ total_N_activated(param_set, aerosolmodel_testcase4, TEMP, UPDFT_VELO, PRESS, P_SAT))
    @test all(tp_total_n_act(param_set, aerosolmodel_testcase1, TEMP, UPDFT_VELO, PRESS) .≈ total_N_activated(param_set, aerosolmodel_testcase5, TEMP, UPDFT_VELO, PRESS, P_SAT))
end