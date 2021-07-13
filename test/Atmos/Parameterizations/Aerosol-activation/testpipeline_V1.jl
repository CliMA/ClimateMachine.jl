"""
This file tests the functions written for Aerosol Activation. Three funtions are 
tested. 
"""

using SpecialFunctions
using Test

# using ClimateMachine.AerosolModel: mode, aerosol_model
# using ClimateMachine.AerosolActivation

# using CLIMAParameters
# using CLIMAParameters: gas_constant
# using CLIMAParameters.Planet: molmass_water, ρ_cloud_liq, grav, T_freeze
# using CLIMAParameters.Atmos.Microphysics

# struct EarthParameterSet <: AbstractEarthParameterSet end
# const param_set = EarthParameterSet()

include("/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Aerosol-activation/AerosolActivation-Shevali.jl")

# using ClimateMachine.Atmos.Parameterizations.CloudPhysics.Aerosol-activation.AerosolActivation-Shevali.jl: alpha_sic, gamma_sic, coeff_of_curvature, mean_hygroscopicity

# Universal parameters:
MOLAR_MASS_WATER = 18
DENSITY_WATER = 1000.0
R = 8.314462618
SURFACE_TENSION = 0.0757
P_SAT = 1.0 # need to fix
LATENT_HEAT = 1000.0
SPECIFIC_HEAT = 1
GRAVITY = 9.81
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



# 2. Create structs that parameters can be pass through
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

# # complete aerosol model struct
struct aerosol_model{T}
    modes::T
    N::Int 
    function aerosol_model(modes::T) where {T}
        return new{T}(modes, length(modes)) #modes new{T}
    end
end 

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
"""
    functionality: calculates the coefficient of curvature
    parameters: temperature, and activation time
    returns: scalar coefficeint of curvature
"""
function tp_coeff_of_curve(temp::Float64)
    value = 2 * SURFACE_TENSION * MOLAR_MASS_WATER / (DENSITY_WATER * R * temp)
    return value
end

"""
    functionality: calculates the total mass in the mode
    parameters: an aerosol mode
    returns: a scalar of the total mass
"""
function total_mass(m::mode)
    num_of_comp = m.n_components
    total_mass = sum(num_of_comp) do j
            m.particle_density[j]
    end
    return total_mass
end

"""
    functionality: calculates the mean hygroscopicity for all the modes
    parameters: an aerosol model
    returns: tuple of the mean hygroscopicities for each mode
"""

function tp_mean_hygroscopicity(am::aerosol_model)
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
        (numerator/denominator) * (MOLAR_MASS_WATER/DENSITY_WATER)
    end
end

"""
    functionality: calculates the size-invariant coefficeint
    parameters: temperature, the mass of the aerosol material
    returns: alpha (affects the supersaturation)
"""

function alpha(temp::Float64, aerosol_mass::Float64)
    value = GRAVITY * MOLAR_MASS_WATER * LATENT_HEAT / (SPECIFIC_HEAT * R * temp^2) - GRAVITY * aerosol_mass/(R * temp)
    return value 
end

"""
    functionality: calculates the size-invariant coefficeint
    parameters: temperature, the mass of the aerosol material, the pressure
    returns: gamma (affects the supersaturation)
"""

function gamma(temp::Float64, aerosol_mass::Float64, press::Float64)
    value = R * temp / (P_SAT * MOLAR_MASS_WATER) + MOLAR_MASS_WATER * LATENT_HEAT ^ 2/(SPECIFIC_HEAT * press * aerosol_mass * temp)
    return value
end

"""
    functionality: calculates the zeta value
    parameters: temperature, the mass of the aerosol material, 
                the updraft velocity, and the diffusion
    returns: zeta (affects the supersaturation)
"""

function zeta(temp::Float64, aerosol_mass::Float64, updraft_velocity::Float64, G_diff::Float64)
    value = 2 * tp_coeff_of_curve(temp) / 3 * (alpha(temp, aerosol_mass) * updraft_velocity / G_diff)^(1/2)
    return value
end

"""
    functionality: calculates the eta value
    parameters: temperature, the mass of the aerosol material, 
                the particle density, the diffusion constant, 
                the updraft velocity, and the pressure
    returns: eta (affects the supersaturation)
"""

function eta(temp::Float64, 
             aerosol_mass::Float64, 
             particle_density::Float64, 
             G_diff::Float64,
             updraft_velocity::Float64,
             press::Float64)
    value = alpha(temp, aerosol_mass) * updraft_velocity / G_diff^(3/2) / (2 * pi * DENSITY_WATER * gamma(temp, aerosol_mass, press) * particle_density)
    return value
end

"""
    functionality: calculates the maximum super saturation for each mode
    parameters: aerosol model, temperature, updraft velocity, diffusion constant,
                and the activation activation time
    returns: a tuple with the max supersaturations for each mode
"""

function tp_max_super_sat(am::aerosol_model, 
                          temp::Float64, 
                          updraft_velocity::Float64, 
                          G_diff::Float64,
                          press::Float64)
    mean_hygro = tp_mean_hygroscopicity(am)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        total_mass_value = total_mass(mode_i)
        num_of_comp = mode_i.n_components
        a = sum(num_of_comp) do j
            f = 0.5 * exp(2.5 * log(mode_i.radius_stdev[j])^2)
            g = 1 + 0.25 * log(mode_i.radius_stdev[j])
            coeff_of_curve = tp_coeff_of_curve(temp)
            surface_tension_effects = zeta(temp, mode_i.molar_mass[j], updraft_velocity, G_diff)
            critsat = 2/sqrt(mean_hygro[i]) * (coeff_of_curve / (3 * mode_i.dry_radius[j])) ^ (3/2) # FILL 
            eta_value = eta(temp, mode_i.molar_mass[j], mode_i.particle_density[j], G_diff, updraft_velocity, press)
            mode_i.particle_density[j]/total_mass_value * (1/(critsat ^ 2) * (f * (surface_tension_effects/eta_value) ^(3/2) + g * (critsat ^ 2)/ (eta_value + 3 * surface_tension_effects)^(3/4)))
        end
        a ^ (1/2)
    end
end

"""
    functionality: calculates the critical supersaturation 
    parameters: aerosol model
    returns: a tuple of the critical supersaturations of each mode

"""

function tp_critical_supersaturation(am::aerosol_model, 
                                     temp::Float64)
    mean_hygro = tp_mean_hygroscopicity(am)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        total_mass_value = total_mass(mode_i)
        a = sum(num_of_comp) do j
            mode_i.particle_density[j]/total_mass_value * 2 / sqrt(mean_hygro[i]) * (tp_coeff_of_curve(temp) / (3 * mode_i.dry_radius[j])) ^ (3/2)
        end
        a
    end
    
end

"""
    functionality: calculates the total number of particles activated across all 
                   modes and components
    parameters: aerosol model, temperature, updraft velocity, diffusion, and
                activation time
    returns: a scalar of the total number of particles activated across all modes 
             and components
"""

function tp_total_n_act(am::aerosol_model, 
                        temp::Float64, 
                        updraft_velocity::Float64, 
                        G_diff::Float64,
                        press::Float64)
    critical_supersaturation = tp_critical_supersaturation(am, temp)
    max_supersat = tp_max_super_sat(am, temp, updraft_velocity, G_diff, press)
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

# Checks that the tests run

# println("total_n_act")
# println(tp_mean_hygroscopicity(aerosolmodel_testcase1))
# println(tp_mean_hygroscopicity(aerosolmodel_testcase2))
# println(tp_mean_hygroscopicity(aerosolmodel_testcase3))
# println(tp_mean_hygroscopicity(aerosolmodel_testcase4))
# println(tp_mean_hygroscopicity(aerosolmodel_testcase5))

# println("test max super sat")
# println(tp_max_super_sat(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 5.0))
# println(tp_max_super_sat(aerosolmodel_testcase2, 2.0, 3.0, 4.0, 5.0))
# println(tp_max_super_sat(aerosolmodel_testcase3, 2.0, 3.0, 4.0, 5.0))
# println(tp_max_super_sat(aerosolmodel_testcase4, 2.0, 3.0, 4.0, 5.0))
# println(tp_max_super_sat(aerosolmodel_testcase5, 2.0, 3.0, 4.0, 5.0))

#println("test total n activated")
#println(tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 5.0))
#println(tp_total_n_act(aerosolmodel_testcase2, 2.0, 3.0, 4.0, 5.0))
#println(tp_total_n_act(aerosolmodel_testcase3, 2.0, 3.0, 4.0, 5.0))
#println(tp_total_n_act(aerosolmodel_testcase4, 2.0, 3.0, 4.0, 5.0))
#println(tp_total_n_act(aerosolmodel_testcase5, 2.0, 3.0, 4.0, 5.0))

# Running the tests

# @testset "mean_hygroscopicity" begin
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase1) == mean_hygroscopicity(aerosolmodel_testcase1)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase2) == mean_hygroscopicity(aerosolmodel_testcase2)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase3) == mean_hygroscopicity(aerosolmodel_testcase3)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase4) == mean_hygroscopicity(aerosolmodel_testcase4)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase5) == mean_hygroscopicity(aerosolmodel_testcase5)
# end

@testset "max_super_sat" begin
    @test tp_max_super_sat(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) == max_supersaturation(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0)
    @test tp_max_super_sat(aerosolmodel_testcase2, 2.0, 3.0, 4.0, 1.0) == max_supersaturation(aerosolmodel_testcase2, 2.0, 3.0, 4.0, 1.0)
    @test tp_max_super_sat(aerosolmodel_testcase3, 2.0, 3.0, 4.0, 1.0) == max_supersaturation(aerosolmodel_testcase3, 2.0, 3.0, 4.0, 1.0)
    @test tp_max_super_sat(aerosolmodel_testcase4, 2.0, 3.0, 4.0, 1.0) == max_supersaturation(aerosolmodel_testcase4, 2.0, 3.0, 4.0, 1.0)
    @test tp_max_super_sat(aerosolmodel_testcase5, 2.0, 3.0, 4.0, 1.0) == max_supersaturation(aerosolmodel_testcase5, 2.0, 3.0, 4.0, 1.0)
end

# @testset "total_n_act" begin
#     @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) = total_N_activated(aerosolmodel_testcase1)
#     @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) = total_N_activated(aerosolmodel_testcase2)
#     @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) = total_N_activated(aerosolmodel_testcase3)
#     @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) = total_N_activated(aerosolmodel_testcase4)
#     @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) = total_N_activated(aerosolmodel_testcase5)
# end