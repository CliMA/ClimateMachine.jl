# Pkg.add("SpecialFunctions")
# TODO: Map correct locations to run the test 
# using Test
# using ClimateMachine.Microphysics_0M
# using ClimateMachine.Microphysics
# using Thermodynamics

# using CLIMAParameters
# using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, R_d, molmass_ratio
# using CLIMAParameters.Atmos.Microphysics
# using CLIMAParameters.Atmos.Microphysics_0M

"""
Isabella Dula and Shevali Kadakia
    
This file has the complete set of tests to verify and validate the parameterization of
the model given in Abdul-Razzak and Ghan (2000). 
The structure of the pipeline is as follows:
--Test classifications:
    --Verfication (VER): ensures that function output has consistent output, 
    no matter inputted values (i.e. verifies that the functions are 
    doing what we want them to)
    --Validation (VAL): checks functions against model data in 
    Abdul-Razzak and Ghan (2000) (i.e., validates the functions outputs 
    against published results)
--Dimension (DIM):
    --Tests are done with multi-dimensional inputs: 
    --0: Only one mode and one component (e.g., coarse sea salt) 
    --1: Considering multiple modes over one component (e.g., a
    ccumulation mode and coarse mode sea salt)
    --2: Considering multiple modes with multiple components (e.g., 
    accumulation and coarse mode for sea salt and dust)
--Modes and Components Considered
        --This testing pipeline uses aerosol data from 
        Porter and Clarke (1997) to provide real-world inputs into the functions
        --Modes: Accumulation (ACC) and coarse (COA)
        --Components: Sea Salt (SS) and Dust (DUS)
"""

# GET FROM CLIMA PARAMATERS
molar_mass_water = 18
density_water = 1000.0
R = 8.314462618 # gas constant ((kg m^2) / (s^2 K mol))

# Universal parameters:

# Building the test structures
# 1. Set Aerosol parameters: 

# Sea Salt--universal parameters
osmotic_coeff_seasalt = 0.9 # osmotic coefficient
molar_mass_seasalt = 0.058443 # sea salt molar mass; kg/mol
rho_seasalt = 2170.0 # sea salt density; kg/m^3
dissoc_seasalt = 2.0 # Sea salt dissociation                         
mass_frac_seasalt = 1.0 # mass fraction                              TODO
mass_mix_ratio_seasalt = 1.0 # mass mixing rati0                     TODO

# Sea Salt -- Accumulation mode
radius_seasalt_accum = 0.000000243 # mean particle radius (m)
radius_stdev_seasalt_accum = 0.0000014 # mean particle stdev (m)
particle_density_seasalt_accum = 100.0 # particle density (1/m^3)

# Sea Salt -- Coarse Mode
radius_seasalt_coarse = 0.0000015 # mean particle radius (m)
radius_stdev_seasalt_coarse = 0.0000021 # mean particle stdev(m)
particle_density_seasalt_coarse = 100.0 # particle density (1/m^3)

# TODO: Dust parameters (just copy and pasted seasalt values rn)
# Dust--universal parameters
osmotic_coeff_dust = 0.9 # osmotic coefficient
molar_mass_dust = 0.058443 # sea salt molar mass; kg/mol
rho_dust = 2170.0 # sea salt density; kg/m^3
dissoc_dust = 2.0 # Sea salt dissociation                         
mass_frac_dust = 1.0 # mass fraction                              TODO
mass_mix_ratio_dust = 1.0 # mass mixing rati0                     TODO

# Dust -- Accumulation mode
radius_dust_accum = 0.000000243 # mean particle radius (m)
radius_stdev_dust_accum = 0.0000014 # mean particle stdev (m)
particle_density_dust_accum = 100.0 # particle density (1/m^3)

# Dust -- Coarse Mode
radius_dust_coarse = 0.0000015 # mean particle radius (m)
radius_stdev_dust_coarse = 0.0000021 # mean particle stdev(m)
particle_density_dust_coarse = 100.0 # particle density (1/m^3)

# Abdul-Razzak and Ghan 

# 2. Create structs that parameters can be pass through
# individual aerosol mode struct
struct mode{T}
    particle_density::T
    osmotic_coeff::T
    molar_mass::T
    dissoc::T
    mass_frac::T
    mass_mix_ratio::T
    radius::T
    radius_stdev::T
    aerosol_density::T
    n_components::T 
end

function create_mode(num_modes::Int64, particle_density::Tuple, osmotic_coeff::Tuple, molar_mass::Tuple, dissoc::Tuple, mass_frac::Tuple, mass_mix_ratio::Tuple, radius::Tuple, radius_stdev::Tuple, aerosol_density::Tuple)
    return ntuple(num_modes) do i
        mode(Tuple(particle_density[i]), 
             Tuple(osmotic_coeff[i]), 
             Tuple(molar_mass[i]), 
             Tuple(dissoc[i]), 
             Tuple(mass_frac[i]), 
             Tuple(mass_mix_ratio[i]), 
             Tuple(radius[i]), 
             Tuple(radius_stdev[i]), 
             Tuple(aerosol_density[i]),
             Tuple(length(particle_density[i]) * 1.0)
             )
    end
end

# complete aerosol model struct
struct aerosol_model{T}
    modes::T
    N::Int 
    function aerosol_model(modes::T) where {T}
        return new{T}(modes, length(modes)) #modes
    end
end 

# 3. Populate structs to pass into functions/run calculations
# Test cases 1-3 (Just Sea Salt)
accum_mode_seasalt = create_mode(1, 
                                (particle_density_seasalt_accum,), 
                                (osmotic_coeff_seasalt,), 
                                (molar_mass_seasalt,), 
                                (dissoc_seasalt,), 
                                (mass_frac_seasalt,), 
                                (mass_mix_ratio_seasalt,), 
                                (radius_seasalt_accum,),
                                (rho_seasalt,),
                                (radius_stdev_seasalt_accum,)
                                )

coarse_mode_seasalt = create_mode(1,
                                 (particle_density_seasalt_coarse,),
                                 (osmotic_coeff_seasalt,), 
                                 (molar_mass_seasalt,), 
                                 (dissoc_seasalt,), 
                                 (mass_frac_seasalt,), 
                                 (mass_mix_ratio_seasalt,),
                                 (radius_seasalt_coarse,),
                                 (rho_seasalt,),
                                 (radius_stdev_seasalt_coarse,),
                                 )

aerosolmodel_testcase1 = aerosol_model(accum_mode_seasalt)
aerosolmodel_testcase2 = aerosol_model(coarse_mode_seasalt)
aerosolmodel_testcase3 = aerosol_model((accum_mode_seasalt, coarse_mode_seasalt))

# Test cases 4-5 (Sea Salt and Dust)
accum_mode_seasalt_dust = create_mode(2,
                                     (particle_density_seasalt_accum, particle_density_dust_accum),
                                     (osmotic_coeff_seasalt, osmotic_coeff_dust), 
                                     (molar_mass_seasalt, molar_mass_dust),
                                     (dissoc_seasalt, dissoc_dust),
                                     (mass_frac_seasalt, mass_frac_dust),
                                     (mass_mix_ratio_seasalt, mass_mix_ratio_dust),
                                     (radius_seasalt_accum, radius_dust_accum),
                                     (rho_seasalt, rho_dust),
                                     (radius_stdev_seasalt_accum, radius_stdev_dust_accum))

coarse_mode_seasalt_dust = create_mode(2,
                                      (particle_density_seasalt_coarse, particle_density_dust_coarse),
                                      (osmotic_coeff_seasalt, osmotic_coeff_dust), 
                                      (molar_mass_seasalt, molar_mass_dust),
                                      (dissoc_seasalt, dissoc_dust),
                                      (mass_frac_seasalt, mass_frac_dust),
                                      (mass_mix_ratio_seasalt, mass_mix_ratio_dust),
                                      (radius_seasalt_coarse, radius_dust_coarse),
                                      (rho_seasalt, rho_dust),
                                      (radius_stdev_seasalt_coarse, radius_stdev_dust_coarse))

aerosolmodel_testcase4 = aerosol_model(accum_mode_seasalt_dust)
aerosolmodel_testcase5 = aerosol_model((accum_mode_seasalt_dust,
                                        coarse_mode_seasalt_dust))


function mean_hygroscopicity(am::aerosol_model)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        numerator = sum(num_of_comp) do j
            mode_i.osmotic_coeff[j] * mode_i.mass_mix_ratio[j] * mode_i.dissoc[j] * mode_i.mass_frac[j] * 1/mode_i.molar_mass[j]
        end
        denominator = sum(num_of_comp) do j
            mode_i.mass_mix_ratio[j] / mode_i.aerosol_density[j]
        end
        (numerator/denominator) * (molar_mass_water/density_water)
    end
end

print(mean_hygroscopicity(aerosolmodel_testcase1))
print(mean_hygroscopicity(aerosolmodel_testcase2))

# test 3 doesnt work
# print(mean_hygroscopicity(aerosolmodel_testcase3))

# questions about temp, 
# need to fill equations: , alpha --> 1.0, eta() --> 2.0
# Key:
# surface tension == A
# surface_tension_effects(zeta) --> 3.0

function max_super_sat_test(am::aerosol_model, temp::Float64, updraft_velocity::Float64, diffusion::Float64)
    mean_hygro = mean_hygroscopicity(am)
    summation = ntuple(am.N) do i
        mode_i = am.modes[i]
        f = 0.5 * exp(2.5 * ln(mode_i.radius_stdev)^2)
        g = 1 + 0.25 * ln(mode_i.radius_stdev)
        surface_tension = 2 * activation_time * molar_mass_water / (density_water * R * temp)
        surface_tension_effects = 2 * surface_tension / 3 * (1.0 * updraft_velocity / diffusion)^(1/2)
        supersat = 2/sqrt(mean_hygro[i]) * (surface_tension / (3 * mode_i.radius)) ^ (3/2)
        1/(supersat ^ 2) * (f * (surface_tension_effects/2.0) ^(3/2) + g * (supersat ^ 2)/ (2.0 + 3*surface_tension_effects)^(3/4))
    end
    # return (summation ^ 1/2)
end

print(max_super_sat_test(aerosolmodel_testcase1, 2.0, 3.0)
# function total_N_Act_test(am::aerosol_model)


# function surface_tension_test()

# function eta_test()