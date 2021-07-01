
Pkg.add("SpecialFunctions")
# TODO: Map correct locations to run the test 
using Test
using ClimateMachine.Microphysics_0M
using ClimateMachine.Microphysics
using Thermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, R_d, molmass_ratio
using CLIMAParameters.Atmos.Microphysics
using CLIMAParameters.Atmos.Microphysics_0M

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
# Universal parameters:

# Building the test structures
# 1. Set Aerosol parameters: 

# Sea Salt--universal parameters
osmotic_coeff_seasalt = 0.9 # osmotic coefficient
molar_mass_seasalt = 0.058443 # sea salt molar mass; kg/mol
rho_seasalt = 2170 # sea salt density; kg/m^3
dissoc_seasalt = 2 # Sea salt dissociation                         
mass_frac_seasalt = 1 # mass fraction                              TODO
mass_mix_ratio_seasalt = 1 # mass mixing rati0                     TODO

# Sea Salt -- Accumulation mode
radius_seasalt_accum = 0.000000243 # mean particle radius (m)
radius_stdev_seasalt_accum = 0.0000014 # mean particle stdev (m)
particle_density_seasalt_accum = 100000000 # particle density (1/m^3)

# Sea Salt -- Coarse Mode
radius_seasalt_coarse = 0.0000015 # mean particle radius (m)
radius_stdev_seasalt_accum = 0.0000021 # mean particle stdev(m)

# TODO: Dust parameters (just copy and pasted seasalt values rn)
# Dust--universal parameters
osmotic_coeff_dust = 0.9 # osmotic coefficient
molar_mass_dust = 0.058443 # sea salt molar mass; kg/mol
rho_dust = 2170 # sea salt density; kg/m^3
dissoc_dust = 2 # Sea salt dissociation                         
mass_frac_dust = 1 # mass fraction                              TODO
mass_mix_ratio_dust = 1 # mass mixing rati0                     TODO

# Dust -- Accumulation mode
radius_dust_accum = 0.000000243 # mean particle radius (m)
radius_stdev_dust_accum = 0.0000014 # mean particle stdev (m)

# Dust -- Coarse Mode
radius_dust_coarse = 0.0000015 # mean particle radius (m)
radius_stdev_dust_accum = 0.0000021 # mean particle stdev(m)

# Abdul-Razzak and Ghan 

# 2. Create structs that parameters can be pass through
# individual aerosol mode struct
struct mode{T}
    osmotic_coeff::T
    molar_mass::T 
    dissoc::T 
    mass_frac::T 
    mass_mix_ratio::T 
    radius::T 
    radius_stdev::T
end

# complete aerosol model struct
struct aerosol_model{T}
    modes::T
    N::T 
    function aerosol_model(modes::T) where {T}
        return new{T}(modes, length(modes))
    end
end 

# 3. Populate structs to pass into functions/run calculations
# Test cases 1-3 (Just Sea Salt)
accum_mode_seasalt = mode(osmotic_coeff_seasalt, molar_mass_seasalt, 
                  dissoc_seasalt, mass_frac_seasalt, radius_seasalt_accum,
                  radius_stdev_seasalt_accum)

coarse_mode_seasalt = mode(osmotic_coeff_seasalt, molar_mass_seasalt, 
                  dissoc_seasalt, mass_frac_seasalt, radius_seasalt_coarse,
                  radius_stdev_seasalt_coarse)

aerosolmodel_testcase1 = aerosol_model(accumlation_mode_seasalt)
aerosolmodel_testcase2 = aerosol_model(coarse_mode_seasalt)
aerosolmodel_testcase3 = aerosol_model((accum_mode_seasalt, coarse_mode_seasalt))

# Test cases 4-5 (Sea Salt and Dust)
accum_mode_seasalt_dust = mode((osmotic_coeff_seasalt, osmotic_coeff_dust), 
                               (molar_mass_seasalt, molar_mass_dust),
                               (dissoc_seasalt, dissoc_dust),
                               (mass_frac_seasalt, mass_frac_dust),
                               (radius_seasalt_accum, radius_dust_accum),
                               (radius_stdev_seasalt_accum, radius_stdev_dust_accum))

coarse_mode_seasalt_dust = ((osmotic_coeff_seasalt, osmotic_coeff_dust), 
                            (molar_mass_seasalt, molar_mass_dust),
                            (dissoc_seasalt, dissoc_dust),
                            (mass_frac_seasalt, mass_frac_dust),
                            (radius_seasalt_coarse, radius_dust_coarse),
                            (radius_stdev_seasalt_coarse, radius_stdev_dust_coarse))

aerosolmodel_testcase4 = aerosol_model(accum_mode_seasalt_dust)
aerosolmodel_testcase5 = aerosol_model((accum_mode_seasalt_dust,
                                        coarse_mode_seasalt_dust))
