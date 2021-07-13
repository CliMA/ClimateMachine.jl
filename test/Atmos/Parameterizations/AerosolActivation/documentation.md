
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


# 1. Set Aerosol parameters: 

# Sea Salt--universal parameters
osmotic_coeff_seasalt = osmotic coefficient
molar_mass_seasalt = sea salt molar mass; kg/mol
rho_seasalt = sea salt density; kg/m^3
dissoc_seasalt = sea salt dissociation                         
mass_frac_seasalt = mass fraction                              TODO
mass_mix_ratio_seasalt = mass mixing rati0                     TODO

# Sea Salt -- Accumulation mode
dry_radius_seasalt_accum = mean particle radius (m)
radius_stdev_seasalt_accum = mean particle stdev (m)
particle_density_seasalt_accum = particle density (1/m^3)

# Sea Salt -- Coarse Mode
dry_radius_seasalt_coarse = mean particle radius (m)
radius_stdev_seasalt_coarse = mean particle stdev(m)
particle_density_seasalt_coarse = particle density (1/m^3)

# TODO: Dust parameters (just copy and pasted seasalt values rn)
# Dust--universal parameters
osmotic_coeff_dust = osmotic coefficient
molar_mass_dust = sea salt molar mass; kg/mol
rho_dust = sea salt density; kg/m^3
dissoc_dust = sea salt dissociation                         
mass_frac_dust = mass fraction                              TODO
mass_mix_ratio_dust = mass mixing rati0                     TODO

# Dust -- Accumulation mode
dry_radius_dust_accum = mean particle radius (m)
radius_stdev_dust_accum = mean particle stdev (m)
particle_density_dust_accum = particle density (1/m^3)

# Dust -- Coarse Mode
dry_radius_dust_coarse = mean particle radius (m)
radius_stdev_dust_coarse = mean particle stdev(m)
particle_density_dust_coarse = particle density (1/m^3)

Functions:

# function total_mass(m::mode)
    functionality: calculates the total mass in the mode
    parameters: an aerosol mode
    returns: a scalar of the total mass

# function tp_mean_hygroscopicity(am::aerosol_model)
    functionality: calculates the mean hygroscopicity for all the modes
    parameters: an aerosol model
    returns: tuple of the mean hygroscopicities for each mode

# function tp_max_super_sat(am::aerosol_model, temp::Float64, updraft_velocity::Float64, diffusion::Float64, 
#                           activation_time::Float64)
    functionality: calculates the maximum super saturation for each mode
    parameters: aerosol model, temperature, updraft velocity, diffusion constant,
                and the activation activation time
    returns: a tuple with the max supersaturations for each mode

# function tp_coeff_of_curve(temp::Float64, activation_time::Float64)
    functionality: calculates the coefficient of curvature
    parameters: temperature, and activation time
    returns: scalar coefficeint of curvature

# function tp_critical_supersaturation(am::aerosol_model, temp::Float64, activation_time::Float64)
    functionality: calculates the critical supersaturation 
    parameters: aerosol model
    returns: a tuple of the critical supersaturations of each mode

# function tp_total_N_Act(am::aerosol_model, temp::Float64, updraft_velocity::Float64, 
#                         diffusion::Float64, activation_time::Float64)
    functionality: calculates the total number of particles activated across all 
                   modes and components
    parameters: aerosol model, temperature, updraft velocity, diffusion, and
                activation time
    returns: a scalar of the total number of particles activated across all modes 
             and components