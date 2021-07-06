
using SpecialFunctions

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
R = 8.314462618

# Universal parameters:

# Building the test structures
# 1. Set Aerosol parameters: 

# Sea Salt--universal parameters
osmotic_coeff_seasalt = 0.9
molar_mass_seasalt = 0.058443
rho_seasalt = 2170.0
dissoc_seasalt = 2.0                        
mass_frac_seasalt = 1.0                           TODO
mass_mix_ratio_seasalt = 1.0                      TODO

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
mass_frac_dust = 1.0                            TODO
mass_mix_ratio_dust = 1.0                       TODO

# Dust -- Accumulation mode
dry_radius_dust_accum = 0.000000243
radius_stdev_dust_accum = 0.0000014
particle_density_dust_accum = 100.0

# Dust -- Coarse Mode
dry_radius_dust_coarse = 0.0000015
radius_stdev_dust_coarse = 0.0000021
particle_density_dust_coarse = 100.0

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
    dry_radius::T
    radius_stdev::T
    aerosol_density::T
    n_components::T 
end

"""
create_mode: creates modes of a certain component
parameters: number of modes to create for component, particle density, 
            osmotic coefficeint, molar mass, mass mixing ratio, dry radius,
            radius standard deviation, and aerosol density
returns: a tuple with all the modes
"""

function create_mode(num_modes::Int64, 
                     particle_density::Tuple, 
                     osmotic_coeff::Tuple, 
                     molar_mass::Tuple, 
                     dissoc::Tuple, 
                     mass_frac::Tuple, 
                     mass_mix_ratio::Tuple, 
                     dry_radius::Tuple, 
                     radius_stdev::Tuple, 
                     aerosol_density::Tuple)
    return ntuple(num_modes) do i
        mode((particle_density[i],), 
             (osmotic_coeff[i],), 
             (molar_mass[i],), 
             (dissoc[i],), 
             (mass_frac[i],), 
             (mass_mix_ratio[i],), 
             (dry_radius[i],), 
             (radius_stdev[i],), 
             (aerosol_density[i],),
             (length(particle_density[i]) * 1.0,))
    end
end

# complete aerosol model struct
struct aerosol_model{T}
    modes::T
    N::Int 
    function aerosol_model(modes::T) where {T}
        return new{T}(modes, length(modes)) #modes new{T}
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
                                (dry_radius_seasalt_accum,),
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
                                 (dry_radius_seasalt_coarse,),
                                 (rho_seasalt,),
                                 (radius_stdev_seasalt_coarse,)
                                 )

aerosolmodel_testcase1 = aerosol_model((accum_mode_seasalt,))
aerosolmodel_testcase2 = aerosol_model((coarse_mode_seasalt,))
aerosolmodel_testcase3 = aerosol_model((accum_mode_seasalt, coarse_mode_seasalt))

# Test cases 4-5 (Sea Salt and Dust)
accum_mode_seasalt_dust = create_mode(2,
                                     (particle_density_seasalt_accum, 
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
                                     (rho_seasalt, 
                                      rho_dust),
                                     (radius_stdev_seasalt_accum, 
                                      radius_stdev_dust_accum))

coarse_mode_seasalt_dust = create_mode(2,
                                      (particle_density_seasalt_coarse, 
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
                                      (rho_seasalt, 
                                       rho_dust),
                                      (radius_stdev_seasalt_coarse, 
                                      radius_stdev_dust_coarse))

aerosolmodel_testcase4 = aerosol_model((accum_mode_seasalt_dust,))
aerosolmodel_testcase5 = aerosol_model((accum_mode_seasalt_dust,
                                        coarse_mode_seasalt_dust))

"""
mean_hygroscopicity: calculates the mean hygroscopicity for all the modes
parameters: an aerosol model
returns: tuple of the mean hygroscopicities for each mode
"""

function mean_hygroscopicity(am::aerosol_model)
    return ntuple(am.N) do i
        mode_i = am.modes[i][1]
        num_of_comp = mode_i.n_components # mode_i.n_components
        numerator = sum(num_of_comp) do j
            mode_i.osmotic_coeff[j] * mode_i.mass_mix_ratio[j] * mode_i.dissoc[j] * mode_i.mass_frac[j] * 1/mode_i.molar_mass[j]
        end
        denominator = sum(num_of_comp) do j
            mode_i.mass_mix_ratio[j] / mode_i.aerosol_density[j]
        end
        (numerator/denominator) * (molar_mass_water/density_water)
    end
end

# questions about temp, 
# need to fill equations: , alpha --> 1.0, eta() --> 2.0
# Key:
# surface tension == A
# surface_tension_effects(zeta) --> 3.0

"""
max_super_sat_test: calculates the maximum super saturation for each mode
parameters: aerosol model, temperature, updraft velocity, diffusion constant,
            and the activation activation time
returns: a tuple with the max supersaturations for each mode
"""

function max_super_sat_test(am::aerosol_model, 
                            temp::Float64, 
                            updraft_velocity::Float64, 
                            diffusion::Float64, 
                            activation_time::Float64)
    mean_hygro = mean_hygroscopicity(am)
    return ntuple(am.N) do i
        mode_i = am.modes[i][1]
        f = 0.5 * exp(2.5 * log(mode_i.radius_stdev[1])^2)
        g = 1 + 0.25 * log(mode_i.radius_stdev[1])
        surface_tension = 2 * activation_time * molar_mass_water / (density_water * R * temp)
        surface_tension_effects = 2 * surface_tension / 3 * (1.0 * updraft_velocity / diffusion)^(1/2)
        supersat = 2/sqrt(mean_hygro[i]) * (surface_tension / (3 * mode_i.dry_radius[1])) ^ (3/2)
        a = 1/(supersat ^ 2) * (f * (surface_tension_effects/2.0) ^(3/2) + g * (supersat ^ 2)/ (2.0 + 3 * surface_tension_effects)^(3/4))
        a ^ (1/2)
    end
end


"""
coeff_of_curve_test: calculates the coefficient of curvature
parameters: temperature, and activation time
returns: scalar coefficeint of curvature
"""

function coeff_of_curve_test(temp::Float64, activation_time::Float64)
    value = 2 * activation_time * density_water / (density_water * R * temp)
    return value
end

"""
critical_supersaturation_test: calculates the critical supersaturation 
parameters: aerosol model
returns: a tuple of the critical supersaturations of each mode
"""

function critical_supersaturation_test(am::aerosol_model, 
                                       temp::Float64, 
                                       activation_time::Float64)
    mean_hygro = mean_hygroscopicity(am)
    return ntuple(am.N) do i
        mode_i = am.modes[i][1]
        2 / sqrt(mean_hygro[i]) * (coeff_of_curve_test(temp, activation_time) / (3 * mode_i.dry_radius[1]) ^ (3/2))
    end
end

"""
total_N_Act_test: calculates the total number of particles activated across all 
                  modes and components
parameters: aerosol model, temperature, updraft velocity, diffusion, and
            activation time
returns: a scalar of the total number of particles activated across all modes 
         and components
"""

function total_N_Act_test(am::aerosol_model, 
                          temp::Float64, 
                          updraft_velocity::Float64, 
                          diffusion::Float64, 
                          activation_time::Float64)
    critical_supersaturation = critical_supersaturation_test(am::aerosol_model, temp::Float64, activation_time::Float64)
    max_supersat = max_super_sat_test(am, temp, updraft_velocity, diffusion, activation_time)
    values = ntuple(am.N) do i
        mode_i = am.modes[i][1]
        sigma = mode_i.radius_stdev[1]
        u_bottom = 2 * log(critical_supersaturation[i] / max_supersat[i])
        u_top = 3 * sqrt(2) * log(sigma)
        u = u_top / u_bottom
        mode_i.particle_density[1] * 1/2 * (1 - erf(u))
    end
    summation = 0.0
    for i in range(1, length=length(values))
        summation += values[i]
    end
    return summation
end

print(mean_hygroscopicity(aerosolmodel_testcase1))
print(mean_hygroscopicity(aerosolmodel_testcase2))
print(mean_hygroscopicity(aerosolmodel_testcase3))
print(mean_hygroscopicity(aerosolmodel_testcase4))
print(mean_hygroscopicity(aerosolmodel_testcase5))

print(max_super_sat_test(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0))
print(max_super_sat_test(aerosolmodel_testcase2, 2.0, 3.0, 4.0, 1.0))
print(max_super_sat_test(aerosolmodel_testcase3, 2.0, 3.0, 4.0, 1.0))
print(max_super_sat_test(aerosolmodel_testcase4, 2.0, 3.0, 4.0, 1.0))
print(max_super_sat_test(aerosolmodel_testcase5, 2.0, 3.0, 4.0, 1.0))

print(total_N_Act_test(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0))
print(total_N_Act_test(aerosolmodel_testcase2, 2.0, 3.0, 4.0, 1.0))
print(total_N_Act_test(aerosolmodel_testcase3, 2.0, 3.0, 4.0, 1.0))
print(total_N_Act_test(aerosolmodel_testcase4, 2.0, 3.0, 4.0, 1.0))
print(total_N_Act_test(aerosolmodel_testcase5, 2.0, 3.0, 4.0, 1.0))