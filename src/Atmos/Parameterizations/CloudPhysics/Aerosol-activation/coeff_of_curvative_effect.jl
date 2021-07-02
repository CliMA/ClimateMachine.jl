# GET FROM CLIMA PARAMATERS
g = 9.81
Mw = 18.1
L = 10.0
Cp = 1000
T = 273.15
R = 8.1
P=100000
P_saturation = 100000
molar_mass_water = 18.0
density_water = 1000.0
# Universal parameters:

# Building the test structures
# 1. Set Aerosol parameters: 

# Sea Salt--universal parameters
osmotic_coeff_seasalt = 0.9 # osmotic coefficient
molar_mass_seasalt = 0.058443 # sea salt molar mass; kg/mol
rho_seasalt = 2170.0 # sea salt density; kg/m^3
dissoc_seasalt = 2.0 # Sea salt dissociation                         
mass_frac_seasalt = 1.0 # mass fraction                              TODO
mass_mix_ratio_seasalt = 1.0 # mass mixing rati0                    TODO
activation_time_seasalt = 1.0 

# Sea Salt -- Accumulation mode
radius_seasalt_accum = 0.000000243 # mean particle radius (m)
radius_stdev_seasalt_accum = 0.0000014 # mean particle stdev (m)
particle_density_seasalt_accum = 100.0 #000000 # particle density (1/m^3)

# Sea Salt -- Coarse Mode
radius_seasalt_coarse = 0.0000015 # mean particle radius (m)
radius_stdev_seasalt_coarse = 0.0000021 # mean particle stdev(m)

# TODO: Dust parameters (just copy and pasted seasalt values rn)
# Dust--universal parameters
osmotic_coeff_dust = 0.9 # osmotic coefficient
molar_mass_dust = 0.058443*1000 # sea salt molar mass; kg/mol
particle_density_dust_coarse = 1000.0
rho_dust = 2170.0 # sea salt density; kg/m^3
dissoc_dust = 2.0 # Sea salt dissociation                         
mass_frac_dust = 1.0 # mass fraction                              TODO
mass_mix_ratio_dust = 1.0 # mass mixing rati0                     TODO
activation_time_dust = 3.0 

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
    particle_density::T
    osmotic_coeff::T
    molar_mass::T 
    dissoc::T
    mass_frac::T 
    mass_mix_ratio::T
    radius::T
    radius_stdev::T
    activation_time::T
end

# complete aerosol model struct
struct aerosol_model{T}
    modes::T
end 

# 3. Populate structs to pass into functions/run calculations
# Test cases 1-3 (Just Sea Salt)
accum_mode_seasalt = mode((particle_density_seasalt_accum,), (osmotic_coeff_seasalt,), 
                        (molar_mass_seasalt,), 
                            (dissoc_seasalt,), (mass_frac_seasalt,), (mass_mix_ratio_seasalt,),
                        (radius_seasalt_accum,),
                        (radius_stdev_seasalt_accum,), (activation_time_seasalt,))

coarse_mode_seasalt = mode((particle_density_seasalt_accum,), (osmotic_coeff_seasalt,), 
                        (molar_mass_seasalt,), 
                            (dissoc_seasalt,), (mass_frac_seasalt,), (mass_mix_ratio_seasalt,),
                        (radius_seasalt_coarse,),
                        (radius_stdev_seasalt_coarse,), (activation_time_seasalt,))

coarse_mode_ssanddust = mode((particle_density_seasalt_accum, particle_density_dust_coarse), 
                            (osmotic_coeff_seasalt, osmotic_coeff_dust), 
                        (molar_mass_seasalt, molar_mass_dust), 
                            (dissoc_seasalt, dissoc_dust), 
                            (mass_frac_seasalt,mass_frac_dust), 
                            (mass_mix_ratio_seasalt, mass_mix_ratio_dust),
                        (radius_seasalt_coarse, radius_dust_coarse),
                        (radius_stdev_seasalt_coarse, radius_stdev_dust_accum),
                         (activation_time_seasalt, activation_time_dust))


"""
Coefficient of curvature effect: 
        coeff_of_curve(act_time)
    - 'act_time' - time of activation
    Returns coeff_of_curve (coefficient of the curvature effect); key 
    input into other functions. Takes in scalar and outputs 
    scalar.
"""
function coeff_of_curve(am::aerosol_model)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # take weighted average of activation times 
        n_comps = length(mode_i.particle_density)
        numerator = sum(n_comps) do j
            mode_i.activation_time[j]*mode_i.particle_density[j]
        end 
        denominator = sum(n_comps) do j
            mode_i.particle_density[j]
        end
        avg_activation_time = numerator/denominator 
        top = 2*avg_activation_time*Mw
        bottom = density_water*R*T
        top/bottom

    end

end

am = aerosol_model((accum_mode_seasalt, coarse_mode_ssanddust))
testoutput1 = coeff_of_curve(am)
println("This is the result of coeff_of_curve: ", testoutput1)
