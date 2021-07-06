using SpecialFunctions

# GET FROM CLIMA PARAMATERS
g = 9.81
Mw = 18.1
L = 1000.0
Cp = 1
T = 273.15
R = 8.1
P=100000
P_saturation = 100000
molar_mass_water = 18.0
density_water = 1000.0
diff = 1 
updft_velo = 5 
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
molar_mass_dust = 0.058443 * 10 # sea salt molar mass; kg/mol
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
radius_dust_coarse = 0.00015 # mean particle radius (m)
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
    density::T
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
                        (radius_stdev_seasalt_accum,), (activation_time_seasalt,), (rho_seasalt,))

coarse_mode_seasalt = mode((particle_density_seasalt_accum,), (osmotic_coeff_seasalt,), 
                        (molar_mass_seasalt,), 
                            (dissoc_seasalt,), (mass_frac_seasalt,), (mass_mix_ratio_seasalt,),
                        (radius_seasalt_coarse,),
                        (radius_stdev_seasalt_coarse,), (activation_time_seasalt,), (rho_seasalt,))

coarse_mode_ssanddust = mode((particle_density_seasalt_accum, particle_density_dust_coarse), 
                            (osmotic_coeff_seasalt, osmotic_coeff_dust), 
                        (molar_mass_seasalt, molar_mass_dust), 
                            (dissoc_seasalt, dissoc_dust), 
                            (mass_frac_seasalt,mass_frac_dust), 
                            (mass_mix_ratio_seasalt, mass_mix_ratio_dust),
                        (radius_seasalt_coarse, radius_dust_coarse),
                        (radius_stdev_seasalt_coarse, radius_stdev_dust_accum),
                         (activation_time_seasalt, activation_time_dust), (rho_seasalt, rho_dust))



# TODO: Revisit this function, do we need to create new struct? 
# function weighted_average(am::aerosol_model, param)
#     return ntuple(length(am.modes)) do i 
#         mode_i = am.modes[i]
#         n_comps = length(mode_i.particle_density)
#         num = sum( 1:n_comps) do j 
#             mode_i.particle_density[j] * mode_i.molar_mass[j]
#         end
#         den = sum( 1:n_comps) do j
#             mode_i.particle_density[j]
#         end
#         num/den
#     end 
# end
# am = aerosol_model((accum_mode_seasalt, coarse_mode_ssanddust))
# testoutput1 = weighted_average(am, molar_mass)
# println("Weighted averages: ", testoutput1)


"""
alpha_sic(aero_mm)
    - am -- aerosol_model                      
    
    Returns coefficient relevant to other functions. Uses aerosol
    Molar mass
"""


function alpha_sic(am::aerosol_model)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # Find weighted molar mass of mode
        n_comps = length(mode_i.particle_density)
        println("This is number of comps:",  1:n_comps, "mode:", i)
        numerator = sum( 1:n_comps) do j
            mode_i.particle_density[j] * mode_i.molar_mass[j]
        end
        denominator = sum( 1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_molar_mass = numerator/denominator
        exp1 = (g * Mw * L) / (Cp * R * T^2)
        exp2 = (g * avg_molar_mass)/(R * T)
        exp1-exp2
    end
end

"""
gamma_sic(aero_mm)
    - am -- aerosol_model                      
    
    Returns coefficient relevant to other functions. Uses aerosol
    Molar mass and water saturation pressure. 
"""
function gamma_sic(am::aerosol_model, P_sat)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # Find weighted molar mass of mode
         n_comps = length(mode_i.particle_density)
        numerator = sum( 1:n_comps) do j
            mode_i.particle_density[j] * mode_i.molar_mass[j]
        end
        denominator = sum( 1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_molar_mass = numerator/denominator
        exp1 = (R * T)/(P_sat * Mw)
        exp2 = (Mw * L^2)/(Cp * P * avg_molar_mass * T)
        exp1+exp2 
    end
end       
"""
mean_hygroscopicity(am::aerosol_model)

DO DOCSTRING
"""
function mean_hygroscopicity(am::aerosol_model)
    return map(am.modes) do mode_i 
        n_comps = length(mode_i.particle_density)
        top = sum(1:n_comps) do j
            mode_i.mass_mix_ratio[j] * mode_i.dissoc[j] * 
            mode_i.osmotic_coeff[j] * mode_i.mass_frac[j] * 
            (1 / mode_i.molar_mass[j])
        end
        bottom = sum(1:n_comps) do j 
            mode_i.mass_mix_ratio[j] / mode_i.density[j]
        end  
        Mw / density_water * top / bottom
    end 
end

function coeff_of_curve(am::aerosol_model)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # take weighted average of activation times 
        n_comps = length(mode_i.particle_density)
        numerator = sum(1:n_comps) do j
            mode_i.activation_time[j] * mode_i.particle_density[j]
        end 
        denominator = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_activation_time = numerator / denominator 
        top = 2 * avg_activation_time * Mw
        bottom = density_water * R * T
        top/bottom

    end

end

function critical_supersaturation(am::aerosol_model)
    coeff_of_curvature = coeff_of_curve(am)
    mh = mean_hygroscopicity(am)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # weighted average of mode radius
         n_comps = length(mode_i.particle_density)
        numerator = sum( 1:n_comps) do j
            mode_i.radius[j] * mode_i.particle_density[j]
        end 
        denominator = sum( 1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_radius = numerator/denominator
        exp1 = 2 / (mh[i])^(.5)
        exp2 = (coeff_of_curvature[i]/3 * avg_radius)^(3/2)
        exp1 * exp2
    end
end

function max_supersatuation(am:aerosol_model)
    alpha = alpha_sic(am)
    gamma = gamma_sic(am, P_saturation)
    A = coeff_of_curve(am)
    Sm = critical_supersaturation(am)
    return sum(1:length(am.modes)) do i 

        mode_i = am.modes[i]

        # weighted avgs of diff params:
        n_comps = length(mode_i.particle_density)
        # radius_stdev
        num = sum( 1:n_comps) do j 
            mode_i.particle_density[j]  *  mode_i.radius_stdev[j]
        end
        den = sum( 1:n_comps) do j 
            mode_i.particle_density[j]
        end 
        avg_radius_stdev = num/den 
        
        total_particles = sum(1:n_comps) do j 
            mode_i.particle_density[j]
        end
        f = 0.5  *  exp(2.5  *  (log(avg_radius_stdev))^2 )
        g = 1 + 0.25  *  log(avg_radius_stdev) 

        zeta = (2 * A[i] * (1/3))  *  ((alpha[i] * updft_velo)/diff)^(.5)
        eta = (((alpha[i]*updft_velo)/(diff))^(3/2))/(2*pi*density_water*gamma[i]*total_particles)

        exp1 = 1/(Sm[i])^2
        exp2 = f*(zeta/eta)^(3/2)
        exp3 = g*((Sm[i]^2)/(eta+3*zeta))^(3/4)

        (exp1*(exp2+exp3))^(1/2)
    end

end


function total_N_activated(am::aerosol_model)
    smax = max_supersatuation(am)
    sm = critical_supersaturation(am)
    return sum(1:length(am.modes)) do i
        mode_i = am.modes[i]
        # weighted avgs of diff params:
        n_comps = length(mode_i.particle_density)
        # radius_stdev
        num = sum( 1:n_comps) do j 
            mode_i.particle_density[j]  *  mode_i.radius_stdev[j]
        end
        den = sum( 1:n_comps) do j 
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
am = aerosol_model((accum_mode_seasalt, coarse_mode_ssanddust))
testoutput1 = total_N_activated(am)
println("This is the result of total_N_activated: ", testoutput1)