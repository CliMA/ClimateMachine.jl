using SpecialFunctions
# Building the test structures
# 1. Set Aerosol parameters: 
# Sea Salt--universal parameters
UPDFT_VELO = 5 
molmass_water = 0.0180153
R = 8.3145
TEMP = 273.15
P_SAT = 100000
K_therm = 2.4e-2
D_vapor = 2.26e-5
LH_v0 = 2.5008e6
cp_v = 1859
grav = 9.81
ρ_cloud_liq  = 1000
R_v = R/molmass_water
G_DIFF = (((LH_v0/(K_therm*TEMP))*(((LH_v0)/(R_v*TEMP))-1))+((R_v * TEMP)/(P_SAT * D_vapor)))^(-1)
surface_tension = 0.0757


osmotic_coeff_seasalt = 0.9 # osmotic coefficient
molar_mass_seasalt = 0.058443 # sea salt molar mass; kg/mol
rho_seasalt = 2170.0 # sea salt density; kg/m^3
dissoc_seasalt = 2.0 # Sea salt dissociation                         
mass_frac_seasalt = 1.0 # mass fraction                              TODO
mass_mix_ratio_seasalt = 1.0 # mass mixing rati0                    TODO
activation_time_seasalt = 0.0757


# Sea Salt -- Accumulation mode
radius_seasalt_accum = 0.000000243 # mean particle radius (m)
radius_stdev_seasalt_accum = 1.4 # mean particle stdev (m)
particle_density_seasalt_accum = 1e8 #000000 # particle density (1/m^3)

# Sea Salt -- Coarse Mode
radius_seasalt_coarse = 0.0000015 # mean particle radius (m)
radius_stdev_seasalt_coarse = 0.0000021 # mean particle stdev(m)

# TODO: Dust parameters (just copy and pasted seasalt values rn)
# Dust--universal parameters
osmotic_coeff_dust = 0.9 # osmotic coefficient
molar_mass_dust = 0.058443 * 10  # sea salt molar mass; kg/mol
particle_density_dust_coarse = 1e5
rho_dust = 2170.0 # sea salt density; kg/m^3
dissoc_dust = 5.0 # Sea salt dissociation                         
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
    dry_radius::T
    radius_stdev::T
    density::T
    n_components::INT64
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
                        (radius_stdev_seasalt_accum,), (activation_time_seasalt,), (rho_seasalt,), 1)

coarse_mode_seasalt = mode((particle_density_seasalt_accum,), (osmotic_coeff_seasalt,), 
                        (molar_mass_seasalt,), 
                            (dissoc_seasalt,), (mass_frac_seasalt,), (mass_mix_ratio_seasalt,),
                        (radius_seasalt_coarse,),
                        (radius_stdev_seasalt_coarse,), (activation_time_seasalt,), (rho_seasalt,), 1)

coarse_mode_ssanddust = mode((particle_density_seasalt_accum, particle_density_dust_coarse), 
                            (osmotic_coeff_seasalt, osmotic_coeff_dust), 
                        (molar_mass_seasalt, molar_mass_dust), 
                            (dissoc_seasalt, dissoc_dust), 
                            (mass_frac_seasalt,mass_frac_dust), 
                            (mass_mix_ratio_seasalt, mass_mix_ratio_dust),
                        (radius_seasalt_coarse, radius_dust_coarse),
                        (radius_stdev_seasalt_coarse, radius_stdev_dust_accum),
                         (activation_time_seasalt, activation_time_dust), (rho_seasalt, rho_dust), 2)



test_mode = mode((particle_density_seasalt_accum,), (osmotic_coeff_seasalt,), 
                 (molar_mass_seasalt,), 
                 (dissoc_seasalt,), (mass_frac_seasalt,), (mass_mix_ratio_seasalt,),
                 (radius_seasalt_accum,),
                 (radius_stdev_seasalt_accum,), (activation_time_seasalt,), 
                 (rho_seasalt,), 1)


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
        numerator = sum(1:n_comps) do j
            mode_i.particle_density[j]*mode_i.molar_mass[j]
        end
        denominator = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_molar_mass = numerator/denominator
        exp1 = (grav*molmass_water*LH_v0) / (cp_v*R*TEMP^2)
        exp2 = (grav*avg_molar_mass)/(R*TEMP)
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
        numerator = sum(1:n_comps) do j
            mode_i.particle_density[j]*mode_i.molar_mass[j]
        end
        denominator = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_molar_mass = numerator/denominator
        exp1 = (R*TEMP)/(P_sat*molmass_water)
        exp2 = (molmass_water*LH_v0^2)/(cp_v*P_SAT*avg_molar_mass*TEMP)
        exp1+exp2 
    end
end

"""
coeff_of_curvature(am::aerosol_model)
    - am -- aerosol_model
    
    Returns coeff_of_curvature (coefficient of the curvature effect); key 
    input into other functions. Utilizes activation time and particle density 
    from modes struct.
"""
function coeff_of_curvature(am::aerosol_model)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # take weighted average of activation times 
        n_comps = length(mode_i.particle_density)
        numerator = sum(1:n_comps) do j
            mode_i.activation_time[j]*mode_i.particle_density[j]
        end 
        denominator = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_activation_time = numerator/denominator 
        top = 2*avg_activation_time*molmass_water
        bottom = ρ_cloud_liq*R*TEMP
        top/bottom

    end

end

"""
mean_hygroscopicity(am::aerosol_model)
    - am -- aerosol model
    Returns the mean hygroscopicty along each mode of an inputted aerosol model. 
    Utilizes mass mixing ratio, dissociation, mass fraction, molar mass, particle 
    density from mode struct. 
"""
function mean_hygroscopicity(am::aerosol_model)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        n_comps = length(mode_i.particle_density)
        top = sum(1:n_comps) do j
            mode_i.mass_mix_ratio[j]*mode_i.dissoc[j]*
            mode_i.osmotic_coeff[j]*mode_i.mass_frac[j]*
            (1/mode_i.molar_mass[j])
        end
        bottom = sum(1:n_comps) do j 
            mode_i.mass_mix_ratio[j]/mode_i.density[j]
        end 
        coeff = molmass_water/ρ_cloud_liq
        coeff*(top/bottom)
    end 
end

"""
TO DO: DOCSTRING 
"""
function critical_supersaturation(am::aerosol_model)
    coeff_of_curve = coeff_of_curvature(am)
    mh = mean_hygroscopicity(am)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # weighted average of mode radius
        n_comps = length(mode_i.particle_density)
        numerator = sum(1:n_comps) do j
            mode_i.radius[j]*mode_i.particle_density[j]
        end 
        denominator = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_radius = numerator/denominator
        exp1 = 2 / (mh[i])^(.5)
        exp2 = (coeff_of_curve[i]/(3*avg_radius))^(3/2)
        exp1*exp2
    end
end

"""
TO DO: DOCSTRING 
"""
function max_supersaturation(am, P_SAT)
    alpha = alpha_sic(am)
    gamma = gamma_sic(am, P_SAT)
    A = coeff_of_curvature(am)
    Sm = critical_supersaturation(am)
    X = sum(1:length(am.modes)) do i 
        mode_i = am.modes[i]

        # weighted avgs of diff params:
        n_comps = length(mode_i.particle_density)
        # radius_stdev
        num = sum(1:n_comps) do j 
            mode_i.particle_density[j]  *  mode_i.radius_stdev[j]
        end
        den = sum(1:n_comps) do j 
            mode_i.particle_density[j]
        end 
        avg_radius_stdev = num/den 
        
        total_particles = sum(1:n_comps) do j 
            mode_i.particle_density[j]
        end
        f = 0.5  *  exp(2.5  *  (log(avg_radius_stdev))^2 )
        g = 1 + 0.25  *  log(avg_radius_stdev)        
        zeta = (2 * A[i] * (1/3))  *  ((alpha[i] * UPDFT_VELO)/G_DIFF)^(.5)
        eta = (((alpha[i]*UPDFT_VELO)/(G_DIFF))^(3/2))/(2*pi*ρ_cloud_liq *gamma[i]*total_particles)
        exp1 = 1/(Sm[i])^2
        exp2 = f*(zeta/eta)^(3/2)
        exp3 = g*((Sm[i]^2)/(eta+3*zeta))^(3/4)

        exp1*(exp2+exp3)
    end
    return 1/((X)^(1/2))

end

"""
TO DO: DOCSTRING 
"""
function total_N_activated(am::aerosol_model)
    smax = max_supersaturation(am, P_SAT)
    sm = critical_supersaturation(am)
    TOTN =  sum(1:length(am.modes)) do i
        mode_i = am.modes[i]
        # weighted avgs of diff params:
        n_comps = length(mode_i.particle_density)
        # radius_stdev
        num = sum(1:n_comps) do j 
            mode_i.particle_density[j]  *  mode_i.radius_stdev[j]
        end
        den = sum(1:n_comps) do j 
            mode_i.particle_density[j]
        end 
        avg_radius_stdev = num/den 
        
        total_particles = sum(1:n_comps) do j 
            mode_i.particle_density[j]
        end
        utop = 2*log(sm[i]/smax)
        ubottom = 3*(2^.5)*log(avg_radius_stdev)
        u = utop/ubottom
        total_particles*(.5)*(1-erf(u))
    end
    return TOTN
end

am = aerosol_model((test_mode, test_mode))
RESULT = total_N_activated(am)
println(RESULT)
#println('max supersat output: ', max_supersatuation(am))
println(754696.317709)


# # import AerosolActivation
# # Pkg.add("AerosolActivation")
# # Pkg.add("ClimateMachine"); Pkg.add("CLIMAParameters"); Pkg.add("AerosolActivation")
# # using RoughDrafts/AerosolActivation

# # using ClimateMachine.AerosolActivation
# #using CLIMAParameters
# #using CLIMAParameters.Planet: ρ_cloud_liq , R_v, grav, molmass_water, LH_v0, cp_v
# #using CLIMAParameters.Atmos: K_therm, D_vapor
# # tester = grav * 10
# # println(typeof(tester))
# using SpecialFunctions


# ρ_cloud_liq  = 1e3
# molmass_water = 18.01528e-3 
# molmass_dryair = 28.97e-3
# R = 8.3144621
# R_v = R/molmass_water
# grav = 9.81
# R_d = R/molmass_dryair
# LH_v0 = 2.5008e6
# cp_v = 1859
# K_therm = 2.4e-2
# D_vapor = 2.26e-5
# TEMP = 273.15
# P_SAT = 100000
# P = 100000
# G_DIFF = (((LH_v0/(K_therm*TEMP))*(((LH_v0)/(R_v*TEMP))-1))+((R_v * TEMP)/(P_SAT * D_vapor)))^(-1)
# UPDFT_VELO = 5 


# # Building the test structures
# # 1. Set Aerosol parameters: 
# # Sea Salt--universal parameters
# osmotic_coeff_seasalt = 0.9 # osmotic coefficient
# molar_mass_seasalt = 0.058443 # sea salt molar mass; kg/mol
# rho_seasalt = 2170.0 # sea salt density; kg/m^3
# dissoc_seasalt = 2.0 # Sea salt dissociation                         
# mass_frac_seasalt = 1.0 # mass fraction                              TODO
# mass_mix_ratio_seasalt = 1.0 # mass mixing rati0                    TODO
# activation_time_seasalt = 1.0 

# # Sea Salt -- Accumulation mode
# radius_seasalt_accum = 0.000000243 # mean particle radius (m)
# radius_stdev_seasalt_accum = 0.0000014 # mean particle stdev (m)
# particle_density_seasalt_accum = 1e8 #000000 # particle density (1/m^3)

# # Sea Salt -- Coarse Mode
# radius_seasalt_coarse = 0.0000015 # mean particle radius (m)
# radius_stdev_seasalt_coarse = 0.0000021 # mean particle stdev(m)

# # TODO: Dust parameters (just copy and pasted seasalt values rn)
# # Dust--universal parameters
# osmotic_coeff_dust = 0.9 # osmotic coefficient
# molar_mass_dust = 0.058443 * 10  # sea salt molar mass; kg/mol
# particle_density_dust_coarse = 1e5
# rho_dust = 2170.0 # sea salt density; kg/m^3
# dissoc_dust = 5.0 # Sea salt dissociation                         
# mass_frac_dust = 1.0 # mass fraction                              TODO
# mass_mix_ratio_dust = 1.0 # mass mixing rati0                     TODO
# activation_time_dust = 3.0 

# # Dust -- Accumulation mode
# radius_dust_accum = 0.000000243 # mean particle radius (m)
# radius_stdev_dust_accum = 0.0000014 # mean particle stdev (m)

# # Dust -- Coarse Mode
# radius_dust_coarse = 0.00015 # mean particle radius (m)
# radius_stdev_dust_accum = 0.0000021 # mean particle stdev(m)

# # Abdul-Razzak and Ghan 

# # 2. Create structs that parameters can be pass through
# # individual aerosol mode struct
# struct mode{T}
#     particle_density::T
#     osmotic_coeff::T
#     molar_mass::T 
#     dissoc::T
#     mass_frac::T 
#     mass_mix_ratio::T
#     radius::T
#     radius_stdev::T
#     activation_time::T
#     density::T
# end

# # complete aerosol model struct
# struct aerosol_model{T}
#     modes::T
# end 

# # 3. Populate structs to pass into functions/run calculations
# # Test cases 1-3 (Just Sea Salt)
# accum_mode_seasalt = mode((particle_density_seasalt_accum,), (osmotic_coeff_seasalt,), 
#                         (molar_mass_seasalt,), 
#                             (dissoc_seasalt,), (mass_frac_seasalt,), (mass_mix_ratio_seasalt,),
#                         (radius_seasalt_accum,),
#                         (radius_stdev_seasalt_accum,), (activation_time_seasalt,), (rho_seasalt,))

# coarse_mode_seasalt = mode((particle_density_seasalt_accum,), (osmotic_coeff_seasalt,), 
#                         (molar_mass_seasalt,), 
#                             (dissoc_seasalt,), (mass_frac_seasalt,), (mass_mix_ratio_seasalt,),
#                         (radius_seasalt_coarse,),
#                         (radius_stdev_seasalt_coarse,), (activation_time_seasalt,), (rho_seasalt,))

# coarse_mode_ssanddust = mode((particle_density_seasalt_accum, particle_density_dust_coarse), 
#                             (osmotic_coeff_seasalt, osmotic_coeff_dust), 
#                         (molar_mass_seasalt, molar_mass_dust), 
#                             (dissoc_seasalt, dissoc_dust), 
#                             (mass_frac_seasalt,mass_frac_dust), 
#                             (mass_mix_ratio_seasalt, mass_mix_ratio_dust),
#                         (radius_seasalt_coarse, radius_dust_coarse),
#                         (radius_stdev_seasalt_coarse, radius_stdev_dust_accum),
#                          (activation_time_seasalt, activation_time_dust), (rho_seasalt, rho_dust))





# """
# alpha_sic(aero_mm)
#     - am -- aerosol_model                      
    
#     Returns coefficient relevant to other functions. Uses aerosol
#     Molar mass
# """
# function alpha_sic(am::aerosol_model)
#     return ntuple(length(am.modes)) do i 
#         mode_i = am.modes[i]
#         # Find weighted molar mass of mode
#         n_comps = length(mode_i.particle_density)
#         numerator = sum(1:n_comps) do j
#             mode_i.particle_density[j]*mode_i.molar_mass[j]
#         end
#         denominator = sum(1:n_comps) do j
#             mode_i.particle_density[j]
#         end
#         avg_molar_mass = numerator/denominator
#         exp1 = (grav*molmass_water*LH_v0) / (cp_v*R*TEMP^2)
#         exp2 = (grav*avg_molar_mass)/(R*TEMP)
#         exp1-exp2
#     end
# end

# """
# gamma_sic(aero_mm)
#     - am -- aerosol_model                      
    
#     Returns coefficient relevant to other functions. Uses aerosol
#     Molar mass and water saturation pressure. 
# """
# function gamma_sic(am::aerosol_model, P_sat)
#     return ntuple(length(am.modes)) do i 
#         mode_i = am.modes[i]
#         # Find weighted molar mass of mode
#         n_comps = length(mode_i.particle_density)
#         numerator = sum(1:n_comps) do j
#             mode_i.particle_density[j]*mode_i.molar_mass[j]
#         end
#         denominator = sum(1:n_comps) do j
#             mode_i.particle_density[j]
#         end
#         avg_molar_mass = numerator/denominator
#         exp1 = (R*TEMP)/(P_sat*molmass_water)
#         exp2 = (molmass_water*LH_v0^2)/(cp_v*P*avg_molar_mass*TEMP)
#         exp1+exp2 
#     end
# end

# """
# coeff_of_curvature(am::aerosol_model)
#     - am -- aerosol_model
    
#     Returns coeff_of_curvature (coefficient of the curvature effect); key 
#     input into other functions. Utilizes activation time and particle density 
#     from modes struct.
# """
# function coeff_of_curvature(am::aerosol_model)
#     return ntuple(length(am.modes)) do i 
#         mode_i = am.modes[i]
#         # take weighted average of activation times 
#         n_comps = length(mode_i.particle_density)
#         numerator = sum(1:n_comps) do j
#             mode_i.activation_time[j]*mode_i.particle_density[j]
#         end 
#         denominator = sum(1:n_comps) do j
#             mode_i.particle_density[j]
#         end
#         avg_activation_time = numerator/denominator 
#         top = 2*avg_activation_time*molmass_water
#         bottom = ρ_cloud_liq *R*TEMP
#         top/bottom

#     end

# end

# """
# mean_hygroscopicity(am::aerosol_model)
#     - am -- aerosol model
#     Returns the mean hygroscopicty along each mode of an inputted aerosol model. 
#     Utilizes mass mixing ratio, dissociation, mass fraction, molar mass, particle 
#     density from mode struct. 
# """
# function mean_hygroscopicity(am::aerosol_model)
#     return ntuple(length(am.modes)) do i 
#         mode_i = am.modes[i]
#         n_comps = length(mode_i.particle_density)
#         top = sum(1:n_comps) do j
#             mode_i.mass_mix_ratio[j]*mode_i.dissoc[j]*
#             mode_i.osmotic_coeff[j]*mode_i.mass_frac[j]*
#             (1/mode_i.molar_mass[j])
#         end
#         bottom = sum(1:n_comps) do j 
#             mode_i.mass_mix_ratio[j]/mode_i.density[j]
#         end 
#         coeff = molmass_water/ρ_cloud_liq 
#         coeff*(top/bottom)
#     end 
# end

# """
# TO DO: DOCSTRING 
# """
# function critical_supersaturation(am::aerosol_model)
#     coeff_of_curve = coeff_of_curvature(am)
#     mh = mean_hygroscopicity(am)
#     return ntuple(length(am.modes)) do i 
#         mode_i = am.modes[i]
#         # weighted average of mode radius
#         n_comps = length(mode_i.particle_density)
#         numerator = sum(1:n_comps) do j
#             mode_i.radius[j]*mode_i.particle_density[j]
#         end 
#         denominator = sum(1:n_comps) do j
#             mode_i.particle_density[j]
#         end
#         avg_radius = numerator/denominator
#         exp1 = 2 / (mh[i])^(.5)
#         exp2 = (coeff_of_curve[i]/3*avg_radius)^(3/2)
#         exp1*exp2
#     end
# end

# """
# TO DO: DOCSTRING 
# """
# function max_supersaturation(am, P_SAT)
#     alpha = alpha_sic(am)
#     gamma = gamma_sic(am, P_SAT)
#     A = coeff_of_curvature(am)
#     Sm = critical_supersaturation(am)
#     X = sum(1:length(am.modes)) do i 

#         mode_i = am.modes[i]

#         # weighted avgs of diff params:
#         n_comps = length(mode_i.particle_density)
#         # radius_stdev
#         num = sum(1:n_comps) do j 
#             mode_i.particle_density[j]  *  mode_i.radius_stdev[j]
#         end
#         den = sum(1:n_comps) do j 
#             mode_i.particle_density[j]
#         end 
#         avg_radius_stdev = num/den 
        
#         total_particles = sum(1:n_comps) do j 
#             mode_i.particle_density[j]
#         end
#         f = 0.5  *  exp(2.5  *  (log(avg_radius_stdev))^2 )
#         g = 1 + 0.25  *  log(avg_radius_stdev) 

#         zeta = (2 * A[i] * (1/3))  *  ((alpha[i] * UPDFT_VELO)/G_DIFF)^(.5)
#         eta = (((alpha[i]*UPDFT_VELO)/(G_DIFF))^(3/2))/(2*pi*ρ_cloud_liq *gamma[i]*total_particles)

#         exp1 = 1/(Sm[i])^2
#         exp2 = f*(zeta/eta)^(3/2)
#         exp3 = g*((Sm[i]^2)/(eta+3*zeta))^(3/4)

#         exp1*(exp2+exp3)
#     end
#     return 1/((X)^(1/2))

# end

# """
# TO DO: DOCSTRING 
# """
# function total_N_activated(am::aerosol_model)
#     smax = max_supersaturation(am, P_SAT)
#     sm = critical_supersaturation(am)
#     return sum(1:length(am.modes)) do i
#         mode_i = am.modes[i]
#         # weighted avgs of diff params:
#         n_comps = length(mode_i.particle_density)
#         # radius_stdev
#         num = sum(1:n_comps) do j 
#             mode_i.particle_density[j]  *  mode_i.radius_stdev[j]
#         end
#         den = sum(1:n_comps) do j 
#             mode_i.particle_density[j]
#         end 
#         avg_radius_stdev = num/den 
        
#         total_particles = sum(1:n_comps) do j 
#             mode_i.particle_density[j]
#         end

#         utop = 2*log(sm[i]/smax)
#         ubottom = 3*(2^.5)*log(avg_radius_stdev)
#         u = utop/ubottom
#         total_particles*(1/2)*(1-erf(u))
#     end 
# end
# am = aerosol_model((accum_mode_seasalt, coarse_mode_ssanddust))
# println("alpha_sic: ", alpha_sic(am))
# println("gamma sic: ", gamma_sic(am, P_SAT))
# println("coeff of curve: ", coeff_of_curvature(am))
# println("mean hygro: ", mean_hygroscopicity(am))
# println("crit sat: ", critical_supersaturation(am))
# println("max sat: ", max_supersaturation(am, P_SAT))
# println("N act: ", total_N_activated(am))