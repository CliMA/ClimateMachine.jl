# GET FROM CLIMA PARAMATERS

molar_mass_water = 18.0
density_water = 1000.0
# Universal parameters:

# Building the test structures
# 1. Set Aerosol parameters: 

# Sea Salt--universal parameters
osmotic_coeff_seasalt = (0.9,) # osmotic coefficient
molar_mass_seasalt = (0.058443,) # sea salt molar mass; kg/mol
rho_seasalt = (2170.0,) # sea salt density; kg/m^3
dissoc_seasalt = (2.0,) # Sea salt dissociation                         
mass_frac_seasalt = (1.0,) # mass fraction                              TODO
mass_mix_ratio_seasalt = (1.0,) # mass mixing rati0                     TODO

# Sea Salt -- Accumulation mode
radius_seasalt_accum = (0.000000243,) # mean particle radius (m)
radius_stdev_seasalt_accum = (0.0000014,) # mean particle stdev (m)
particle_density_seasalt_accum = (100.0,) #000000 # particle density (1/m^3)

# Sea Salt -- Coarse Mode
radius_seasalt_coarse = (0.0000015,) # mean particle radius (m)
radius_stdev_seasalt_coarse = (0.0000021,) # mean particle stdev(m)

# TODO: Dust parameters (just copy and pasted seasalt values rn)
# Dust--universal parameters
osmotic_coeff_dust = (0.9,) # osmotic coefficient
molar_mass_dust = (0.058443,) # sea salt molar mass; kg/mol
rho_dust = (2170.0,) # sea salt density; kg/m^3
dissoc_dust = (2.0,) # Sea salt dissociation                         
mass_frac_dust = (1.0,) # mass fraction                              TODO
mass_mix_ratio_dust = (1.0,) # mass mixing rati0                     TODO

# Dust -- Accumulation mode
radius_dust_accum = (0.000000243,) # mean particle radius (m)
radius_stdev_dust_accum = (0.0000014,) # mean particle stdev (m)

# Dust -- Coarse Mode
radius_dust_coarse = (0.0000015,) # mean particle radius (m)
radius_stdev_dust_accum = (0.0000021,) # mean particle stdev(m)

# Abdul-Razzak and Ghan 

# 2. Create structs that parameters can be pass through
# individual aerosol mode struct
struct mode{T}
    particle_density::Tuple
    osmotic_coeff::Tuple
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
end 

# 3. Populate structs to pass into functions/run calculations
# Test cases 1-3 (Just Sea Salt)
accum_mode_seasalt = mode(particle_density_seasalt_accum, osmotic_coeff_seasalt, 
                        molar_mass_seasalt, 
                            dissoc_seasalt, mass_frac_seasalt, mass_mix_ratio_seasalt,
                        radius_seasalt_coarse,
                        radius_stdev_seasalt_coarse)

coarse_mode_seasalt = mode(particle_density_seasalt_accum, osmotic_coeff_seasalt, 
                    molar_mass_seasalt, 
                   dissoc_seasalt, mass_frac_seasalt, mass_mix_ratio_seasalt,
                    radius_seasalt_accum,
                   radius_stdev_seasalt_accum)

# aerosolmodel_testcase1 = aerosol_model(accum_mode_seasalt)
# aerosolmodel_testcase2 = aerosol_model(coarse_mode_seasalt)
test_mode = mode(particle_density_seasalt_accum, osmotic_coeff_seasalt, 
                molar_mass_seasalt, 
                dissoc_seasalt, mass_frac_seasalt, mass_mix_ratio_seasalt,
                radius_seasalt_coarse,
                radius_stdev_seasalt_coarse)
# aerosolmodel_testcase3 = aerosol_model((test_mode, test_mode))


"""
alpha_sic(aero_mm)
    - am -- aerosol_model                      
    
    Returns coefficient relevant to other functions. Uses aerosol
    Molar mass
"""
g = 9.81
Mw = 18.1
L = 10.0
Cp = 1000
T = 273.15
R = 8.1
P=100000
P_saturation = 100000

function alpha_sic(am::aerosol_model)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        aerosol_molar_mass = mode_i.molar_mass[1]
        exp1 = (g*Mw*L) / (Cp*R*T^2)
        exp2 = (g*aerosol_molar_mass)/(R*T)
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
        aerosol_molar_mass = mode_i.molar_mass[1]
        exp1 = (R*T)/(P_sat*Mw)
        exp2 = (Mw*L^2)/(Cp*P*aerosol_molar_mass*T)
        exp1+exp2 
end

am = aerosol_model((accum_mode_seasalt, coarse_mode_seasalt))
print(length(am.modes))
testoutput1 = alpha_sic(am)
testoutput2 = gamma_sic(am, P_saturation)
print("This is the result of alpha_sic:", testoutput1)
print("This is the result of gamma_sic:", testouput2)


