
"""
Aerosol activation module, which includes:
- mean hygroscopicity for each mode of an aerosol model
- critical supersaturation for each mode of an aerosol model
- maximum supersaturation for an entire aerosol model
- total number of particles actived in a system given an aerosol model
"""
#module AerosolActivation

using SpecialFunctions

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, T_freeze
using CLIMAParameters.Atmos.Microphysics

#export alpha_sic
#export gamma_sic
export mean_hygroscopicity
#export coeff_of_curvature
#export critical_supersaturation
#export max_supersatuation
#export total_N_activated

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
"""
alpha_sic(aero_mm)
    - am -- aerosol_model

    Returns coefficient relevant to other functions. Uses aerosol
    Molar mass
"""
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

function alpha_sic(am::aerosol_model)
    return ntuple(length(am.modes)) do i
        mode_i = am.modes[i]
        # Find weighted molar mass of mode
        n_comps = length(mode_i.particle_density)
        numerator = sum(n_comps) do j
            mode_i.particle_density[j]*mode_i.molar_mass[j]
        end
        denominator = sum(n_comps) do j
            mode_i.particle_density[j]
        end
        avg_molar_mass = numerator/denominator
        exp1 = (g*Mw*L) / (Cp*R*T^2)
        exp2 = (g*avg_molar_mass)/(R*T)
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
        numerator = sum(n_comps) do j
            mode_i.particle_density[j]*mode_i.molar_mass[j]
        end
        denominator = sum(n_comps) do j
            mode_i.particle_density[j]
        end
        avg_molar_mass = numerator/denominator
        exp1 = (R*T)/(P_sat*Mw)
        exp2 = (Mw*L^2)/(Cp*P*avg_molar_mass*T)
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
        top = sum(n_comps) do j
            mode_i.mass_mix_ratio[j]*mode_i.dissoc[j]*
            mode_i.osmotic_coeff[j]*mode_i.mass_frac[j]*
            (1/mode_i.molar_mass[j])
        end
        bottom = sum(n_comps) do j
            mode_i.mass_mix_ratio[j]/mode_i.density[j]
        end
        coeff = Mw/density_water
        coeff*(top/bottom)
    end
end

#end # module AerosolActivation.jl
