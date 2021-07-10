  
"""
Aerosol activation module, which includes:
- mean hygroscopicity for each mode of an aerosol model
- critical supersaturation for each mode of an aerosol model
- maximum supersaturation for an entire aerosol model
- total number of particles actived in a system given an aerosol model 
"""


using SpecialFunctions

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, T_freeze
using CLIMAParameters.Atmos.Microphysics

export alpha_sic
export gamma_sic
export mean_hygroscopicity
export coeff_of_curvature
export critical_supersaturation
export max_supersatuation
export total_N_activated

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

