"""
Aerosol activation module, which includes:
- mean hygroscopicity for each mode of an aerosol model
- critical supersaturation for each mode of an aerosol model
- maximum supersaturation for an entire aerosol model
- total number of particles actived in a system given an aerosol model 
- a number of helper functions
"""

using SpecialFunctions

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, molmass_water, LH_v0, cp_d
using CLIMAParameters.Atmos: K_therm, D_vapor

export alpha_sic
export gamma_sic
export mean_hygroscopicity
export coeff_of_curvature
export critical_supersaturation
export max_supersatuation
export total_N_activated

# Constants
TEMP = 273.15
P_SAT = 100000
P = 100000
R = molmass_water * R_v
G_DIFF = (((LH_v0/(K_therm*TEMP))*(((LH_v0/(*TEMP)R_v)-1))+((R_v*TEMP)/(P_SAT*D_vapor)))^(-1)
UPDFT_VELO = 5 



"""
alpha_sic(aero_mm)
    - am -- aerosol_model                      
    
    Returns coefficient relevant to other functions. Uses aerosol
    Molar mass
"""
function alpha_sic(am::aerosol_model, TEMP)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # Find weighted molar mass of mode
        n_comps = mode_i.n_components
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
function gamma_sic(am::aerosol_model, TEMP, P_SAT, PRESS)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # Find weighted molar mass of mode
        n_comps = mode_i.n_components
        numerator = sum(1:n_comps) do j
            mode_i.particle_density[j]*mode_i.molar_mass[j]
        end
        denominator = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_molar_mass = numerator/denominator
        exp1 = (R*TEMP)/(P_SAT*molmass_water)
        exp2 = (molmass_water*LH_v0^2)/(cp_v*PRESS*avg_molar_mass*TEMP)
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
function coeff_of_curvature(am::aerosol_model, TEMP)
        mode_i = am.modes[i]
        # take weighted average of activation times
        top = 2*surface_tension*molmass_water
        bottom = ρ_cloud_liq*R*TEMP
        return top/bottom
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
        n_comps = mode_i.n_components
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
        n_comps = mode_i.n_components
        numerator = sum(1:n_comps) do j
            mode_i.radius[j]*mode_i.particle_density[j]
        end 
        denominator = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_radius = numerator/denominator
        exp1 = 2 / (mh[i])^(.5)
        exp2 = (coeff_of_curve/(3*avg_radius))^(3/2)
        exp1*exp2
    end
end

"""
TO DO: DOCSTRING 
"""
function max_supersaturation(am::aerosol_model, TEMP, UPDFT_VELO, G_DIFF, PRESS, P_SAT)
    alpha = alpha_sic(am)
    gamma = gamma_sic(am, P_SAT, PRESS)
    A = coeff_of_curvature(am)
    Sm = critical_supersaturation(am)
    X = sum(1:length(am.modes)) do i 
        mode_i = am.modes[i]

        # weighted avgs of diff params:
        n_comps = mode_i.n_components
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
        eta = (((alpha[i]*UPDFT_VELO)/(G_DIFF))^(3/2))/(2*pi*ρ_cloud_liq*gamma[i]*total_particles)
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
function total_N_activated(am::aerosol_model, TEMP, UPDFT_VELO, G_DIFF, PRESS, P_SAT)
    smax = max_supersaturation(am, TEMP, UPDFT_VELO, G_DIFF, P_SAT, PRESS)
    sm = critical_supersaturation(am)
    TOTN =  sum(1:length(am.modes)) do i
        mode_i = am.modes[i]
        # weighted avgs of diff params:
        n_comps = mode_i.n_components
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

