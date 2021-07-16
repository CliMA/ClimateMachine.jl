using SpecialFunctions
# universal parameters: 
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

N_samples = [15.65732013, 205.38735, 277.3022744, 349.2332106, 421.1801587, 
             565.095718, 637.0680968, 709.0451848, 781.0279242, 853.0097216, 
             924.9990541, 997.5008006, 1068.997498, 1140.997191, 1212.991233, 
             1284.99281, 1356.989677, 1428.984661, 1500.988122, 1572.988757, 
             1644.996927, 1717.000387, 1789.008557, 1861.011076, 1935.751163, 
             2011.162737, 2077.028992, 2149.039988, 2221.0491, 2293.060095, 
             2365.068265, 2437.078319, 2509.082721, 2581.094659, 2653.099061, 
             2725.110999, 2797.119169, 2869.129222, 2941.144927, 3013.154039, 
             3085.162209, 3157.167553, 3229.187026, 3301.185777, 3373.201482, 
             3445.201175, 3517.217822, 3589.217515, 3661.231336, 3733.249867, 
             3805.248618, 3877.267148, 3949.273435, 4021.283488, 4093.294484, 
             4165.296061, 4237.318359, 4309.31711, 4381.334699, 4453.340043, 
             4525.345387, 4597.358266, 4669.358901, 4741.377432, 4813.389369, 
             4885.397539, 4954.006328]

osmotic_coeff_AG = 1 
molar_mass_AG = 132
dissoc_AG = 3
mass_mix_ratio_AG = 1
for i in 1:length(N_samples)
    datapointi = mode((N_samples[i],), (osmotic_coeff_AG,), 
                      (molas_mass_AG, ), (dissoc_AG,)), 
                      (,)
end
