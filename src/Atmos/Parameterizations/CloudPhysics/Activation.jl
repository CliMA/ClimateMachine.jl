"""
    Activation
TODO
"""

Pkg.add("SpecialFunctions")
module Activation

using Thermodynamics
using CLIMAParameters
using CLIMAParameters.Atmos.Microphysics_0M
const APS = AbstractParameterSet

export activation_cloud_droplet_number

"""
    Isabella Dula and Shevali Kadakia
    
    This file has all the functions for the parametrization, according to the model
    given in Abdul-Razzak and Ghan (2000).
    
"""
#--------------------------------------------------------------------#
#CONSTANTS 

WATER_MOLECULAR_MASS = 0.01801528 # kg/mol
WATER_MOLAR_DENSITY = 0.022414 # m^3/mol
WATER_DENSITY = 1000 # kg/m^3
R = 8.31446261815324 # ((kg m^2) / (s^2 K mol))
AVOGADRO =  6.02214076 × 10^23 # particles/mole
GRAVITY = 9.81 # m/s^2
LATENT_HEAT_EVAPORATION = 2.26 * 10^-6  # J/kg
SPECIFIC_HEAT_AIR = 1000
T = 273.1 # K

#--------------------------------------------------------------------#
# Critical supersaturation
function S_m(activation_time, particle_radius, mass_mixing_ratio, 
            disassociation, osmotic_coefficient, mass_ratio,
            aerosol_molecular_mass, aerosol_density)
    A = A(activation_time)
    B_bar = mean_hygroscopicity(mass_mixing_ratio, disassociation, 
            osmotic_coefficient, mass_ratio,
            aerosol_molecular_mass, aerosol_density)
    S_m = (2./(B_bar).^.5).*(A./(3.*particle_radius)).^(3/2)
    return S_m
end

#--------------------------------------------------------------------#
# Surface tension effects on Kohler equilibrium equation (s/(kg*m))
function A(activation_time)
    A = (2*activation_time.*WATER_MOLECULAR_MASS)./(WATER_DENSITY.*R.*T)
    return A
end
#--------------------------------------------------------------------#
# Mean Hygroscopicity
function mean_hygroscopicity(mass_mixing_ratio, disassociation, 
                             osmotic_coefficient, mass_ratio,
                             aerosol_molecular_mass, aerosol_density)
    
    B_bar = zeros(size(mass_mixing_ratio, 2))
    for i in range(size(mass_mixing_ratio, 2))
        r = mass_mixing_ratio[i]
        nu = disassociation[i]
        phi = osmotic_coefficient[i]
        epsilon = mass_ratio[i]
        M_a = aerosol_molecular_mass[i]
        rho_a = aerosol_density[i]
        B_bar[i] = ((WATER_MOLECULAR_MASS)/(WATER_DENSITY))*
                   ((sum(r.*nu.*phi.*epsilon.* (1./M_a)))/(sum(r./rho_a)))
    end
    return B_bar
end


function maxsupersat(particle_radius, particle_radius_stdev, activation_time, 
                     temperature, alpha, updraft_velocity, diffusion, 
                     aerosol_particle_density, gamma, mass_mixing_ratio, 
                     disassociation, 
                     osmotic_coefficient, mass_ratio,
                     aerosol_molecular_mass, aerosol_density)

    # Internal calculations: 
    f = 0.5.*exp.(2.5*(log.(particle_radius_stdev)).^2)
    g = 1 .+ 0.25 * log.(particle_radius_stdev)        

    A = A(activation_time)

    S_m = S_m(activation_time, particle_radius, mass_mixing_ratio, 
              disassociation, osmotic_coefficient, mass_ratio,
              aerosol_molecular_mass, aerosol_density)

    zeta = ((2.*A)./3).*((alpha.*updraft_velocity)./diffusion).^(.5)
    eta = (((alpha.*updraft_velocity)./diffusion).^(3/2))./(2.*pi.*WATER_DENSITY.*gamma.*aerosol_particle_density)

    # Final calculation:
    mss = sum(1./(((1/S_m.^2).*((f.*(zeta/eta).^(3/2)).+
             (g.*((S_m.^2)./(eta.+3.*zeta)).^(3/4)))).^.5))

    return mss
end


function total_N_Act(particle_radius, activation_time, 
                     particle_radius_stdev, alpha, updraft_velocity, 
                     diffusion, aerosol_particle_density, gamma)

    # Internal calculations:
    S_m = S_m(activation_time, particle_radius, mass_mixing_ratio, 
              disassociation, osmotic_coefficient, mass_ratio,
              aerosol_molecular_mass, aerosol_density)

    S_max = maxsupersat(particle_radius, particle_radius_stdev, activation_time, 
                        temperature, alpha, updraft_velocity, diffusion, 
                        aerosol_particle_density, gamma, mass_mixing_ratio, 
                        disassociation, 
                        osmotic_coefficient, mass_ratio,
                        aerosol_molecular_mass, aerosol_density)

    u = (2.*log.(S_m./S_max))./(3.*(2.^.5).*log.(particle_radius_stdev))
    
    # Final Calculation: 
    totN = sum(aerosol_particle_density.*.5.*(1-erf.(u)))

    return totN
end

function alpha_sic(T, M_a)
    a = (GRAVITY * WATER_MOLECULAR_MASS * LATENT_HEAT_EVAPORATION) 
        / (SPECIFIC_HEAT_AIR * R * T^2) - GRAVITY * M_a / (R * T)
    return a
end

function gamma_sic(T, M_a, rho_s)
    g = R * T / (rho_s * WATER_MOLECULAR_MASS) + WATER_MOLECULAR_MASS 
        * SPECIFIC_HEAT_AIR ^ 2 / (SPECIFIC_HEAT_AIR * WATER_DENSITY * M_a * T)
    return g
end

end #module Activation.jl
