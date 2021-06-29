

Pkg.add("SpecialFunctions")
module Activation

using Thermodynamics
using CLIMAParameters
using CLIMAParameters.Atmos.Microphysics_0M
const APS = AbstractParameterSet
export activation_cloud_droplet_number
# TODO: Check over these ^^^^

"""
    Isabella Dula and Shevali Kadakia
    
    This file has all the functions for the parametrization, according to the model
    given in Abdul-Razzak and Ghan (2000).
    
"""
#--------------------------------------------------------------------#
# TODO: 
# --Write function to determine mass fraction of soluble 
#   material (epsilon)
#--Determine P_saturation origins (do we get this from model?)
#--------------------------------------------------------------------#
#CONSTANTS 

WTR_MM = 0.01801528 # kg/mol
WTR_MLR_ρ = 0.022414 # m^3/mol TODO (used for epsilon)
WTR_ρ = 1000 # kg/m^3
R = 8.31446261815324 # (kg m^2) / (s^2 K mol)
AVOGADRO =  6.02214076 × 10^23 # particles/mole (used for epsilon)
G = 9.81 # m/s^2
LAT_HEAT_EVP = 2.26 * 10^-6  # J/kg
SPC_HEAT_AIR = 1000
T = 273.1 # K (STP)
P = 100000 # Pa (N/m^2) (STP)

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
    A = (2*activation_time.*WTR_MM)./(WTR_ρ.*R.*T)
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
        B_bar[i] = ((WTR_MM)/(WTR_ρ))*
                   ((sum(r.*nu.*phi.*epsilon.* (1./M_a)))/(sum(r./rho_a)))
    end
    return B_bar
end

#--------------------------------------------------------------------#
# Maximum Supersaturation
function maxsupersat(particle_radius, particle_radius_stdev, activation_time, 
                     updraft_velocity, diffusion, 
                     aerosol_particle_density, mass_mixing_ratio, 
                     disassociation, 
                     osmotic_coefficient, mass_ratio,
                     aerosol_molecular_mass, aerosol_density)

    # Internal calculations: 
    alpha = alpha_sic(aerosol_molecular_mass)
    f = 0.5.*exp.(2.5*(log.(particle_radius_stdev)).^2)
    g = 1 .+ 0.25 * log.(particle_radius_stdev)        

    A = A(activation_time)

    S_m = S_m(activation_time, particle_radius, mass_mixing_ratio, 
              disassociation, osmotic_coefficient, mass_ratio,
              aerosol_molecular_mass, aerosol_density)
    
    gamma = gamma_sic(aerosol_molecular_mass, P_saturation)

    zeta = ((2.*A)./3).*((alpha.*updraft_velocity)./diffusion).^(.5)
    eta = (((alpha.*updraft_velocity)./diffusion).^(3/2))./(2.*pi.*WTR_ρ.*gamma.*aerosol_particle_density)

    # Final calculation:
    mss = sum(1./(((1/S_m.^2).*((f.*(zeta/eta).^(3/2)).+
             (g.*((S_m.^2)./(eta.+3.*zeta)).^(3/4)))).^.5))

    return mss
end

#--------------------------------------------------------------------#
# Total Number of Activated Particles
function total_N_Act(particle_radius, activation_time, 
                     particle_radius_stdev, updraft_velocity, 
                     diffusion, aerosol_particle_density)

    # Internal calculations:
    S_m = S_m(activation_time, particle_radius, mass_mixing_ratio, 
              disassociation, osmotic_coefficient, mass_ratio,
              aerosol_molecular_mass, aerosol_density)

    S_max = maxsupersat(particle_radius, particle_radius_stdev, activation_time,
                        updraft_velocity, diffusion, 
                        aerosol_particle_density, mass_mixing_ratio, 
                        disassociation, 
                        osmotic_coefficient, mass_ratio,
                        aerosol_molecular_mass, aerosol_density)

    u = (2.*log.(S_m./S_max))./(3.*(2.^.5).*log.(particle_radius_stdev))
    
    # Final Calculation: 
    totN = sum(aerosol_particle_density.*.5.*(1-erf.(u)))

    return totN
end

#--------------------------------------------------------------------#
# Size Invariant Coefficients 
function alpha_sic(aerosol_molecular_mass)
    a = ((G * WTR_MM * LAT_HEAT_EVP) 
        / (SPC_HEAT_AIR * R * T^2)) .- 
        ((G * aerosol_molecular_mass ) ./ (R * T))
    return a
end

function gamma_sic(aerosol_molecular_mass, P_saturation)
    g = (R * T)./ (P_saturation * WTR_MM) .+ 
        (WTR_MM * LAT_HEAT_EVP) ^ 2 ./ 
        (SPC_HEAT_AIR * P * aerosol_molecular_mass * T)
    return g
end

end #module Activation.jl
