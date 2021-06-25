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
    activation_cloud_droplet_number(param_set::APS, q, temperature, ρ)
TODO
"""

WATER_MOLECULAR_MASS = 0.01801528 # kg/mol
WATER_DENSITY = 1000 # kg/m^3
R = 8.31446261815324
AVOGADRO =  6.02214076 × 10^23
GRAVITY = 9.8
LATENT_HEAT_EVAPORATION = 2.26 * 10^-6
SPECIFIC_HEAT_AIR = 1000

function hygroscopicity_test(osmotic_coefficient,
               temperature,
               aerosol_density,
               aerosol_molecular_mass,
               aerosol_particle_density
               water_density,
               water_molar_density,
               water_molecular_mass, updraft_velocity)

    h = (updraft_velocity .* osmotic_coefficient 
        .* (aerosol_particle_density .* 1./avogadro 
        .* aerosol_molecular_weight)./(1./water_molar_density
        .* 1/1000 .* WATER_MOLECULAR_MASS) .* WATER_MOLECULAR_MASS
        .* aerosol_particle_density) ./ (aerosol_molecular_weight
        .* water_density)
    return h 
end 

function mean_hygroscopicity(aerosol_mode_number, 
                             aerosol_component, 
                             mass_mixing_ratio,
                             disassociation,
                             osmotic_coefficient,
                             mass_ratio,
                             molecular_weight)
    
    add_top = 0
    add_bottom = 0
    
    for j in 1:length(aerosol_component)
        add_top = mass_mixing_ratio[aerosol_mode_number][j] 
                  * disassociation[aerosol_mode_number][j] 
                  * osmotic_coefficient[aerosol_mode_number][j]
                  * mass_ratio[aerosol_mode_number][j]
                  * molecular_weight[aerosol_mode_number][j]
    end
    m_h = WATER_MOLECULAR_MASS * (add_top) / (add_bottom * WATER_DENSITY)
    return m_h
end

# Minimum supersaturation
function S_min(B_bar, A, particle_radius)
    S_m = (2./(B_bar).^.5).*(A./(3.*particle_radius)).^(3/2)
    return S_m
end

# Surface tension effects on Kohler equilibrium equation (s/(kg*m))
function A(activation_time, water_molecular_mass, water_density, R, temperature)
    A = (2.*activation_time.*water_molecular_mass)./(water_density.*R.*temperature)
    return A
end

function smallactpartdryrad(particle_radius, 
                            particle_radius_stdev,
                            activation_time, 
                            water_molecular_mass, 
                            water_density, 
                            R, 
                            temperature, 
                            B_bar,
                            alpha,
                            updraft_velocity, 
                            diffusion, 
                            aerosol_particle_density, 
                            gamma)

    A =  A(activation_time, water_molecular_mass, water_density, R, temperature)
    S_min =  S_min(B_bar, A, particle_radius)
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density, R, temperature, B_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)

    drsap = particle_radius.*((S_min)./(S_max)).^(2/3)
    return DRSAP
end



function maxsupersat(particle_radius, particle_radius_stdev, activation_time, water_molecular_mass, water_density, R, temperature, B_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)

    # Internal calculations: 
    f = 0.5.*exp.*(2.5.*(log.(particle_radius_stdev)).^2)
    g = 1 .+ 0.25 .* log.(particle_radius_stdev)        

    A = A(activation_time, water_molecular_mass, water_density, R, temperature)

    S_m = S_min(B_bar, A, particle_radius)
    
    zeta = ((2.*A)./3).*((alpha.*updraft_velocity)./diffusion).^(.5)
    eta = (((alpha.*updraft_velocity)./diffusion).^(3/2))./(2.*pi.*water_density.*gamma.*aerosol_particle_density)

    # Final calculation:
    mss = sum(1./(((1/S_m.^2).*((f.*(zeta/eta).^(3/2)).+(g.*((S_m.^2)./(eta.+3.*zeta))))).^.5))

    return mss


end


function total_N_Act(particle_radius, activation_time, water_molecular_mass, water_density, R, temperature, B_bar, particle_radius_stdev, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma, S_max)

    # Internal calculations:
    A = A(activation_time, water_molecular_mass, water_density, R, temperature)

    S_m = S_min(B_bar, A, particle_radius)
    
    u = (2.*log.(S_m./S_max))./(3.*(2.^.5).*log.(particle_radius_stdev))
    
    # Final Calculation: 
    
    totN = sum(aerosol_particle_density.*.5.*(1-erf.(u)))

    return totN

end

function alpha_sic(T, M_a)
    a = (GRAVITY * WATER_MOLECULAR_MASS * LATENT_HEAT_EVAPORATION) / (SPECIFIC_HEAT_AIR * R * T^2) - GRAVITY * M_a / (R * T)
    return a
end

function gamma_sic(T, M_a, rho_s)
    g = R * T / (rho_s * WATER_MOLECULAR_MASS) + WATER_MOLECULAR_MASS * SPECIFIC_HEAT_AIR ^ 2 / (SPECIFIC_HEAT_AIR * WATER_DENSITY * M_a * T)
    return g
end



end #module Activation.jl
