Pkg.add("SpecialFunctions")
using Test
using ClimateMachine.Microphysics_0M
using ClimateMachine.Microphysics
using Thermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, R_d, molmass_ratio
using CLIMAParameters.Atmos.Microphysics
using CLIMAParameters.Atmos.Microphysics_0M

struct LiquidParameterSet <: AbstractLiquidParameterSet end
struct IceParameterSet <: AbstractIceParameterSet end
struct RainParameterSet <: AbstractRainParameterSet end
struct SnowParameterSet <: AbstractSnowParameterSet end

struct MicropysicsParameterSet{L, I, R, S} <: AbstractMicrophysicsParameterSet
    liquid::L
    ice::I
    rain::R
    snow::S
end

struct EarthParamSet{M} <: AbstractEarthParameterSet
    microphys_param_set::M
end

microphys_param_set = MicropysicsParameterSet(
    LiquidParameterSet(),
    IceParameterSet(),
    RainParameterSet(),
    SnowParameterSet(),
)

prs = EarthParamSet(microphys_param_set)
liq_prs = prs.microphys_param_set.liquid
ice_prs = prs.microphys_param_set.ice
rai_prs = prs.microphys_param_set.rain
sno_prs = prs.microphys_param_set.snow


WATER_DENSITY = 1000 # kg/m^3
WATER_MOLAR_DENSITY = 0.022414 # m^3/mol
WATER_MOLECULAR_MASS = 0.01801528 # kg/mol
R = 8.31446261815324
AVOGADRO = 6.02214076 × 10^23
GRAVITY = 9.8
LATENT_HEAT_EVAPORATION = 2.26 * 10^-6
SPECIFIC_HEAT_AIR = 1000
#-----------------------------------------------------------------------------------------------#

@testset "hygroscopicity_test" begin

    # assumptions: standard conditions
    # parameters
    osmotic_coefficient = 3 # no units
    temperature = 298 # K
    aerosol_density = 1770 # kg/m^3
    aerosol_molecular_mass = 0.132 # kg/mol
    aerosol_particle_density = 100000000# 1/cm^3

    # hand calculated value
    updraft_velocity = 3

    h = (updraft_velocity * osmotic_coefficient 
        * (aerosol_particle_density * 1/avogadro 
        * aerosol_molecular_mass)/(1/water_molar_density
        * 1/1000 * water_molecular_mass) * water_molecular_mass
        * aerosol_particle_density) / (aerosol_molecular_mass
        * water_density)

    @test hygroscopicity(
               osmotic_coefficient,
               temperature,
               aerosol_density,
               aerosol_molecular_mass,
               aerosol_particle_density
               ) ≈ h
end 

@testset "Mean Hygroscopicity" begin
    aerosol_component = [1, 2, 3]        
    aerosol_mode_number = [1, 2, 3, 4, 5]
    mass_mixing_ratio = [[1, 2, 3, 4, 5]
                         [0.1, 0.2, 0.3, 0.4, 0.5]
                         [0.01, 0.02, 0.03, 0.04, 0.05]]
    dissociation = [[1, 2, 3, 4, 5]
                      [0.1, 0.2, 0.3, 0.4, 0.5]
                      [0.01, 0.02, 0.03, 0.04, 0.05]]
    osmotic_coefficient = [[1, 2, 3, 4, 5]
           [0.1, 0.2, 0.3, 0.4, 0.5]
           [0.01, 0.02, 0.03, 0.04, 0.05]]
    mass_fraction = [[1, 2, 3, 4, 5]
               [0.1, 0.2, 0.3, 0.4, 0.5]
               [0.01, 0.02, 0.03, 0.04, 0.05]]
    aerosol_molecular_mass = [[1, 2, 3, 4, 5]
                        [0.1, 0.2, 0.3, 0.4, 0.5]
                        [0.01, 0.02, 0.03, 0.04, 0.05]]
    aerosol_density = [[1, 2, 3, 4, 5]
                       [0.1, 0.2, 0.3, 0.4, 0.5]
                       [0.01, 0.02, 0.03, 0.04, 0.05]]
    add_top = 0
    add_bottom = 0

    m_h = zeros(3)
    top_values = mass_mixing_ratio .* dissociation .* osmotic_coefficient .* mass_fraction .*aerosol_molecular_mass
    top_values *= WATER_MOLECULAR_MASS
    bottom_values = mass_mixing_ratio ./ aerosol_density
    bottom_values *= WATER_DENSITY

    for i in 1:length(aerosol_component)
        m_h[i] = sum(top_values[aerosol_component]) * WATER_MOLECULAR_MASS / (sum(bottom_values))
        end
        m_h[i] = add_top / bottom_values
    return m_h
    end

    @test mean_hygroscopicity(aerosol_mode_number[1], 
                              aerosol_component), 
                              mass_mixing_ratio,
                              disassociation,
                              osmotic_coefficient,
                              mass_fraction,
                              aerosol_molecular_mass,
                              ) ≈ m_h[1]
    @test mean_hygroscopicity(aerosol_component[2], 
                              aerosol_mode_number), 
                              mass_mixing_ratio,
                              disassociation,
                              osmotic_coefficient,
                              mass_fraction,
                              aerosol_molecular_mass,
                              ) ≈ m_h[2]
    @test mean_hygroscopicity(aerosol_component[3], 
                              aerosol_mode_number), 
                              mass_mixing_ratio,
                              disassociation,
                              osmotic_coefficient,
                              mass_fraction,
                              aerosol_molecular_mass,
                              ) ≈ m_h[3]
end


@testset "max_supersat_test" begin

    # parameters inputted into function:
    particle_radius = [5*(10^(-8))] # particle mode radius (m)
    particle_radius_stdev = [2] # standard deviation of mode radius (m)
    activation_time = [1] # time of activation (s)                                                  
    water_molecular_mass = [0.01801528] # Molecular weight of water (kg/mol)
    water_density  = [1000] # Density of water (kg/m^3)
    R = [8.31446261815324] # Gas constant (kg*m^2/s^2*K*mol)
    temperature = [273.15] # Temperature (K)
    alpha = [1] # Coefficient in superaturation balance equation       
    updraft_velocity = [1] # Updraft velocity (m/s)
    diffusion = [1] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [100000000] # Initial particle concentration (1/m^3)
    gamma = [1] # coefficient 

    # Internal calculations:
    B_bar = mean_hygrosopicity() # calculated in earlier function    ------ INCOMPLETE-------
    f = 0.5 .* exp(2.5*(log.(particle_radius_stdev)).^2) # function of particle_radius_stdev (check units)
    g = 1 .+ 0.25 .* log.(particle_radius_stdev) # function of particle_radius_stdev (log(m))
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_i_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    zeta = ((2.*A)./(3)).*((alpha.*updraft_velocity)/(diffusion)).^(.5) # dependent parameter 
    eta = (  ((alpha.*updraft_velocity)./(diffusion)).^(3/2)  ./    (2*pi.*water_density .*gamma.*aerosol_particle_density)   )    # dependent parameter

    
    # Final value for maximum supersaturation:
    MS = sum(((1)./(((S_min).^2) * (    f_i.*((zeta./eta).^(3/2))     +    g_i.*(((S_min.^2)./(eta+3.*zeta)).^(3/4))    ) )))

    # Comaparing calculated MS value to function output: 
    @test maxsupersat(particle_radius, particle_radius_stdev, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = MS
end

@testset "smallactpartdryrad_test" begin
    # Parameter inputs:
    particle_radius = [5*(10^(-8))] # particle mode radius (m)
    activation_time = [1] # time of activation (s)                                                  
    water_molecular_mass = [0.01801528] # Molecular weight of water (kg/mol)
    water_density  = [1000] # Density of water (kg/m^3)
    R = [8.31446261815324] # Gas constant (kg)
    temperature = [273.15] # Temperature (K)
    particle_radius_stdev = [2] # standard deviation of mode radius (m)
    alpha = [1] # Coefficient in superaturation balance equation       
    updraft_velocity = [1] # Updraft velocity (m/s)
    diffusion = [1] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [100000000] # Initial particle concentration (1/m^3)
    gamma = [1] # coefficient 
    
    # Internal calculations:
    B_bar = mean_hygrosopicity() # calculated in earlier function    ------ INCOMPLETE-------
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_i_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)
    
    # Final calculation:
    DRSAP = particle_radius.*((S_mi)./(S_max)).^(2/3)
    
    # Running test:
    @test smallactpartdryrad(particle_radius, particle_radius_stdev, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = DRSAP
    
    end
        
    Pkg.add("SpecialFunctions")

@testset "total_N_Act_test" begin
    # Input parameters
    particle_radius = [5*(10^(-8))] # particle mode radius (m)
    activation_time = [1] # time of activation (s)                                                  
    water_molecular_mass = [0.01801528] # Molecular weight of water (kg/mol)
    water_density  = [1000] # Density of water (kg/m^3)
    R = [8.31446261815324] # Gas constant (kg)
    temperature = [273.15] # Temperature (K)
    particle_radius_stdev = [2] # standard deviation of mode radius (m)
    alpha = [1] # Coefficient in superaturation balance equation       
    updraft_velocity = [1] # Updraft velocity (m/s)
    diffusion = [1] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [100000000] # Initial particle concentration (1/m^3)
    gamma = [1] # coefficient 
    
    # Internal calculations:
    B_bar = mean_hygrosopicity() # calculated in earlier function    ------ INCOMPLETE-------
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)

    u = ((2*log.(S_min/S_max))./(3.*(2.^.5).*log.(particle_radius_stdev)))
    # Final Calculation: 
    totN = sum(aerosol_particle_density.*.5.*(1-erf.(u)))

    # Compare results:
    @test total_N_Act(particle_radius, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, particle_radius_stdev, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = totN

end

@testset alpha()
    t = 273.15
    M_a = 0.058443
    a = (GRAVITY * WATER_MOLECULAR_MASS * LATENT_HEAT_EVAPORATION) 
        / (SPECIFIC_HEAT_AIR * R * t^2) - GRAVITY * M_a / (R * t)
    
    @test alpha_sic(T, M_a) ≈ a
end

@testset gamma()
    t = 273.15
    M_a = 0.058443
    rho_s = 2170
    g = R * t / (rho_s * WATER_MOLECULAR_MASS) + WATER_MOLECULAR_MASS 
        * SPECIFIC_HEAT_AIR ^ 2 / (SPECIFIC_HEAT_AIR * WATER_DENSITY * M_a * T)
    
    @test gamma_sic(t, M_a, rho_s) ≈ g
end

end 