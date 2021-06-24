#=
Full testing pipeline for the implementation of code in Abdul-Razzal and Ghan (2000).

Isabella Dula and Shevali Kadakia


Test classifications:
    --Verfication (VER): ensures that function output has consistent output, no matter inputted values (i.e.,
    verifies that the functions are doing what we want them to)
    --Validation (VAL): checks functions against model data in Abdul-Razzak and Ghan (2000) (i.e., validates 
    the functions outputs against published results)

Dimension (DIM):
    --Tests are done with multi-dimensional inputs. 
    --0: Only one mode and one component (e.g., coarse sea salt) 
    --1: Considering multiple modes over one component (e.g., accumulation mode and coarse mode sea salt)
    --2: Considering multiple modes with multiple components (e.g., accumulation and coarse mode for sea 
    salt and dust)

Modes and Components Considered
    --This testing pipeline uses data from Porter and Clarke (1997) to provide real-world inputs into 
    the functions
    --Modes: Accumulation (ACC) and coarse (COA)
    --Components: Sea Salt (SS) and Dust (DUS)
=#

# IMPORTS:
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

# Vars:
# General: 
T = 273.15 # temperature, assuming STP (K) 
N = 100000000 # aerosol particle density (particles/m^3)
V = 5.5 # updraft velocity (m/s)
R = 8.314462618 # gas constant ((kg m^2) / (s^2 K mol))
avocado = 6.02214076 × 10^23 # avogadro's number 
tau = 1              #
alpha = 1            #
diffusion = 1        # INCOMPLETE
gamma  = 1           # 

# Water Properties:
rho_w = 1000 # water density  (kg/m^3)
M_w = 0.01801528 # water molecular mass (kg/mol)
N_w = 0.022414 # water molar density (m^3/mol)

# Sea Salt accumulation and coarse modes: 
OC_SS = 0.9 # osmotic coefficient
M_SS = 0.058443 # sea salt molar mass; kg/mol
rho_SS = 2170 # sea salt density; kg/m^3
G_SS = 1 # Sea salt dissociation                           INCOMPLETE
epsilon = 1 # mass fraction                                INCOMPLETE

a_SS_ACC = 0.000000243 # mean particle radius (m)
a_SS_COA = 0.0000015 # mean particle radius (m)
sigma_SS_ACC = 0.0000014 # mean particle stdev (m)
sigma_SS_COA = 0.0000021 # mean particle stdev(m)







#------------------------------------------------------------------------------------------------------
# Test No. 1
# VER; All tests done; DIM 0; ACC, SS
#------------------------------------------------------------------------------------------------------

@testset "hygroscopicity_test" begin

    # assumptions: standard conditions
    # parameters
    osmotic_coefficient = [OC_SS] # no units
    temperature = [T] # K
    aerosol_density = [rho_SS] # kg/m^3
    aerosol_molecular_mass = [M_SS] # kg/mol
    aerosol_particle_density = [N] # 1/cm^3
    water_density = [rho_w] # kg/m^3
    water_molar_density = [N_w] # m^3/mol
    water_molecular_mass = [M_w] # kg/mol

    # hand calculated value
    updraft_velocity = [V]

    avogadro =  avocado

    h = (updraft_velocity .* osmotic_coefficient 
        .* (aerosol_particle_density .* 1./avogadro 
        .* aerosol_molecular_mass)./(1./water_molar_density
        .* 1./1000 .* water_molecular_mass) .* water_molecular_mass
        .* aerosol_particle_density) ./ (aerosol_molecular_mass
        .* water_density)

    @test hygroscopicity(
               osmotic_coefficient,
               temperature,
               aerosol_density,
               aerosol_molecular_mass,
               aerosol_particle_density
               water_density,
               water_molar_density,
               water_molecular_mass,
               ) ≈ h
end 

@testset "Mean Hygroscopicity" begin
    aerosol_component = [1, 2, 3]         
    aerosol_mode_number = [1, 2, 3, 4, 5]
    mass_mixing_ratio = [[1, 2, 3, 4, 5]
                         [0.1, 0.2, 0.3, 0.4, 0.5]
                         [0.01, 0.02, 0.03, 0.04, 0.05]]
    disassociation = [[1, 2, 3, 4, 5]
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
    add_top = 0
    add_bottom = 0
     water_molecular_mass = 0.01801528 # kg/mol
    water_density = 1000 # kg/m^3
    m_h = zeros(3)
    for i in 1:length(aerosol_mode_number)
        for j in 1:length(aerosowater_molecular_weightl_component)
            add_top = mass_mixing_ratio[i][j] 
                      * disassociation[i][j] 
                      * osmotic_coefficient[i][j]
                      * mass_fraction[i][j]
                      * aerosol_molecular_mass[i][j]
        end
        m_h[i] = water_molecular_mass * (add_top) / (add_bottom * water_density)
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