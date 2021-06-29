Pkg.add("SpecialFunctions")
# TODO: Map correct locations to run the test 
using Test
using ClimateMachine.Microphysics_0M
using ClimateMachine.Microphysics
using Thermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, R_d, molmass_ratio
using CLIMAParameters.Atmos.Microphysics
using CLIMAParameters.Atmos.Microphysics_0M

"""
Isabella Dula and Shevali Kadakia
    
This file has the complete set of tests to verify and validate the parameterization of
the model given in Abdul-Razzak and Ghan (2000). 

The structure of the pipeline is as follows:

--Test classifications:
    --Verfication (VER): ensures that function output has consistent output, no matter inputted values (i.e.,
    verifies that the functions are doing what we want them to)
    --Validation (VAL): checks functions against model data in Abdul-Razzak and Ghan (2000) (i.e., validates 
    the functions outputs against published results)

--Dimension (DIM):
    --Tests are done with multi-dimensional inputs: 
    --0: Only one mode and one component (e.g., coarse sea salt) 
    --1: Considering multiple modes over one component (e.g., accumulation mode and coarse mode sea salt)
    --2: Considering multiple modes with multiple components (e.g., accumulation and coarse mode for sea 
    salt and dust)

--Modes and Components Considered
        --This testing pipeline uses aerosol data from Porter and Clarke (1997) to provide real-world inputs into 
        the functions
        --Modes: Accumulation (ACC) and coarse (COA)
        --Components: Sea Salt (SS) and Dust (DUS)

"""
# CONSTANTS
WTR_ρ = 1000
WTR_MLR_ρ = 0.022414
WTR_MM = 0.01801528
R = 8.31446261815324
AVO = 6.02214076 × 10^23
G = 9.8
LAT_HEAT_EVP = 2.26 * 10^-6
SPC_HEAT_AIR = 1000
T = 273.1 # K (STP)
P = 100000 # Pa (N/m^2) (STP)

# TEST INPUTS 
P_saturation = 100000 # Pa                                  TODO
# --Sea Salt accumulation and coarse modes: 
OC_SS = 0.9 # osmotic coefficient
M_SS = 0.058443 # sea salt molar mass; kg/mol
rho_SS = 2170 # sea salt density; kg/m^3
nu_SS = 2 # Sea salt dissociatio                         TODO
epsilon_SS = 1 # mass fraction                              TODO
r_SS = 1 # mass mixing rati0                               TODO
a_SS_ACC = 0.000000243 # mean particle radius (m)
a_SS_COA = 0.0000015 # mean particle radius (m)
sigma_SS_ACC = 0.0000014 # mean particle stdev (m)
sigma_SS_COA = 0.0000021 # mean particle stdev(m)
DIF = 1 # diffusion                                         TODO

#------------------------------------------------------------------------------------------------------
# Test No. 1
# VER; All tests done; DIM 0; ACC, SS
#------------------------------------------------------------------------------------------------------

@testset "Mean_Hygroscopicity_1" begin
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS
    aero_ρ = rho_SS

    B_bar = zeros(size(mass_mx_rat, 2))
    for i in range(size(mass_mx_rat, 2))
        B_bar[i] = ((WTR_MM)/(WTR_ρ)).*
                   (sum(mass_mx_rat[i].*diss[i].*
                   epsilon_SS[i].*(1./aero_mm[i])  ) 
                   / sum(mass_mx_rat[i]./aero_ρ[i]))

    end
    
    for i in size(B_bar)
        @test mean_hygroscopicity(mass_mx_rat, diss, 
                                  osm_coeff, mass_frac,
                                  aero_mm, 
                                  aero_ρ)[i] 
                                  ≈ B_bar[i]
    end

end

@testset "max_supersat_1" begin
    # parameters inputted into function:
    part_radius = a_SS_ACC # particle mode radius (m)
    part_radius_stdev = sigma_SS_ACC # standard deviation of mode radius (m)
    act_time = tau # time of activation (s)                                                  
    updft_velo = V # Updraft velocity (m/s)
    diffusion = DIF # Diffusion of heat and moisture for particles 
    aero_part_ρ = N # Initial particle concentration (1/m^3)
    aero_ρ = rho_SS
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS

    gamma = gamma_sic(M_SS, P_saturation) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       

    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 
    
    f = 0.5 .* exp(2.5*(log.(part_radius_stdev)).^2) 
    g = 1 .+ 0.25 .* log.(part_radius_stdev) 

    A = A(act_time)
    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    zeta = ((2.*A)./(3)).*((alpha.*updft_velo)/(diffusion)).^
           (.5) 

    eta = (  ((alpha.*updft_velo)./(diffusion)).^(3/2)  ./    
          (2*pi.*WTR_ρ .*gamma.*aero_part_ρ)   )    
          

    
    # Final value for maximum supersaturation:
    MS = sum(((1)./(((S_m).^2) * (    f.*((zeta./eta).^(3/2))     
    .+    g.*(((S_m.^2)./(eta+3.*zeta)).^(3/4))    ) )))

    # Comaparing calculated MS value to function output: 
    @test maxsupersat(part_radius, part_radius_stdev, act_time, 
                      updft_velo, diffusion, 
                      aero_part_ρ, mass_mx_rat, 
                      diss, 
                      osm_coeff, mass_frac,
                      aero_mm, aero_ρ) 
                      ≈ MS
end
        
@testset "total_N_Act_1" begin
    # Input parameters
    part_radius = a_SS_ACC # particle mode radius (m)
    act_time = tau # time of activation (s)                                                  
    part_radius_stdev = sigma_SS_ACC # standard deviation of mode radius (m)
    updft_velo = V # Updraft velocity (m/s)
    diffusion = DIF # Diffusion of heat and moisture for particles 
    aero_part_ρ = N # Initial particle concentration (1/m^3)
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS
    aero_ρ = rho_SS

    gamma = gamma_sic(M_SS, P_saturation) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       


    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 

    A = A(act_time)

    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    S_max = maxsupersat(part_radius, part_radius_stdev, act_time, 
                        updft_velo, diffusion, 
                        aero_part_ρ, mass_mx_rat, 
                        diss, 
                        osm_coeff, mass_frac,
                        aero_mm, aero_ρ)

    u = ((2*log.(S_m/S_max))./(3.*(2.^.5).*
        log.(part_radius_stdev)))
    # Final Calculation: 
    totN = sum(aero_part_ρ.*.5.*(1-erf.(u)))

    # Compare results:
    @test total_N_Act(part_radius, act_time, 
                      part_radius_stdev, updft_velo, 
                      diffusion, aero_part_ρ) ≈ totN

end

#------------------------------------------------------------------------------------------------------
# Test No. 2
# VER; All tests done; DIM 0; COA, SS
#------------------------------------------------------------------------------------------------------

@testset "Mean_Hygroscopicity_2" begin
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS
    aero_ρ = rho_SS

    B_bar = zeros(size(mass_mx_rat, 2))
    for i in range(size(mass_mx_rat, 2))
        B_bar[i] = ((WTR_MM)/(WTR_ρ)).*
                   (sum(mass_mx_rat[i].*diss[i].*
                   epsilon_SS[i].*(1./aero_mm[i])  ) 
                   / sum(mass_mx_rat[i]./aero_ρ[i]))

    end
    
    for i in size(B_bar)
        @test mean_hygroscopicity(mass_mx_rat, diss, 
                                  osm_coeff, mass_frac,
                                  aero_mm, 
                                  aero_ρ)[i] 
                                  ≈ B_bar[i]
    end

end

@testset "max_supersat_2" begin
    # parameters inputted into function:
    part_radius = a_SS_COA # particle mode radius (m)
    part_radius_stdev = sigma_SS_COA # standard deviation of mode radius (m)
    act_time = tau # time of activation (s)                                                  
    updft_velo = V # Updraft velocity (m/s)
    diffusion = DIF # Diffusion of heat and moisture for particles 
    aero_part_ρ = N # Initial particle concentration (1/m^3)
    aero_ρ = rho_SS
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS

    gamma = gamma_sic(M_SS, P_saturation) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       

    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 
    
    f = 0.5 .* exp(2.5*(log.(part_radius_stdev)).^2) 
    g = 1 .+ 0.25 .* log.(part_radius_stdev) 

    A = A(act_time)
    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    zeta = ((2.*A)./(3)).*((alpha.*updft_velo)/(diffusion)).^
           (.5) 

    eta = (  ((alpha.*updft_velo)./(diffusion)).^(3/2)  ./    
          (2*pi.*WTR_ρ .*gamma.*aero_part_ρ)   )    
          

    
    # Final value for maximum supersaturation:
    MS = sum(((1)./(((S_m).^2) * (    f.*((zeta./eta).^(3/2))     
    .+    g.*(((S_m.^2)./(eta+3.*zeta)).^(3/4))    ) )))

    # Comaparing calculated MS value to function output: 
    @test maxsupersat(part_radius, part_radius_stdev, act_time, 
                      updft_velo, diffusion, 
                      aero_part_ρ, mass_mx_rat, 
                      diss, 
                      osm_coeff, mass_frac,
                      aero_mm, aero_ρ) 
                      ≈ MS
end
        
@testset "total_N_Act_2" begin
    # Input parameters
    part_radius = a_SS_COA # particle mode radius (m)
    act_time = tau # time of activation (s)                                                  
    part_radius_stdev = sigma_SS_COA # standard deviation of mode radius (m)
    updft_velo = V # Updraft velocity (m/s)
    diffusion = DIF # Diffusion of heat and moisture for particles 
    aero_part_ρ = N # Initial particle concentration (1/m^3)
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS
    aero_ρ = rho_SS

    gamma = gamma_sic(M_SS, P_saturation) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       


    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 

    A = A(act_time)

    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    S_max = maxsupersat(part_radius, part_radius_stdev, act_time, 
                        updft_velo, diffusion, 
                        aero_part_ρ, mass_mx_rat, 
                        diss, 
                        osm_coeff, mass_frac,
                        aero_mm, aero_ρ)

    u = ((2*log.(S_m/S_max))./(3.*(2.^.5).*
        log.(part_radius_stdev)))
    # Final Calculation: 
    totN = sum(aero_part_ρ.*.5.*(1-erf.(u)))

    # Compare results:
    @test total_N_Act(part_radius, act_time, 
                      part_radius_stdev, updft_velo, 
                      diffusion, aero_part_ρ) ≈ totN

end

#------------------------------------------------------------------------------------------------------
# Test No. 3
# VER; All tests done; DIM 1; ACC & COA, SS
#------------------------------------------------------------------------------------------------------

@testset "Mean_Hygroscopicity_3" begin
    mass_mx_rat = [r_SS, r_SS]
    diss = [nu_SS, nu_SS]
    osm_coeff = [OC_SS, OC_SS]
    mass_frac = [epsilon_SS, epsilon_SS]
    aero_mm = [M_SS, M_SS]
    aero_ρ = [rho_SS, rho_SS]

    B_bar = zeros(size(mass_mx_rat, 2))
    for i in range(size(mass_mx_rat, 2))
        B_bar[i] = ((WTR_MM)/(WTR_ρ)).*
                   (sum(mass_mx_rat[i].*diss[i].*
                   epsilon_SS[i].*(1./aero_mm[i])  ) 
                   / sum(mass_mx_rat[i]./aero_ρ[i]))

    end
    
    for i in size(B_bar)
        @test mean_hygroscopicity(mass_mx_rat, diss, 
                                  osm_coeff, mass_frac,
                                  aero_mm, 
                                  aero_ρ)[i] 
                                  ≈ B_bar[i]
    end

end

@testset "max_supersat_3" begin
    # parameters inputted into function:
    part_radius = [a_SS_ACC, a_SS_COA] 
    part_radius_stdev = [sigma_SS_ACC, sigma_SS_COA] 
    act_time = [tau, tau]                                                 
    updft_velo = [V, V] 
    diffusion = [DIF, DIF] # Diffusion of heat and moisture for particles 
    aero_part_ρ = [N, N] # Initial particle concentration (1/m^3)
    aero_ρ = [rho_SS, rho_SS]
    mass_mx_rat = [r_SS, r_SS]
    diss = [nu_SS, nu_SS]
    osm_coeff = [OC_SS, OC_SS]
    mass_frac = [epsilon_SS, epsilon_SS]
    aero_mm = [M_SS, M_SS]

    gamma = gamma_sic(M_SS, P_saturation) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       

    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 
    
    f = 0.5 .* exp(2.5*(log.(part_radius_stdev)).^2) 
    g = 1 .+ 0.25 .* log.(part_radius_stdev) 

    A = A(act_time)
    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    zeta = ((2.*A)./(3)).*((alpha.*updft_velo)/(diffusion)).^
           (.5) 

    eta = (  ((alpha.*updft_velo)./(diffusion)).^(3/2)  ./    
          (2*pi.*WTR_ρ .*gamma.*aero_part_ρ)   )    
          

    
    # Final value for maximum supersaturation:
    MS = sum(((1)./(((S_m).^2) * (    f.*((zeta./eta).^(3/2))     
    .+    g.*(((S_m.^2)./(eta+3.*zeta)).^(3/4))    ) )))

    # Comaparing calculated MS value to function output: 
    @test maxsupersat(part_radius, part_radius_stdev, act_time, 
                      updft_velo, diffusion, 
                      aero_part_ρ, mass_mx_rat, 
                      diss, 
                      osm_coeff, mass_frac,
                      aero_mm, aero_ρ) 
                      ≈ MS
end
        
@testset "total_N_Act_3" begin
    # Input parameters
    part_radius = [a_SS_ACC, a_SS_COA] 
    act_time = [tau, tau]                                                 
    part_radius_stdev = [sigma_SS_ACC, sigma_SS_COA] 
    updft_velo = [V, V]
    diffusion = [DIF, DIF] 
    aero_part_ρ = [N, N]
    mass_mx_rat = [r_SS, r_SS]
    diss = [nu_SS, nu_SS]
    osm_coeff = [OC_SS, OC_SS]
    mass_frac = [epsilon_SS, epsilon_SS]
    aero_mm = [M_SS, M_SS]
    aero_ρ = [rho_SS, rho_SS]

    gamma = gamma_sic(M_SS, P_saturation) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       


    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 

    A = A(act_time)

    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    S_max = maxsupersat(part_radius, part_radius_stdev, act_time, 
                        updft_velo, diffusion, 
                        aero_part_ρ, mass_mx_rat, 
                        diss, 
                        osm_coeff, mass_frac,
                        aero_mm, aero_ρ)

    u = ((2*log.(S_m/S_max))./(3.*(2.^.5).*
        log.(part_radius_stdev)))
    # Final Calculation: 
    totN = sum(aero_part_ρ.*.5.*(1-erf.(u)))

    # Compare results:
    @test total_N_Act(part_radius, act_time, 
                      part_radius_stdev, updft_velo, 
                      diffusion, aero_part_ρ) ≈ totN

end

#------------------------------------------------------------------------------------------------------
# Test No. 4
# VER; All tests done; DIM 2; ACC & COA, SS & DUS
#------------------------------------------------------------------------------------------------------

# TODO: DETERMINE PARAMETERS

#------------------------------------------------------------------------------------------------------
# Test No. 5
# VAL; ONLY total_N_Act_test; DIM 0; 0 CHECK

# ~~~|This test section checks that, if given an initial input of 0 particle density, the number of 
# ~~~|activated particles is 0. 

#------------------------------------------------------------------------------------------------------
@testset "total_N_Act_5" begin
    # Input parameters
    part_radius = a_SS_ACC 
    act_time = tau                                               
    part_radius_stdev = sigma_SS_ACC
    updft_velo = V 
    diffusion = DIF  
    aero_part_ρ = 0 # ZERO
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS
    aero_ρ = rho_SS


    gamma = gamma_sic(M_SS, P_saturation) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       

    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 

    A = A(act_time)

    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    S_max = maxsupersat(part_radius, part_radius_stdev, act_time, 
                        updft_velo, diffusion, 
                        aero_part_ρ, mass_mx_rat, 
                        diss, 
                        osm_coeff, mass_frac,
                        aero_mm, aero_ρ)

    u = ((2*log.(S_m/S_max))./(3.*(2.^.5).*
        log.(part_radius_stdev)))
    
    # Final Calculation: 
    totN = sum(aero_part_ρ.*.5.*(1-erf.(u)))

    # Compare results:
    @test total_N_Act(part_radius, act_time, 
                      part_radius_stdev, updft_velo, 
                      diffusion, aero_part_ρ) ≈ totN

end

#------------------------------------------------------------------------------------------------------
# Test No. 6
# VAL; ONLY total_N_Act_test; DIM 0; Paper validation

# ~~~|This test section checks compares outputs to results give in the 
# ~~~|Abdul-Razzal and Ghan paper to validate the accuracy of the 
# ~~~|functions implemented. 

#------------------------------------------------------------------------------------------------------

# TODO: Finalize; See runtests_actpipeline.jl for current version