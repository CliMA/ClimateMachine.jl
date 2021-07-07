using SpecialFunctions
using Test
using CLIMAParameters: gas_constant
using CLIMAParameters.Planet: molmass_water, ρ_cloud_liq, grav, T_freeze
using CLIMAParameters.Atmos.Microphysics

# include("/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Aerosol-activation/AerosolActivation-Shevali.jl")

# using ClimateMachine.Atmos.Parameterizations.CloudPhysics.Aerosol-activation.AerosolActivation-Shevali.jl: alpha_sic, gamma_sic, coeff_of_curvature, mean_hygroscopicity
# GET FROM CLIMA PARAMATERS
molar_mass_water = 18
density_water = 1000.0
R = 8.314462618

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
osmotic_coeff_seasalt = 0.9
molar_mass_seasalt = 0.058443
rho_seasalt = 2170.0
dissoc_seasalt = 2.0                        
mass_frac_seasalt = 1.0                           
mass_mix_ratio_seasalt = 1.0                      

# Sea Salt -- Accumulation mode
dry_radius_seasalt_accum = 0.000000243
radius_stdev_seasalt_accum = 0.0000014
particle_density_seasalt_accum = 100.0

# Sea Salt -- Coarse Mode
dry_radius_seasalt_coarse = 0.0000015
radius_stdev_seasalt_coarse = 0.0000021
particle_density_seasalt_coarse = 100.0

# TODO: Dust parameters (just copy and pasted seasalt values rn)
# Dust--universal parameters
osmotic_coeff_dust = 0.9
molar_mass_dust = 0.058443
rho_dust = 2170.0
dissoc_dust = 2.0                        
mass_frac_dust = 1.0                           
mass_mix_ratio_dust = 1.0                       

# Dust -- Accumulation mode
dry_radius_dust_accum = 0.000000243
radius_stdev_dust_accum = 0.0000014
particle_density_dust_accum = 100.0

# Dust -- Coarse Mode
dry_radius_dust_coarse = 0.0000015
radius_stdev_dust_coarse = 0.0000021
particle_density_dust_coarse = 100.0

# Abdul-Razzak and Ghan 

# 2. Create structs that parameters can be pass through
# individual aerosol mode struct
struct mode{T}
    particle_density::T
    osmotic_coeff::T
    molar_mass::T
    dissoc::T
    mass_frac::T
    mass_mix_ratio::T
    dry_radius::T
    radius_stdev::T
    aerosol_density::T
    n_components::Int64
end

# complete aerosol model struct
struct aerosol_model{T}
    modes::T
    N::Int 
    function aerosol_model(modes::T) where {T}
        return new{T}(modes, length(modes)) #modes new{T}
    end
end 

# 3. Populate structs to pass into functions/run calculations
# Test cases 1-3 (Just Sea Salt)
accum_mode_seasalt = mode((particle_density_seasalt_accum,), 
                          (osmotic_coeff_seasalt,), 
                          (molar_mass_seasalt,), 
                          (dissoc_seasalt,), 
                          (mass_frac_seasalt,), 
                          (mass_mix_ratio_seasalt,), 
                          (dry_radius_seasalt_accum,),
                          (rho_seasalt,),
                          (radius_stdev_seasalt_accum,), 
                          1)

coarse_mode_seasalt = mode((particle_density_seasalt_coarse,),
                           (osmotic_coeff_seasalt,), 
                           (molar_mass_seasalt,), 
                           (dissoc_seasalt,), 
                           (mass_frac_seasalt,), 
                           (mass_mix_ratio_seasalt,),
                           (dry_radius_seasalt_coarse,),
                           (rho_seasalt,),
                           (radius_stdev_seasalt_coarse,), 
                           1)

aerosolmodel_testcase1 = aerosol_model((accum_mode_seasalt,))
aerosolmodel_testcase2 = aerosol_model((coarse_mode_seasalt,))
aerosolmodel_testcase3 = aerosol_model((accum_mode_seasalt, coarse_mode_seasalt))

# Test cases 4-5 (Sea Salt and Dust)
accum_mode_seasalt_dust = mode((particle_density_seasalt_accum, 
                                particle_density_dust_accum),
                               (osmotic_coeff_seasalt, 
                                osmotic_coeff_dust), 
                               (molar_mass_seasalt, 
                                molar_mass_dust),
                               (dissoc_seasalt, 
                                dissoc_dust),
                               (mass_frac_seasalt, 
                                mass_frac_dust),
                               (mass_mix_ratio_seasalt, 
                                mass_mix_ratio_dust),
                               (dry_radius_seasalt_accum, 
                                dry_radius_dust_accum),
                               (rho_seasalt, 
                                rho_dust),
                               (radius_stdev_seasalt_accum, 
                                radius_stdev_dust_accum),
                                2)

coarse_mode_seasalt_dust = mode((particle_density_seasalt_coarse, 
                                 particle_density_dust_coarse),
                                (osmotic_coeff_seasalt, 
                                 osmotic_coeff_dust), 
                                (molar_mass_seasalt, 
                                 molar_mass_dust),
                                (dissoc_seasalt, 
                                 dissoc_dust),
                                (mass_frac_seasalt, 
                                 mass_frac_dust),
                                (mass_mix_ratio_seasalt, 
                                 mass_mix_ratio_dust),
                                (dry_radius_seasalt_coarse, 
                                 dry_radius_dust_coarse),
                                (rho_seasalt, 
                                 rho_dust),
                                (radius_stdev_seasalt_coarse, 
                                 radius_stdev_dust_coarse),
                                 2)

aerosolmodel_testcase4 = aerosol_model((accum_mode_seasalt_dust,))
aerosolmodel_testcase5 = aerosol_model((accum_mode_seasalt_dust,
                                        coarse_mode_seasalt_dust))

function total_mass(m::mode)
    num_of_comp = m.n_components
    total_mass = sum(num_of_comp) do j
            m.particle_density[j]
    end
    return total_mass
end

function tp_mean_hygroscopicity(am::aerosol_model)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        total_mass_value = total_mass(mode_i)
        num_of_comp = mode_i.n_components # mode_i.n_components
        numerator = sum(num_of_comp) do j
            mode_i.particle_density[j]/total_mass_value * mode_i.osmotic_coeff[j] * mode_i.mass_mix_ratio[j] * mode_i.dissoc[j] * mode_i.mass_frac[j] * 1/mode_i.molar_mass[j]
        end
        denominator = sum(num_of_comp) do j
            mode_i.particle_density[j]/total_mass_value * mode_i.mass_mix_ratio[j] / mode_i.aerosol_density[j]
        end
        (numerator/denominator) * (molar_mass_water/density_water)
    end
end

# questions about temp, 
# need to fill equations: , alpha --> 1.0, eta() --> 2.0
# Key:
# surface tension == A
# surface_tension_effects(zeta) --> 3.0

function tp_max_super_sat(am::aerosol_model, 
                            temp::Float64, 
                            updraft_velocity::Float64, 
                            diffusion::Float64, 
                            activation_time::Float64)
    mean_hygro = tp_mean_hygroscopicity(am)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        total_mass_value = total_mass(mode_i)
        num_of_comp = mode_i.n_components
        a = sum(num_of_comp) do j
            f = 0.5 * exp(2.5 * log(mode_i.radius_stdev[j])^2)
            g = 1 + 0.25 * log(mode_i.radius_stdev[j])
            surface_tension = 2 * activation_time * molar_mass_water / (density_water * R * temp)
            surface_tension_effects = 2 * surface_tension / 3 * (1.0 * updraft_velocity / diffusion)^(1/2)
            supersat = 2/sqrt(mean_hygro[i]) * (surface_tension / (3 * mode_i.dry_radius[j])) ^ (3/2)
            mode_i.particle_density[j]/total_mass_value * (1/(supersat ^ 2) * (f * (surface_tension_effects/2.0) ^(3/2) + g * (supersat ^ 2)/ (2.0 + 3 * surface_tension_effects)^(3/4)))
        end
        a ^ (1/2)
    end
end

function tp_coeff_of_curve(temp::Float64, activation_time::Float64)
    value = 2 * activation_time * density_water / (density_water * R * temp)
    return value
end

function tp_critical_supersaturation(am::aerosol_model, 
                                       temp::Float64, 
                                       activation_time::Float64)
    mean_hygro = tp_mean_hygroscopicity(am)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        total_mass_value = total_mass(mode_i)
        a = sum(num_of_comp) do j
            mode_i.particle_density[j]/total_mass_value * 2 / sqrt(mean_hygro[i]) * (tp_coeff_of_curve(temp, activation_time) / (3 * mode_i.dry_radius[j]) ^ (3/2))
        end
        a
    end
end

function tp_total_n_act(am::aerosol_model, 
                          temp::Float64, 
                          updraft_velocity::Float64, 
                          diffusion::Float64, 
                          activation_time::Float64)
    critical_supersaturation = tp_critical_supersaturation(am::aerosol_model, temp::Float64, activation_time::Float64)
    max_supersat = tp_max_super_sat(am, temp, updraft_velocity, diffusion, activation_time)
    values = ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        total_mass_value = total_mass(mode_i)
        a = sum(num_of_comp) do j
            sigma = mode_i.radius_stdev[j]
            u_bottom = 2 * log(critical_supersaturation[i] / max_supersat[i])
            u_top = 3 * sqrt(2) * log(sigma)
            u = u_top / u_bottom
            mode_i.particle_density[j]/total_mass_value * mode_i.particle_density[j] * 1/2 * (1 - erf(u))
        end
    end
    summation = 0.0
    for i in range(1, length=length(values))
        summation += values[i]
    end
    return summation
end

# @testset "mean_hygroscopicity" begin
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase1) ≈ (1,0)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase2) ≈ (1,0)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase3) ≈ (1,0)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase4) ≈ (1,0)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase5) ≈ (1,0)
# end
# @testset "max_super_sat" begin
#     @test tp_max_super_sat(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) ≈ (1,0)
#     @test tp_max_super_sat(aerosolmodel_testcase2, 2.0, 3.0, 4.0, 1.0) ≈ (1,0)
#     @test tp_max_super_sat(aerosolmodel_testcase3, 2.0, 3.0, 4.0, 1.0) ≈ (1,0)
#     @test tp_max_super_sat(aerosolmodel_testcase4, 2.0, 3.0, 4.0, 1.0) ≈ (1,0)
#     @test tp_max_super_sat(aerosolmodel_testcase5, 2.0, 3.0, 4.0, 1.0) ≈ (1,0)
# end

# @testset "total_n_act" begin
#     @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) ≈ 1
#     @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) ≈ 1
#     @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) ≈ 1
#     @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) ≈ 1
#     @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) ≈ 1 
# end
# println("total_n_act") 
# println("LOOK HERE")
# println(tp_mean_hygroscopicity(aerosolmodel_testcase1))
# println(tp_mean_hygroscopicity(aerosolmodel_testcase2))
# println(tp_mean_hygroscopicity(aerosolmodel_testcase3))
# println(tp_mean_hygroscopicity(aerosolmodel_testcase4))
# println(tp_mean_hygroscopicity(aerosolmodel_testcase5))

# println("test max super sat")
# println(tp_max_super_sat(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0))
# println(tp_max_super_sat(aerosolmodel_testcase2, 2.0, 3.0, 4.0, 1.0))
# println(tp_max_super_sat(aerosolmodel_testcase3, 2.0, 3.0, 4.0, 1.0))
# println(tp_max_super_sat(aerosolmodel_testcase4, 2.0, 3.0, 4.0, 1.0))
# println(tp_max_super_sat(aerosolmodel_testcase5, 2.0, 3.0, 4.0, 1.0))

# println("test total n activated")
# println(tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0))
# println(tp_total_n_act(aerosolmodel_testcase2, 2.0, 3.0, 4.0, 1.0))
# println(tp_total_n_act(aerosolmodel_testcase3, 2.0, 3.0, 4.0, 1.0))
# println(tp_total_n_act(aerosolmodel_testcase4, 2.0, 3.0, 4.0, 1.0))
# println(tp_total_n_act(aerosolmodel_testcase5, 2.0, 3.0, 4.0, 1.0))



function alpha_sic(am::aerosol_model)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # Find weighted molar mass of mode
        n_comps = length(mode_i.particle_density)
        println("This is number of comps:",  1:n_comps, "mode:", i)
        numerator = sum(1:n_comps) do j
            mode_i.particle_density[j] * mode_i.molar_mass[j]
        end
        denominator = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_molar_mass = numerator/denominator
        exp1 = (g * Mw * L) / (Cp * R * T^2)
        exp2 = (g * avg_molar_mass)/(R * T)
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
        numerator = sum(1:n_comps) do j
            mode_i.particle_density[j] * mode_i.molar_mass[j]
        end
        denominator = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_molar_mass = numerator/denominator
        exp1 = (R * T)/(P_sat * Mw)
        exp2 = (Mw * L^2)/(Cp * P * avg_molar_mass * T)
        exp1+exp2 
    end
end       
"""
mean_hygroscopicity(am::aerosol_model)

DO DOCSTRING
"""
function mean_hygroscopicity(am::aerosol_model)
    return map(am.modes) do mode_i 
        n_comps = length(mode_i.particle_density)
        top = sum(1:n_comps) do j
            mode_i.mass_mix_ratio[j] * mode_i.dissoc[j] * 
            mode_i.osmotic_coeff[j] * mode_i.mass_frac[j] * 
            (1 / mode_i.molar_mass[j])
        end
        bottom = sum(1:n_comps) do j 
            mode_i.mass_mix_ratio[j] / mode_i.particle_density[j]
        end  
        Mw / density_water * top / bottom
    end 
end

function coeff_of_curve(am::aerosol_model)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # take weighted average of activation times 
        n_comps = length(mode_i.particle_density)
        numerator = sum(1:n_comps) do j
            mode_i.activation_time[j] * mode_i.particle_density[j]
        end 
        denominator = sum(1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_activation_time = numerator / denominator 
        top = 2 * avg_activation_time * Mw
        bottom = density_water * R * T
        top/bottom

    end

end

function critical_supersaturation(am::aerosol_model)
    coeff_of_curvature = coeff_of_curve(am)
    mh = mean_hygroscopicity(am)
    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # weighted average of mode radius
         n_comps = length(mode_i.particle_density)
        numerator = sum( 1:n_comps) do j
            mode_i.radius[j] * mode_i.particle_density[j]
        end 
        denominator = sum( 1:n_comps) do j
            mode_i.particle_density[j]
        end
        avg_radius = numerator/denominator
        exp1 = 2 / (mh[i])^(.5)
        exp2 = (coeff_of_curvature[i]/3 * avg_radius)^(3/2)
        exp1 * exp2
    end
end

function max_supersatuation(am::aerosol_model)
    alpha = alpha_sic(am)
    gamma = gamma_sic(am, P_saturation)
    A = coeff_of_curve(am)
    Sm = critical_supersaturation(am)
    return sum(1:length(am.modes)) do i 

        mode_i = am.modes[i]

        # weighted avgs of diff params:
        n_comps = length(mode_i.particle_density)
        # radius_stdev
        num = sum( 1:n_comps) do j 
            mode_i.particle_density[j]  *  mode_i.radius_stdev[j]
        end
        den = sum( 1:n_comps) do j 
            mode_i.particle_density[j]
        end 
        avg_radius_stdev = num/den 
        
        total_particles = sum(1:n_comps) do j 
            mode_i.particle_density[j]
        end
        f = 0.5  *  exp(2.5  *  (log(avg_radius_stdev))^2 )
        g = 1 + 0.25  *  log(avg_radius_stdev) 

        zeta = (2 * A[i] * (1/3))  *  ((alpha[i] * updft_velo)/diff)^(.5)
        eta = (((alpha[i]*updft_velo)/(diff))^(3/2))/(2*pi*density_water*gamma[i]*total_particles)

        exp1 = 1/(Sm[i])^2
        exp2 = f*(zeta/eta)^(3/2)
        exp3 = g*((Sm[i]^2)/(eta+3*zeta))^(3/4)

        (exp1*(exp2+exp3))^(1/2)
    end

end


function total_N_activated(am::aerosol_model)
    smax = max_supersatuation(am)
    sm = critical_supersaturation(am)
    return sum(1:length(am.modes)) do i
        mode_i = am.modes[i]
        # weighted avgs of diff params:
        n_comps = length(mode_i.particle_density)
        # radius_stdev
        num = sum( 1:n_comps) do j 
            mode_i.particle_density[j]  *  mode_i.radius_stdev[j]
        end
        den = sum( 1:n_comps) do j 
            mode_i.particle_density[j]
        end 
        avg_radius_stdev = num/den 
        
        total_particles = sum(1:n_comps) do j 
            mode_i.particle_density[j]
        end

        utop = 2*log(sm[i]/smax)
        ubottom = 3*(2^.5)*log(avg_radius_stdev)
        u = utop/ubottom
        total_particles*(1/2)*(1-erf(u))
    end 
end

# println("LOOK HERE")
# println(mean_hygroscopicity(aerosolmodel_testcase1))
# max_supersatuation(aerosolmodel_testcase1)
# total_N_activated(aerosolmodel_testcase1)

# @testset "mean_hygroscopicity" begin
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase1) == mean_hygroscopicity(aerosolmodel_testcase1)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase2) == mean_hygroscopicity(aerosolmodel_testcase2)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase3) == mean_hygroscopicity(aerosolmodel_testcase3)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase4) == mean_hygroscopicity(aerosolmodel_testcase4)
#     @test tp_mean_hygroscopicity(aerosolmodel_testcase5) == mean_hygroscopicity(aerosolmodel_testcase5)
# end

# @testset "max_super_sat" begin
#     @test tp_max_super_sat(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) == max_supersatuation(aerosolmodel_testcase1)
#     @test tp_max_super_sat(aerosolmodel_testcase2, 2.0, 3.0, 4.0, 1.0) == max_supersatuation(aerosolmodel_testcase2)
#     @test tp_max_super_sat(aerosolmodel_testcase3, 2.0, 3.0, 4.0, 1.0) == max_supersatuation(aerosolmodel_testcase3)
#     @test tp_max_super_sat(aerosolmodel_testcase4, 2.0, 3.0, 4.0, 1.0) == max_supersatuation(aerosolmodel_testcase4)
#     @test tp_max_super_sat(aerosolmodel_testcase5, 2.0, 3.0, 4.0, 1.0) == max_supersatuation(aerosolmodel_testcase5)
# end

@testset "total_n_act" begin
    @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) = total_N_activated(aerosolmodel_testcase1)
    @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) = total_N_activated(aerosolmodel_testcase2)
    @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) = total_N_activated(aerosolmodel_testcase3)
    @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) = total_N_activated(aerosolmodel_testcase4)
    @test tp_total_n_act(aerosolmodel_testcase1, 2.0, 3.0, 4.0, 1.0) = total_N_activated(aerosolmodel_testcase5)
end