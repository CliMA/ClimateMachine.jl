"""
Aerosol activation module, which includes:
- mean hygroscopicity for each mode of an aerosol model
- critical supersaturation for each mode of an aerosol model
- maximum supersaturation for an entire aerosol model
- total number of particles actived in a system given an aerosol model 
- a number of helper functions
"""
module AerosolActivation

using SpecialFunctions

using Thermodynamics

using ClimateMachine.AerosolModel
using ClimateMachine.Microphysics: G_func

using CLIMAParameters
using CLIMAParameters: gas_constant
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, molmass_water, molmass_dryair, LH_v0, cp_d
using CLIMAParameters.Atmos.Microphysics: K_therm, D_vapor

const APS = AbstractParameterSet

export mean_hygroscopicity
export critical_supersaturation
export max_supersaturation
export total_N_activated

# export alpha_sic
# export gamma_sic
# export mean_hygroscopicity
# export coeff_of_curvature
# export critical_supersaturation
# export max_supersatuation
# export total_N_activated


# """
# alpha_sic(aero_mm)
#     - am -- aerosol_model                      
    
# """
# function alpha_sic(param_set::APS, am::aerosol_model, T::Float64)

#     _gas_constant = gas_constant()
#     _grav = grav(param_set)
#     _molmass_water = molmass_water(param_set)
#     _LH_v0 = LH_v0(param_set)
#     _cp_v = cp_v(param_set)

#     return ntuple(length(am.modes)) do i 
#         mode_i = am.modes[i]
#         # Find weighted molar mass of mode
#         n_comps = mode_i.n_components
#         numerator = sum(1:n_comps) do j
#             mode_i.particle_density[j]*mode_i.molar_mass[j]
#         end
#         denominator = sum(1:n_comps) do j
#             mode_i.particle_density[j]
#         end
#         avg_molar_mass = numerator/denominator
#         exp1 = (_grav*_molmass_water*_LH_v0) / (_cp_v*_gas_constant*T^2)
#         exp2 = (_grav*avg_molar_mass)/(_gas_constant*T)
#         exp1-exp2
#     end
# end

# """
# gamma_sic(aero_mm, T, P_SAT, PRESS)
#     - am -- aerosol_model  
#     - T -- temperature (K)
#     - PRESS -- pressure (Pa)   
#     - P_SAT -- saturation pressure (Pa)
                 
    
#     Returns coefficient relevant to other functions. 
# """
# function gamma_sic(param_set::APS, am::aerosol_model, T::Float64, PRESS::Float64, P_SAT::Float64)

#     _gas_constant = gas_constant()
#     _molmass_water = molmass_water(param_set)
#     _LH_v0 = LH_v0(param_set)
#     _cp_v = cp_v(param_set)

#     return ntuple(length(am.modes)) do i 
#         mode_i = am.modes[i]
#         # Find weighted molar mass of mode
#         n_comps = mode_i.n_components
#         numerator = sum(1:n_comps) do j
#             mode_i.particle_density[j]*mode_i.molar_mass[j]
#         end
#         denominator = sum(1:n_comps) do j
#             mode_i.particle_density[j]
#         end
#         avg_molar_mass = numerator/denominator
#         exp1 = (_gas_constant*T)/(P_SAT*_molmass_water)
#         exp2 = (_molmass_water*_LH_v0^2)/(_cp_v*PRESS*avg_molar_mass*T)
#         exp1+exp2 
#     end
# end

"""
coeff_of_curvature(am::aerosol_model)
    - am -- aerosol_model
    - T -- temperature (K)
    
    Returns coeff_of_curvature (coefficient of the curvature effect); key 
    input into other functions. 
"""
function coeff_of_curvature(param_set::APS, T::FT) where {FT <: Real}

    _molmass_water = molmass_water(param_set)
    _gas_constant = gas_constant()
    _ρ_cloud_liq = ρ_cloud_liq(param_set)

    surface_tension = 0.072 #TODO - take it from CLIMAParameters

    return 2 * surface_tension * _molmass_water / _ρ_cloud_liq / _gas_constant / T
end

"""
mean_hygroscopicity(am::aerosol_model)

    - am -- aerosol model

    Returns the mean hygroscopicty along each mode of an inputted aerosol model. 
"""
function mean_hygroscopicity(param_set::APS, am::aerosol_model) where {FT <: Real}

    _molmass_water = molmass_water(param_set)
    _ρ_cloud_liq = ρ_cloud_liq(param_set)

    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        n_comps = mode_i.n_components
        top = sum(1:n_comps) do j
            mode_i.mass_mix_ratio[j]*mode_i.dissoc[j]*
            mode_i.osmotic_coeff[j]*mode_i.mass_frac[j]*
            (1/mode_i.molar_mass[j])
        end
        bottom = sum(1:n_comps) do j 
            mode_i.mass_mix_ratio[j]/mode_i.aerosol_density[j]
        end 
        _molmass_water/_ρ_cloud_liq *(top/bottom)
    end 
end

"""
critical_supersaturation(am::aerosol_model)

    - am -- aerosol model 

    Returns the critical superation for each mode of an aerosol model. 
"""
function critical_supersaturation(param_set::APS, am::aerosol_model, 
                                  T::FT) where {FT <: Real}

    coeff_of_curve = coeff_of_curvature(param_set, T)
    mh = mean_hygroscopicity(param_set, am)

    return ntuple(length(am.modes)) do i 
        mode_i = am.modes[i]
        # weighted average of mode dry_radius
        n_comps = mode_i.n_components
        numerator = sum(1:n_comps) do j
            mode_i.dry_radius[j]*mode_i.particle_density[j]
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
max_supsaturation(am::aerosol_model, T, w, PRESS, P_SAT)

    - am -- aerosol model
    - T -- temperature (K)
    - w -- updraft velocity (m/s)
    - PRESS -- pressure (Pa)

    Returns the maximum supersaturation for an entire aerosol model. 
"""
function max_supersaturation(param_set::APS, am::aerosol_model, 
                             T::FT, p::FT, w::FT) where {FT <: Real}

    _grav::FT = grav(param_set)
    _molmass_water::FT = molmass_water(param_set)
    _molmass_dryair::FT = molmass_dryair(param_set)
    _gas_constant::FT = gas_constant()
    _R_v::FT = R_v(param_set)
    _cp_d::FT = cp_d(param_set)
    _K_therm::FT = K_therm(param_set)
    _D_vapor::FT = D_vapor(param_set)
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)

    L::FT = latent_heat_vapor(param_set, T)
    p_vs::FT = saturation_vapor_pressure(param_set, T, Liquid())
    _G::FT = G_func(param_set, T, Liquid())

    # eq 11, 12 in Razzak et al 1998
    α::FT = _grav * _molmass_water * L / _cp_d / _gas_constant / T^2 -
            _grav * _molmass_dryair / _gas_constant / T
    γ::FT = _gas_constant * T / p_vs / _molmass_water +
            _molmass_water * L^2 / _cp_d / p / _molmass_dryair / T

    A = coeff_of_curvature(param_set, T)
    Sm = critical_supersaturation(param_set, am, T)
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
        zeta = (2 * A * (1/3))  *  ((α * w)/_G)^(.5)
        eta = (((α*w)/(_G))^(3/2))/(2*pi*_ρ_cloud_liq*γ*total_particles)
        exp1 = 1/(Sm[i])^2
        exp2 = f*(zeta/eta)^(3/2)
        exp3 = g*((Sm[i]^2)/(eta+3*zeta))^(3/4)

        exp1*(exp2+exp3)
    end
    return 1/((X)^(1/2))

end

"""
total_N_activated(am::aerosol_model, T, w, P)

    -- am - aerosol model
    -- T - temperature (K)
    -- w - updraft velocity (m/s)
    -- PRESS - pressure (Pa)
    
"""
function total_N_activated(param_set::APS, 
                           am::aerosol_model, 
                           T::Float64, 
                           p::Float64, 
                           w::Float64) where {FT <: Real}
                           
    smax = max_supersaturation(param_set, am, T, p, w)
    sm = critical_supersaturation(param_set, am, T) # TODO how to specify a type here?

    return sum(1:length(am.modes)) do i        
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
end

end # module AerosolActivation.jl