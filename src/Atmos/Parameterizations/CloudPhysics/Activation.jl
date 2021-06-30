using SpecialFunctions
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

WTR_MM = 0.01801528 # kg/mol
WTR_MLR_ρ = 0.022414 # m^3/mol TODO (used for epsilon)
WTR_ρ = 1000 # kg/m^3
R = 8.31446261815324 # (kg m^2) / (s^2 K mol)
AVOGADRO =  6.02214076 × 10^23 # particles/mole (used for epsilon)
G = 9.81 # m/s^2
LAT_HEAT_EVP = 2.26 * 10^-6  # J/kg
SPC_HEAT_AIR = 1000
TEMP = 273.1 # K (STP)
P = 100000 # Pa (N/m^2) (STP)

function S_m(act_time, part_radius, mass_mx_rat, diss, osm_coeff, mass_frac,
            aero_mm, aero_ρ)

    a = A(act_time)
    b_bar = mean_hygroscopicity(mass_mx_rat, diss, osm_coeff, mass_frac,
            aero_mm, aero_ρ)
    S_m = (2./(b_bar).^.5).*(a./(3.*part_radius)).^(3/2)
    return S_m

end

function A(act_time)

    a = (2 * act_time .* WTR_MM) ./ (WTR_ρ .* R .* TEMP)
    return a

end
# input one mode at a time, returns the summation of all the compositions.
function mean_hygroscopicity(mass_mx_rat, diss, osm_coeff, mass_frac, aero_mm, 
                             aero_ρ)
    b_bar = zeros(size(mass_mx_rat)[2])
    for i in range(size(mass_mx_rat)[2])
        b_bar[i] = ((WTR_MM)/(WTR_ρ)).*
                   (sum(mass_mx_rat[i].*diss[i].*
                   mass_frac[i].*(1./aero_mm[i])  ) 
                   / sum(mass_mx_rat[i]./aero_ρ[i]))
    return b_bar

end

function maxsupersat(part_radius, part_radius_stdev, act_time, updft_velo, diff, 
                     aero_part_ρ, mass_mx_rat, diss, osm_coeff, mass_frac,
                     aero_mm, aero_ρ)

    # Internal calculations: 
    a = alpha_sic(aero_mm)
    f = 0.5.*exp.(2.5*(log.(part_radius_stdev)).^2)
    g = 1 .+ 0.25 * log.(part_radius_stdev)        
    a = A(act_time)
    s_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)
    gamma = gamma_sic(aero_mm, P_sat)
    zeta = ((2.*a)./3).*((a.*updft_velo)./diff).^(.5)
    eta = (((a.*updft_velo)./diff).^(3/2))./(2.*pi.*WTR_ρ.*gamma.*aero_part_ρ)

    # Final calculation:
    mss = sum(1./(((1/s_m .^ 2).*((f.*(zeta/eta).^(3/2)).+
             (g.*((s_m.^2)./(eta.+3.*zeta)).^(3/4)))).^.5))

    return mss

end

function total_N_Act(part_radius, act_time, part_radius_stdev, updft_velo, diff, 
                     aero_part_ρ)

    # Internal calculations:
    s_m = S_m(act_time, part_radius, mass_mx_rat, diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)
    s_max = maxsupersat(part_radius, part_radius_stdev, act_time, updft_velo, 
                        diff, aero_part_ρ, mass_mx_rat, diss, osm_coeff, 
                        mass_frac, aero_mm, aero_ρ)
    u = (2.*log.(s_m./s_max))./(3.*(2.^.5).*log.(part_radius_stdev))

    # Final Calculation: 
    totN = sum(aero_part_ρ.*.5.*(1-erf.(u)))
    return totN

end

function alpha_sic(aero_mm)

    a = ((G * WTR_MM * LAT_HEAT_EVP) / (SPC_HEAT_AIR * R * T^2)) .- 
        ((G * aero_mm) ./ (R * T))
    return a

end

function gamma_sic(aero_mm, P_sat)

    g = (R * T)./ (P_sat * WTR_MM) .+ (WTR_MM * LAT_HEAT_EVP) ^ 2 ./ 
        (SPC_HEAT_AIR * P * aero_mm * TEMP)
    return g

end

end #module Activation.jl
