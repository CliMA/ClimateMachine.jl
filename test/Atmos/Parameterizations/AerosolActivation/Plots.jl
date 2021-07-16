using SpecialFunctions

using Thermodynamics

using ClimateMachine.AerosolModel: mode, aerosol_model
using ClimateMachine.AerosolActivation

using CLIMAParameters
using CLIMAParameters: gas_constant
using CLIMAParameters.Planet: molmass_water, ρ_cloud_liq, grav, cp_d
using CLIMAParameters.Atmos.Microphysics

struct EarthParameterSet <: AbstractEarthParameterSet end
const EPS = EarthParameterSet
const param_set = EarthParameterSet()

T = 283.15     # air temperature
p = 100000.0   # air pressure
w = 5.0        # vertical velocity



# Create sample set (scraped from plots in paper)
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



osmotic_coeff_AG = 1.0 
molar_mass_AG = 132.0
dissoc_AG = 3.0
mass_mix_ratio_AG = 1.0
mass_frac_AG = .9
dry_radius_AG = 5e-8 
radius_stdev_AG = 2.0
density_AG = 1770.0 
TEMP_plot = 294.0

plotting_mode = mode((N_samples[1], ), (osmotic_coeff_AG, ), (molar_mass_AG, ),
                     (dissoc_AG,), (mass_frac_AG,), (mass_mix_ratio_AG,),
                     (dry_radius_AG, ), (radius_stdev_AG, ), (density_AG,), 1)
plotting_model = aerosol_model((plotting_mode,))
println(length(plotting_model.modes))
println(total_N_activated(param_set, plotting_model, T, p, w))
# for i in 1:length(N_samples)
#     datapointi = mode((N_samples[i],), (osmotic_coeff_AG,), 
#                       (molas_mass_AG, ), (dissoc_AG,)), 
#                       (,)
# end