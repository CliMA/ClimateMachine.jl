import Pkg; Pkg.add("Plots")
using SpecialFunctions
using Plots

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

PAPER_N_fraction_activated = [0.757982763, 0.68872084, 0.675045956, 0.663318114, 
                              0.653537315, 0.636609951, 0.629921514, 0.623805736, 
                              0.61837715, 0.612834031, 0.608207168, 0.6046414, 
                              0.601358612, 0.5979916, 0.593937397, 0.590799449, 
                              0.587088841, 0.583149169, 0.580240285, 0.576987805, 
                              0.57465158, 0.571742695, 0.56940647, 0.566383054, 
                              0.564485505, 0.564617756, 0.558572656, 0.556580027, 
                              0.554358334, 0.552365705, 0.55002948, 0.547922319, 
                              0.545127966, 0.543249869, 0.540455517, 0.538577419, 
                              0.536241194, 0.534134033, 0.532714064, 0.530492371, 
                              0.528156146, 0.525476325, 0.524514483, 0.521032939, 
                              0.51961297, 0.516245958, 0.51494052, 0.511573508, 
                              0.509924474, 0.508848101, 0.505366557, 0.504290183, 
                              0.501724894, 0.499617733, 0.497625104, 0.494487156, 
                              0.493868909, 0.490387365, 0.48919646, 0.486516639, 
                              0.483836818, 0.482073253, 0.478820773, 0.477744399, 
                              0.475866302, 0.473530077, 0.479362567]

osmotic_coeff_AG = 1.0 
molar_mass_AG = 132.0
dissoc_AG = 3.0
mass_mix_ratio_AG = 1.0
mass_frac_AG = 1.0
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


EXP_fraction_particles_activated = []

for i in 1:length(N_samples)
    # initialize aerosol model
    modei = mode(dry_radius_AG, radius_stdev_AG, N_samples[i], 
                 (mass_mix_ratio_AG,), (mass_frac_AG,), (osmotic_coeff_AG,)
                 (molas_mass_AG,), (dissoc_AG, ), (density_AG, ), 1)
    modeli = aerosol_model((modei,))
    parts_act = total_N_activated(param_set, modeli, T, p, w)
    EXP_fraction_particles_activated[i] = parts_act/N_samples[i]  
end

# begin plotting
pyplot()
    plot(N_samples, PAPER_N_fraction_activated)
    plot!(N_samples, EXP_fraction_particles_activated)
end