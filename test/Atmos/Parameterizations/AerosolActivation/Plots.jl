# import Pkg; Pkg.add("GR")
# using GR
# import Pkg; Pkg.add("Plots")
# import Pkg; Pkg.add("PyPlot")
# using PyPlot
using SpecialFunctions
using Plots
using Test

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

T = 274.0     # air temperature
p = 100000.0   # air pressure
w = .5        # vertical velocity



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

w_samples = [0.011504886, 0.012698618, 0.014016211, 0.015470515, 0.017075716, 0.01884747, 
             0.020803059, 0.022961558, 0.02534402, 0.027973683, 0.030876197, 0.034079871, 
             0.037615955, 0.041518939, 0.04582689, 0.05058183, 0.055830136, 0.061622999, 
             0.068016923, 0.075074272, 0.082863882, 0.091461732, 0.100951685, 0.111426302, 
             0.122987753, 0.135748805, 0.149833927, 0.165380503, 0.182540172, 0.201480307, 
             0.222385646, 0.245460096, 0.270928721, 0.299039938, 0.33006794, 0.364315366, 
             0.402116263, 0.443839331, 0.491779424, 0.540183119, 0.57865937, 0.635523737, 
             0.701464867, 0.774247963, 0.854582941, 0.944665327, 1.040405634, 1.169962553, 
             1.29135637, 1.425345854, 1.573237916, 1.736475068, 2.079128785,2.305522285, 
             2.554274059, 2.819302266, 3.111829461, 3.434708902, 3.791089901, 4.184448536, 
             4.655340819, 5.097843821]

PAPER_wN_fraction_activated = [0.137851736, 0.150252044, 0.163750129, 0.175821105, 0.18931919, 
                               0.200621722, 0.21401003, 0.226959227, 0.239469313, 
                               0.251979399, 0.264599263, 0.277328905, 0.289619436, 0.3022393, 
                               0.314529831, 0.327808361, 0.341306446, 0.355133864, 0.368522171, 
                               0.38136159, 0.396177007, 0.410882646, 0.426356729, 0.440952591, 
                               0.456975561, 0.473986531, 0.491546389, 0.508118248, 0.526336772, 
                               0.545323739, 0.562993375, 0.582968341, 0.602504196, 0.622369384, 
                               0.641356352, 0.662099761, 0.679988952, 0.698427031, 0.717975636, 
                               0.731911217, 0.751251743, 0.760724213, 0.777625405, 0.792221267, 
                               0.80626824, 0.821275532, 0.83653167, 0.85087999, 0.862694818, 
                               0.874436461, 0.885299883, 0.896492637, 0.913487868, 0.922574175, 
                               0.928629662, 0.933784646, 0.939817851, 0.947168388, 0.953640704, 
                               0.95967391, 0.966773894, 0.970810439]


osmotic_coeff_AG = 1.0 
molar_mass_AG = 132.0
dissoc_AG = 3.0
mass_mix_ratio_AG = 1.0
mass_frac_AG = 1.0
dry_radius_AG = 5e-8 
radius_stdev_AG = 2.0
density_AG = 1770.0 
TEMP_plot = 294.0

# plotting_mode = mode((N_samples[1], ), (osmotic_coeff_AG, ), (molar_mass_AG, ),
#                      (dissoc_AG,), (mass_frac_AG,), (mass_mix_ratio_AG,),
#                      (dry_radius_AG, ), (radius_stdev_AG, ), (density_AG,), 1)
# plotting_model = aerosol_model((plotting_mode,))
# println(length(plotting_model.modes))
# println(total_N_activated(param_set, plotting_model, T, p, w))
# for i in 1:length(N_samples)
#     datapointi = mode((N_samples[i],), (osmotic_coeff_AG,), 
#                       (molas_mass_AG, ), (dissoc_AG,)), 
#                       (,)
# end


EXP_fraction_particles_activated = zeros(length(N_samples))

for i in 1:length(N_samples)
    # initialize aerosol model
    modei = mode(dry_radius_AG, radius_stdev_AG, N_samples[i], 
                 (mass_mix_ratio_AG,), (mass_frac_AG,), (osmotic_coeff_AG,),
                 (molar_mass_AG,), (dissoc_AG, ), (density_AG, ), 1)
    modeli = aerosol_model((modei,))
    parts_act = total_N_activated(param_set, modeli, T, p, w)
    EXP_fraction_particles_activated[i] = parts_act/N_samples[i]  
end

EXP_fraction_particles_activated_vdependent = zeros(length(w_samples))


for i in 1:length(w_samples)
    # initialize aerosol model
    modei = mode(dry_radius_AG, radius_stdev_AG, 100, 
                 (mass_mix_ratio_AG,), (mass_frac_AG,), (osmotic_coeff_AG,),
                 (molar_mass_AG,), (dissoc_AG, ), (density_AG, ), 1)
    modeli = aerosol_model((modei,))
    parts_act = total_N_activated(param_set, modeli, T, p, w_samples[i])
    EXP_fraction_particles_activated_vdependent[i] = parts_act/100
end

println(EXP_fraction_particles_activated_vdependent)

# begin plotting
# pyplot()
#     plot(N_samples, PAPER_N_fraction_activated)
#     # plot!(N_samples, EXP_fraction_particles_activated)
# end
default(show = true)
# println(EXP_fraction_particles_activated)
plot(N_samples, PAPER_N_fraction_activated)
plot!(N_samples, EXP_fraction_particles_activated)



