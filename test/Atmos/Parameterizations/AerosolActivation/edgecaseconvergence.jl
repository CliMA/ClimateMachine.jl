# import Pkg; Pkg.add("Thermodynamics")
# import Pkg; Pkg.add("ClimateMachine")
import Statistics
using SpecialFunctions
using Statistics
using Test

using Thermodynamics

using ClimateMachine.AerosolModel: mode, aerosol_model
using ClimateMachine.AerosolActivation

using ClimateMachine.Microphysics: G_func

using CLIMAParameters
using CLIMAParameters: gas_constant
using CLIMAParameters.Planet: molmass_water, ρ_cloud_liq, grav, cp_d, molmass_dryair
using CLIMAParameters.Atmos.Microphysics


struct EarthParameterSet <: AbstractEarthParameterSet end
const EPS = EarthParameterSet
const param_set = EarthParameterSet()

# shevali's machine version: 
# include("/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Mode_creation.jl")
# include("/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/save_data.jl")

# DATA_PATH = "/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/saved_data.txt"

# isabella's machine version: 
include("/home/idularaz/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Mode_creation.jl")
include("/home/idularaz/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/save_data.jl")

DATA_PATH = "/home/idularaz/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/saved_data.txt"


# include("/home/idularaz/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Mode_creation.jl")
# prinln("length of AM", length(AM.N))
# CONSTANTS FOR TEST


T = 283.15     # air temperature
p = 100000.0   # air pressure
w = 5.0
Ts = collect(LinRange(100, 300, 100))
ws = collect(LinRange(.1, 100, 100))

N_frac_AM1_w = zeros(length(ws))
N_frac_AM8_w = zeros(length(ws))

N_frac_AM1_T = zeros(length(ws))
N_frac_AM8_T = zeros(length(ws))

for i in 1:length(ws)
    N_tot = AM_1.modes[1].N
    parts_act_1 = total_N_activated(param_set, AM_1, Ts[i], p, w)
    parts_act_8 = total_N_activated(param_set, AM_8, Ts[i], p, w)
    N_frac_AM1_T[i] = parts_act_1/N_tot
    N_frac_AM8_T[i] = parts_act_8/N_tot
end



println("This is the T range:")
println(Ts)
println("This is AM_1:")
println(N_frac_AM1_T)
println("This is AM_8:")
println(N_frac_AM8_T)
