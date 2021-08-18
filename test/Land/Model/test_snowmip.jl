using MPI
using OrderedCollections
using StaticArrays
using Test
using Statistics
using DelimitedFiles
using Plots
using CLIMAParameters.Planet: cp_i, LH_f0, T_0
using Dierckx


using CLIMAParameters
using CLIMAParameters.Planet: cp_i
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SnowModel
using ClimateMachine.Land.SnowModel.SnowModelParameterizations
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state
using ClimateMachine.ArtifactWrappers


ClimateMachine.init()
FT = Float64
data = readdlm("/Users/katherinedeck/Downloads/SNOWMIP2_Alptal_openarea_input4CliMA_v3.csv",',')
start = 7161
indexrange = start:start+350 # no rain or snow in this window, no melting either
Qsurf = data[:, data[1,:] .== "Qsurf (W/m2)"][2:end, :][indexrange] # J/m^2/s
Qsurf = -Qsurf # sign convention is opposite as ours
G = data[:, data[1,:] .== data[1,:][12]][2:end, :][indexrange]# J/m^2/s
G = -G #their convention is downward positive
dates = data[:, data[1,:] .== "Day"][2:end, :][indexrange]


swe = FT(mean((data[:, data[1,:] .== "SWE (m)"][2:end,:][indexrange])))
z_snow = FT(mean((data[:, data[1,:] .== "depsn(m)"][2:end,:][indexrange])))#*0.8
ρ_snow = 1e3*swe ./ z_snow
κ_air = FT(0.023)
κ_ice = FT(2.29)
κ_snow = FT(κ_air + (7.75*1e-5 *ρ_snow + 1.105e-6*ρ_snow^2)*(κ_ice-κ_air)) 


# actual observed value
Tsurf = FT.(data[:, data[1,:] .== "surtsn (K)"][2:end,:][indexrange])
Tave = FT.(data[:, data[1,:] .== "SnowTemp (K)"][2:end,:][indexrange])
t = FT.(0:1800:length(Qsurf)*1800-1)
soil_water_model = PrescribedWaterModel()
soil_heat_model = PrescribedTemperatureModel()
soil_param_functions = nothing

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)

snow_parameters = SnowParameters{FT,FT,FT,FT}(κ_snow,ρ_snow,z_snow)
Qsurf_spline = Spline1D(t, Qsurf)
function Q_surf(t::FT, Q::Spline1D) where {FT}
    return Q(t)
end
Qbott_spline = Spline1D(t, G)
function Q_bott(t::FT, Q::Spline1D) where {FT}
    return Q(t)
end
    

forcing = PrescribedForcing(FT;Q_surf = (t) -> eltype(t)(Q_surf(t,Qsurf_spline)),Q_bottom = (t) -> eltype(t)(Q_bott(t,Qbott_spline)))
#Initial conditions
Tave0 = Tave[1]
ρe_int0 = volumetric_internal_energy(Tave0, snow_parameters.ρ_snow, 0.0, param_set)
ic = (aux) -> eltype(aux)(ρe_int0)
eq_snow_model = SingleLayerSnowModel{typeof(snow_parameters), typeof(forcing),typeof(ic)}(
    snow_parameters,
    forcing,
    ic
)


function init_land_model!(land, state, aux, localgeo, time)
    state.snow.ρe_int = land.snow.initial_ρe_int(aux)
end

sources = (FluxDivergence{FT}(),)

m = LandModel(
    param_set,
    m_soil;
    snow = eq_snow_model,
    source = sources,
    init_state_prognostic = init_land_model!,
)

N_poly = 1
nelem_vert = 1

# Specify the domain boundaries
zmax = FT(1)
zmin = FT(0)

driver_config = ClimateMachine.SingleStackConfiguration(
    "LandModel",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m;
    zmin = zmin,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
)

t0 = FT(0)
timeend = FT((indexrange[end]-start)*1800)
dt = FT(60*30)

solver_config = ClimateMachine.SolverConfiguration(
    t0,
    timeend,
    driver_config,
    ode_dt = dt,
)
n_outputs = length(indexrange);

every_x_simulation_time = ceil(Int, (timeend-t0) / n_outputs);

# Create a place to store this output.
state_types = (Prognostic(),)
dons_arr = Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
time_data = FT[0] # store time data

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

# # Run the integration
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));
z = 0:0.01:FT(snow_parameters.z_snow)
N = length(dons_arr)
ρe_int = [dons_arr[k]["snow.ρe_int"][1] for k in 1:N]
clima_Tave = snow_temperature.(ρe_int, Ref(0.0), Ref(ρ_snow),Ref(param_set))
qb = Q_bott.(time_data, Ref(Qbott_spline))
qs = Q_surf.(time_data, Ref(Qsurf_spline))
coeffs = compute_profile_coefficients.(qs,qb, Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
t_profs = get_temperature_profile.(qs,qb, Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
        
tsurf_pw = [coeffs[k][1] for k in 1:N]
tbottom = [coeffs[k][2] for k in 1:N]
th = [coeffs[k][4] for k in 1:N]
#We can try different definitions of G and see how that affects bulk I
# simplest approx to integral is cumsum(div flux) *dt
divflux_orig = -(Qsurf.-G)./z_snow
predicted_orig = snow_temperature.(cumsum(divflux_orig*1800) .+ρe_int[1], Ref(0.0), Ref(ρ_snow), Ref(param_set))

plot1 = plot(time_data, clima_Tave, label = "Our Model")
plot!(time_data, Tave, label = "Est from data")
plot!(time_data, predicted_orig, label = "Our Model, predicted")
plot!(title = "bulk T")
plot!(legend = :bottomright)
plot!(xticks = ([time_data[1],time_data[250],time_data[end]], [dates[1], dates[250], dates[end]]))


plot2 = plot(indexrange, tsurf_pw, label = "Our Model", title = "Tsurf")
plot!(indexrange, Tsurf, label = "Observed ")
plot!(xlabel = "date", ylabel = "T surf (K)", legend = :topleft)
plot!(xticks = ([1,250,indexrange[end]], [dates[1], dates[250], dates[end]]))

plot(plot1,plot2)
