# how to set G? Is there a single set of ρ,κ_snow that leads to fitting Tsurf, Tbottom, and Tbulk?

# Also, as we vary density, z_snow changes. we should have a better/more robust estimate
# of bulk T? Maybe it doesnt matter so much...

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
#data = readdlm("/Users/shuangma/Downloads/USU_data.csv",',')
data = readdlm("/Users/katherinedeck/Downloads/USU_data.csv",',')
start = 8
indexrange = start:500 # no rain or snow in this window, no melting either
#Qsurf = data[:, data[1,:] .== "Qsurf (kJ/m2/hr)"][2:end, :][indexrange] ./ 3600 .*1000 # per second
# this comes closest to matching data; treat as truth?
Qsurf = data[:, data[1,:] .== "Flux(MFR)"][2:end, :][indexrange] ./ 3600 .*1000 # per second
#Qsurf = data[:, data[1,:] .== "Flux(FR)"][2:end, :][indexrange] ./ 3600 .*1000 # per second
#Qsurf = data[:, data[1,:] .== "Flux(Tave)"][2:end, :][indexrange] ./ 3600 .*1000 # per second
Qsurf = -Qsurf # sign convention is opposite as ours
G = data[:, data[1,:] .== "G (kJ/hr/m2)"][2:end, :][indexrange] ./ 3600 .*1000 # per second
dates = data[:, data[1,:] .== "Date"][2:end, :][indexrange]

ρ_snow = FT(220) #this is a guess based on "max depth = 0.5m with SWE = 0.14m"
κ_air = FT(0.023)
κ_ice = FT(2.29)
κ_snow = FT(κ_air + (7.75*1e-5 *ρ_snow + 1.105e-6*ρ_snow^2)*(κ_ice-κ_air))
swe = FT(mean((data[:, data[1,:] .== "mod_SWE(m)"][2:end,:][indexrange])))

z_snow = 1e3 * swe ./ ρ_snow


# actual observed value
Tsurf = FT.(data[:, data[1,:] .== "obs_Ts (C)"][2:end,:][indexrange]) .+ 273.15


bulk_uobs = data[:, data[1,:] .== "obs_int (kJ)"][2:end,:][indexrange]
ρcz = (1e3*swe*2100+2100*1700*0.1)./1000 # they give U in kJ/m^2
bulk_tobs = bulk_uobs ./ ρcz .+ 273.15 # Really, this is estimated from the Tobs profile.


# Now we estimate the snow, soil and mixed snow/soil bulk T from the profile
layer_two_below = FT.(data[:, data[1,:] .== "obs_-7.5cm"][2:end,:][indexrange]) .+ 273.15
layer_one_below = FT.(data[:, data[1,:] .== "obs_-2.5cm"][2:end,:][indexrange]) .+ 273.15
ground_T =  FT.(data[:, data[1,:] .== "obs_0cm"][2:end,:][indexrange]) .+ 273.15
layer1 =    FT.(data[:, data[1,:] .== "obs_5cm"][2:end,:][indexrange]) .+ 273.15
layer2 =    FT.(data[:, data[1,:] .== "obs_12.5cm"][2:end,:][indexrange]) .+ 273.15
layer3 =    FT.(data[:, data[1,:] .== "obs_20cm"][2:end,:][indexrange]) .+ 273.15
layer4 =    FT.(data[:, data[1,:] .== "obs_27.5cm"][2:end,:][indexrange]) .+ 273.15
layer5 =    FT.(data[:, data[1,:] .== "obs_35cm"][2:end,:][indexrange]) .+ 273.15
# Snow alone (constant params ρc)
halfway = (z_snow +0.35)/2*100
estimated = (ground_T .* 2.5 .+ layer1 .* (8.75-2.5) .+ layer2 .*(16.25-8.75) .+ layer3 .* (23.75-16.25) .+ layer4*(31.25 .- 23.75) .+ layer5*(halfway - 31.25) .+ Tsurf .*(z_snow*100 - halfway)) ./(z_snow*100)
# Ground alone (constant ρc)
estimated_ground = ((10-5) .* layer_two_below .+ (5-1.25) .* layer_one_below .+ ground_T *(1.25)) ./ 10
# Here is our estimate of Tobs based on the observed profile.
# Idea is to "integrate" over profile dU ( = ρcz T), then divide by ρcz (c is the same for snow and soil in their model) to get bulk T
total_estimated = (estimated_ground .* 10 .*1700 .+ estimated .* (z_snow*100).*ρ_snow) ./ (1e3*swe*100 + 1700*0.1*100)

κ_soil = 2.5^0.6*0.57^0.4 # hacky OOM guess
#ρc_snow = volumetric_heat_capacity(0.0, ρ_snow, param_set)
#ν = FT(2.0*π/24/3600)
#d = (FT(2)*κ_snow/(ρc_snow*ν))^FT(0.5)

ground_5cm = (layer_one_below .+ layer_two_below) ./2

estimated_G = -2*κ_snow*κ_soil/(κ_soil+κ_snow)/(0.05)*(layer1 .- ground_5cm)
estimated_G2 = -κ_snow/0.05*(layer1 .- ground_T)


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
Tave0 = estimated[1]
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
T_ave = snow_temperature.(ρe_int, Ref(0.0), Ref(ρ_snow),Ref(param_set))
qb = Q_bott.(time_data, Ref(Qbott_spline))
qs = Q_surf.(time_data, Ref(Qsurf_spline))
coeffs = compute_profile_coefficients.(qs,qb, Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
t_profs = get_temperature_profile.(qs,qb, Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
        
tsurf_pw = [coeffs[k][1] for k in 1:N]
tbottom = [coeffs[k][2] for k in 1:N]
th = [coeffs[k][4] for k in 1:N]
#We can try different definitions of G and see how that affects bulk I
# simplest approx to integral is cumsum(div flux) *dt

divflux = -(Qsurf.-estimated_G)./z_snow
predicted = snow_temperature.(cumsum(divflux*1800) .+ρe_int[1], Ref(0.0), Ref(ρ_snow), Ref(param_set))

divflux2 = -(Qsurf.-estimated_G2)./z_snow
predicted2 = snow_temperature.(cumsum(divflux2*1800) .+ρe_int[1], Ref(0.0), Ref(ρ_snow), Ref(param_set))

divflux_orig = -(Qsurf.-G)./z_snow
predicted_orig = snow_temperature.(cumsum(divflux_orig*1800) .+ρe_int[1], Ref(0.0), Ref(ρ_snow), Ref(param_set))

plot1 = plot(time_data, T_ave, label = "Our Model, snow only")
#plot!(time_data, bulk_tobs, label = "UEB reported; Tave soil/snow")
plot!(time_data, total_estimated, label = "Est from data, snow/soil")
plot!(time_data, estimated, label = "Est from data, snow only")
#plot!(time_data, predicted, label = "Alt G[-5 to 5], snow only")
#plot!(time_data, predicted, label = "Our Model, using alt G, snow only")
plot!(title = "bulk T")
plot!(legend = :bottomright)
plot!(xticks = ([time_data[1],time_data[250],time_data[end]], [dates[1], dates[250], dates[end]]))


plot3 = plot(time_data, tbottom, label = "Our Model")
plot!(time_data, ground_T, label = "ground T")
plot!(title = "T(0)")
plot!(legend = :bottomright)
plot!(xticks = ([time_data[1],time_data[250],time_data[end]], [dates[1], dates[250], dates[end]]))

plot2 = plot(indexrange, tsurf_pw, label = "Our Model, Tsurf")
plot!(indexrange, Tsurf, label = "Observed Tsurf")
#plot!(indexrange, tbottom, label = "Our Model, Tbott")
#plot!(indexrange, ground_T, label = "Observed Tbott")
plot!(xlabel = "date", ylabel = "T surf (K)", legend = :topleft)
plot!(xticks = ([1,250,indexrange[end]], [dates[1], dates[250], dates[end]]))



### Force restore model below, but wait on until we get the EQ right? The output for Tsurf actually looks
### extremely similar to EQ - so let's work on getting it better in EQ first.
#=
# Use the same forcing, and initial condition for ρe_int. But now we need an IC for ρe_surf

Tsurf_0 = Tsurf[1]
l_0 = FT(0.0)
ρe_surf0 = volumetric_internal_energy(Tsurf_0, snow_parameters.ρ_snow, l_0, param_set)
ics = (aux) -> eltype(aux)(ρe_surf0)
fr_snow_model = FRSingleLayerSnowModel{typeof(snow_parameters), typeof(forcing),typeof(ics),typeof(ic)}(
    snow_parameters,
    forcing,
    ics,
    ic
)


function fr_init_land_model!(land, state, aux, localgeo, time)
    state.snow.ρe_int = land.snow.initial_ρe_int(aux)
    state.snow.ρe_surf = land.snow.initial_ρe_surf(aux)
end

fr_sources = (FluxDivergence{FT}(),ForceRestore{FT}(),)

fr_m = LandModel(
    param_set,
    m_soil;
    snow = fr_snow_model,
    source = fr_sources,
    init_state_prognostic = fr_init_land_model!,
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
    fr_m;
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
state_types = (Prognostic(),Auxiliary(),)
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
ρe_int_fr = [dons_arr[k]["snow.ρe_int"][1] for k in 1:N]



T_ave_fr = snow_temperature.(ρe_int_fr, Ref(0.0), Ref(ρ_snow),Ref(param_set))
l = 0.0
ρc_snow = volumetric_heat_capacity(l,ρ_snow,param_set)
ρe_surf = [dons_arr[k]["snow.ρe_surf"][1] for k in 1:N]
T_surf_fr = snow_temperature.(ρe_surf, Ref(0.0), Ref(ρ_snow),Ref(param_set))
z = 0:0.01:z_snow
T_h = [dons_arr[k]["snow.t_h"][1] for k in 1:N]


plot1 = plot(time_data, T_ave, label = "Our Model, G [0,5], snow only")
plot!(time_data, T_ave_fr, label = "Our Model, G [0,5], snow only, FR")
#plot!(time_data, bulk_tobs, label = "UEB reported; Tave soil/snow")
plot!(time_data, total_estimated, label = "Est from data, snow/soil")
plot!(time_data, estimated, label = "Est from data, snow only")
#plot!(time_data, predicted, label = "Alt G[-5 to 5], snow only")
plot!(time_data, predicted_orig, label = "Our Model, using their G, snow only")
plot!(title = "bulk T")
plot!(legend = :bottomright)
plot!(xticks = ([time_data[1],time_data[250],time_data[end]], [dates[1], dates[250], dates[end]]))


plot2 = plot(indexrange, tsurf_pw, label = "Our Model, Tsurf",ylim = [245,270])
plot!(indexrange, T_surf_fr, label = "FR, Us")
plot!(indexrange, Tsurf, label = "Observed Tsurf")
plot!(xlabel = "date", ylabel = "T surf (K)", legend = :topleft)
plot!(xticks = ([1,250,indexrange[end]], [dates[1], dates[250], dates[end]]))
=#
