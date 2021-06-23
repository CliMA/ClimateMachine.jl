# to do - make it so bulk ρc_snow can be different...
# figure out if Qsurf changes if Tsurf changes, so that even the bulk T changes if we change from EQ to FR
# map between U and ρe_int
# 

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

data = readdlm("/Users/katherinedeck/Downloads/USU_data.csv",',')
Qsurf = data[:, data[1,:] .== "Qsurf (kJ/m2/hr)"][2:end, :][:] ./ 3600 .*1000 # per second
Qsurf = -Qsurf # sign convention is opposite as ours
G = data[:, data[1,:] .== "G (kJ/hr/m2)"][2:end, :][:] ./ 3600 .*1000 # per second
dates = data[:, data[1,:] .== "Date"][2:end, :][:]
#mod_flux =data[:, data[1,:] .== "Flux(MFR)"][2:end, :][:]
#Qsurf = -mod_flux
start = 1
range = start:500 # no rain or snow in this window, no melting either

Tave = FT.(data[:, data[1,:] .== "mod_Tave(C)"][2:end,:][range]) .+ 273.15
Tsurf = FT.(data[:, data[1,:] .== "obs_Ts (C)"][2:end,:][range]) .+ 273.15
ρ_snow = FT(280) #this is a guess based on "max depth = 0.5m with SWE = 0.14m"
κ_air = FT(0.023)
κ_ice = FT(2.29)
κ_snow = FT(κ_air + (7.75*1e-5 *ρ_snow + 1.105e-6*ρ_snow^2)*(κ_ice-κ_air))
swe = FT(mean((data[:, data[1,:] .== "mod_SWE(m)"][2:end,:][range])))

z_snow = 1e3 * swe ./ ρ_snow

t0 = (start-1)*1800
t = FT.(0:1800:length(Qsurf)*1800-1) .- t0
soil_water_model = PrescribedWaterModel()
soil_heat_model = PrescribedTemperatureModel()
soil_param_functions = nothing

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)

snow_parameters = SnowParameters{FT,FT,FT,FT}(κ_snow,ρ_snow,z_snow)
Qsurf_spline = Spline1D(t[range], Qsurf[range])
function Q_surf(t::FT, Q::Spline1D) where {FT}
    return Q(t)
end
Qbott_spline = Spline1D(t[range], G[range])
function Q_bott(t::FT, Q::Spline1D) where {FT}
    return Q(t)
end
    

forcing = PrescribedForcing(FT;Q_surf = (t) -> eltype(t)(Q_surf(t,Qsurf_spline)),Q_bottom = (t) -> eltype(t)(Q_bott(t,Qbott_spline)))
Tave0 = Tave[1]
ρe_int0 = volumetric_internal_energy(Tave0, snow_parameters.ρ_snow, 0.0, param_set)
#ρe_int0 = U0/z_snow-ρ_snow*2100*(0.01)-LH_f0(param_set)*200
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
timeend = FT((range[end]-start)*1800)
dt = FT(60*30)

solver_config = ClimateMachine.SolverConfiguration(
    t0,
    timeend,
    driver_config,
    ode_dt = dt,
)
n_outputs = length(range);

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
coeffs = compute_profile_coefficients.(Q_surf.(time_data, Ref(Qsurf_spline)), Ref(0.0), Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
t_profs = get_temperature_profile.(Q_surf.(time_data, Ref(Qsurf_spline)), Ref(0.0), Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
        
tsurf_pw = [coeffs[k][1] for k in 1:N]
tsurf_obs = Tsurf
plot1 = plot(time_data, ρe_int)
#u_int = (ρe_int .+ ρ_snow*(2100*273.16+LH_f0(param_set))) .*z_snow .-z_snow*ρ_snow*2100*273.15
plot!(xticks = ([time_data[1],time_data[250],time_data[500]], [dates[1], dates[250], dates[500]]))
plot2 = plot(range, tsurf_pw, label = "piecewise model, EQ")
plot!(range, tsurf_obs, label = "obs")
plot!(xlabel = "date", ylabel = "T surf (K)", legend = :topleft)
plot!(xticks = ([1,250,500], [dates[1], dates[250], dates[500]]))
#savefig("snow_scatter2_eq.png")




mod_flux =data[:, data[1,:] .== "Flux(MFR)"][2:end, :][:]
Qsurf_fr = -mod_flux[range[4:end]]
Qsurf_spline = Spline1D(t[range[4:end]], Qsurf)
function Q_surf(t::FT, Q::Spline1D) where {FT}
    return Q(t)
end

forcing = PrescribedForcing(FT;Q_surf = (t) -> eltype(t)(Q_surf(t,Qsurf_spline)),Q_bottom = (t) -> eltype(t)(Q_bott(t,Qbott_spline)))
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
timeend = FT((range[end]-start)*1800)
dt = FT(60*30)
solver_config = ClimateMachine.SolverConfiguration(
    t0,
    timeend,
    driver_config,
    ode_dt = dt,
)
n_outputs = length(range);
    
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
T_ave = snow_temperature.(ρe_int, Ref(0.0), Ref(ρ_snow),Ref(param_set))
l = 0.0
ρc_snow = volumetric_heat_capacity(l,ρ_snow,param_set)
ρe_surf = [dons_arr[k]["snow.ρe_surf"][1] for k in 1:N]
T_surf_fr = snow_temperature.(ρe_surf, Ref(0.0), Ref(ρ_snow),Ref(param_set))
z = 0:0.01:z_snow
T_h = [dons_arr[k]["snow.t_h"][1] for k in 1:N]
T_bottom = T_h #Q_bottom = 0

plot!(range, T_surf_fr, label = "force restore")
scatter!(range, T_ave, label = "bulk T")
scatter!(range, data[:, data[1,:] .== "2mTair"][2:end, :][range], label = "2mTair")
plot(plot1,plot2)
