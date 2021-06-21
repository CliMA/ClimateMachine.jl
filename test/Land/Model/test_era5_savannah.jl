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

data = readdlm("/Users/katherinedeck/Downloads/SB_SAV_2017_hourly_revised.csv",',')
Qsurf = data[:, data[1,:] .== "Qsurf"][2:end, :][:] ./ 3600 ./ 24 # per second
Qsurf = -Qsurf ./ 31
data_old = readdlm("/Users/katherinedeck/Downloads/SB_SAV_2017_hourly.csv",',')
Qsurf_old = data_old[:, data_old[1,:] .== "Qsurf"][2:end, :][:] ./ 3600 ./ 24 # per second
# 300:440 - no snowfall.
start = 340
range = 340:411
Tsurf = FT.(data[:, data[1,:] .== "surtsn(K)"][2:end,:][range])
ρ_snow = FT(mean(data[:, data[1,:] .== "rhosn(kgm-3)"][2:end,:][range]))
κ_air = FT(0.023)
κ_ice = FT(2.29)
κ_snow = FT(κ_air + (7.75*1e-5 *ρ_snow + 1.105e-6*ρ_snow^2)*(κ_ice-κ_air))
z_snow = FT(mean(data[:,data[1,:] .== "depsn(m)"][2:end,:][range]))
t0 = start*3600
t = FT.(0:3600:length(Qsurf)*3600-1) .- t0
soil_water_model = PrescribedWaterModel()
soil_heat_model = PrescribedTemperatureModel()
soil_param_functions = nothing

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)

snow_parameters = SnowParameters{FT,FT,FT,FT}(κ_snow,ρ_snow,z_snow)
Qsurf_spline = Spline1D(t, Qsurf)
function Q_surf(t::FT, Q::Spline1D) where {FT}
    return Q(t)
end
    

forcing = PrescribedForcing(FT;Q_surf = (t) -> eltype(t)(Q_surf(t,Qsurf_spline)))
Tave_0 = mean(Tsurf)
l_0 = FT(0.0)
ρe_int0 = volumetric_internal_energy(Tave_0, snow_parameters.ρ_snow, l_0, param_set)
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
timeend = FT((range[end]-start)*3600)
dt = FT(60*60)

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
T_ave = snow_temperature.(ρe_int, Ref(0.0), Ref(100.0),Ref(param_set))
coeffs = compute_profile_coefficients.(Q_surf.(time_data, Ref(Qsurf_spline)), Ref(0.0), Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
t_profs = get_temperature_profile.(Q_surf.(time_data, Ref(Qsurf_spline)), Ref(0.0), Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
        
tsurf_pw = [coeffs[k][1] for k in 1:N]
tsurf_era5 = Tsurf
#anim = @animate for i ∈ 1:N
#    plot(t_profs[i].(z),z, ylim = [0,1.5], xlim = [240,300], label = "Piecewise model")
#    t = time_data[i]
#    scatter!([Tsurf[range[i]]], [FT(snow_parameters.z_snow)], label = "ERA5 Tsurf")
#end
#gif(anim, "snow2.gif", fps = 6)

#=
RMSEv  = round(sqrt(mean((tsurf_pw - tsurf_era5).^2)),digits = 2) 
titlev = "snow_scatter_rho+100_eq"
snowscatter = plot(tsurf_pw, tsurf_era5, seriestype = :scatter, reg = true,
                   title = titlev,
                   legend = false,
                   ylim = [220,280],xlim=[220,280],
                   xlab = "Tsurf Piecewise model", ylab = "Tsurf ERA5")
annotate!(260,270,text(string("RMSE = ", RMSEv), :left, 12))
Plots.abline!(1, 1, line=:dash)
#savefig(snowscatter,string(titlev,".png"))

#snowts = plot(1:28,tsurf_pw)
=#
plot1 = plot(range, Qsurf[range])
plot2 = scatter(range, tsurf_pw, label = "piecewise model, EQ")
scatter!(range, tsurf_era5, label = "ERA5")
plot!(xlabel = "hour", ylabel = "T surf (K)", legend = :topleft)
#savefig("snow_scatter2_eq.png")

Tsurf_0 = Tsurf[1]
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
timeend = FT((range[end]-start)*3600)
dt = FT(60*60)

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
ρe_int = [dons_arr[k]["snow.ρe_int"][1] for k in 1:N]
T_ave = snow_temperature.(ρe_int, Ref(0.0), Ref(ρ_snow),Ref(param_set))
l = 0.0
ρc_snow = volumetric_heat_capacity(l,ρ_snow,param_set)
ρe_surf = [dons_arr[k]["snow.ρe_surf"][1] for k in 1:N]
T_surf_fr = snow_temperature.(ρe_surf, Ref(0.0), Ref(ρ_snow),Ref(param_set))
z = 0:0.01:z_snow
T_h = [dons_arr[k]["snow.t_h"][1] for k in 1:N]
T_bottom = T_h #Q_bottom = 0

scatter!(range, T_surf_fr, label = "force restore")
scatter!(range, T_ave, label = "bulk T")
scatter!(range, data[:, data[1,:] .== "2mTair"][2:end, :][range], label = "2mTair")
plot(plot1,plot2)
