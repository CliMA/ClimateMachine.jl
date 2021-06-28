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
start = 8
range = start:500 # no rain or snow in this window, no melting either

#Qsurf = data[:, data[1,:] .== "Qsurf (kJ/m2/hr)"][2:end, :][range] ./ 3600 .*1000 # per second
Qsurf = data[:, data[1,:] .== "Flux(Tave)"][2:end, :][range] ./ 3600 .*1000 # per second
Qsurf = -Qsurf # sign convention is opposite as ours
G = data[:, data[1,:] .== "G (kJ/hr/m2)"][2:end, :][range] ./ 3600 .*1000 # per second
dates = data[:, data[1,:] .== "Date"][2:end, :][range]
uobs = data[:, data[1,:] .== "obs_int (kJ)"][2:end,:][range]
u_model = data[:, data[1,:] .=="U from TS (Tave)"][2:end][range]


Tsurf = FT.(data[:, data[1,:] .== "obs_Ts (C)"][2:end,:][range]) .+ 273.15
ρ_snow = FT(280) 
κ_air = FT(0.023)
κ_ice = FT(2.29)
κ_snow = FT(κ_air + (7.75*1e-5 *ρ_snow + 1.105e-6*ρ_snow^2)*(κ_ice-κ_air))
#κ_snow = 0.09
swe = FT(mean((data[:, data[1,:] .== "mod_SWE(m)"][2:end,:][range])))

z_snow = 1e3 * swe ./ ρ_snow
ρcz = (1e3*swe*2100+2100*1700*0.1)./1000
tobs = uobs ./ ρcz .+ 273.15
modelt = u_model ./ ρcz .+ 273.15
#Tave = FT.(data[:, data[1,:] .== "mod_Tave(C)"][2:end,:][range]) .+ 273.15

function compute_u_from_T(Tave, ρ_snow, z_snow)
    return (Tave -273.15)*2100*(ρ_snow*z_snow+0.1*1700)
end


#t0 = (start-1)*1800
t = FT.(0:1800:length(Qsurf)*1800-1)# .- t0
soil_water_model = PrescribedWaterModel()
soil_heat_model = PrescribedTemperatureModel()
soil_param_functions = nothing

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)

snow_parameters = SnowParameters{FT,FT,FT,FT}(κ_snow,ρ_snow,z_snow)
Qsurf_spline = Spline1D(t, Qsurf )
function Q_surf(t::FT, Q::Spline1D) where {FT}
    return Q(t)
end
Qbott_spline = Spline1D(t, G)
function Q_bott(t::FT, Q::Spline1D) where {FT}
    return Q(t)
end
    

forcing = PrescribedForcing(FT;Q_surf = (t) -> eltype(t)(Q_surf(t,Qsurf_spline)),Q_bottom = (t) -> eltype(t)(Q_bott(t,Qbott_spline)))

ρe_int0 = u_model[1]/z_snow*FT(1000)#they report in kJ per m^2
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
#our_T_ave = snow_temperature.(ρe_int, Ref(0.0), Ref(ρ_snow),Ref(param_set))
our_T_ave = usu_bulk_snow_T.(ρe_int, Ref(ρ_snow),Ref(z_snow), Ref(param_set))
qb = Q_bott.(time_data, Ref(Qbott_spline))
qs = Q_surf.(time_data, Ref(Qsurf_spline))
ρc_snow = volumetric_heat_capacity(0.0, ρ_snow, param_set)
ν = FT(2.0*π/24/3600)
d = (FT(2)*κ_snow/(ρc_snow*ν))^FT(0.5)
our_T_surf = -qs*κ_snow/d .+ our_T_ave


coeffs = compute_profile_coefficients.(qs,qb, Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
t_profs = get_temperature_profile.(qs,qb, Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
        
tsurf_pw = [coeffs[k][1] for k in 1:N]
tsurf_obs = Tsurf



#function compute_u_from_ρe_int(ρe_int, ρ_snow, z_snow, l, param_set)
#    T_f = 273.15
#    T0 = T_0(param_set)
#    lf = LH_f0(param_set)
#    c_snow = c_i = cp_i(param_set)
    #T = snow_temperature(ρe_int, l, ρ_snow,param_set)
#    T = usu_bulk_snow_T(ρe_int, ρ_snow, z_snow, param_set)
#    Usnow = z_snow*ρe_int + ρ_snow*c_snow*z_snow*(T0-T_f)+ρ_snow*z_snow*lf
#    U_soil = 1700*2100*0.1*(T-T_f)
#    return (Usnow + U_soil)/1e3 # they report in kJ
#end
#our_U = compute_u_from_ρe_int.(ρe_int, Ref(ρ_snow), Ref(z_snow), Ref(0.0), Ref(param_set))
    
#plot1 = plot(time_data, our_U)

#plot!(time_data, uobs)
#plot!(time_data, u_model)


plot1a = plot(time_data, our_T_ave , label = "Our model, EQ")
plot!(time_data, tobs,label = "Data")
plot!(time_data, modelt, label = "UEB, EQ")
plot!(ylabel = "bulk T (K)")
plot!(xticks = ([time_data[1],time_data[250],time_data[end]], [dates[1], dates[250], dates[end]]))
plot2a = plot(time_data, tsurf_pw , label = "original model, EQ")
plot!(time_data, tsurf_obs, label = "Data")
plot!(time_data, our_T_surf, label = "fixed EQ")
plot!(xlabel = "date", ylabel = "T surf (K)", legend = :topleft)
plot!(xticks = ([time_data[1],time_data[250],time_data[end]], [dates[1], dates[250], dates[end]]))


###Our original model is incorrect b/c (1) T_ave is incorrect b/c of different thermal mass. Even correcting that,
# Tsurf is incorrect, because of ??? incorrect Tbottom? Qbottom formula isnt quite correct, wrong kappa?

#=


mod_flux =data[:, data[1,:] .== "Flux(MFR)"][2:end, :][range]
Qsurf_fr = -mod_flux
Qsurf_spline = Spline1D(t, Qsurf_fr)
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
ρe_int = [dons_arr[k]["snow.ρe_int"][1] for k in 1:N]
our_T_ave = snow_temperature.(ρe_int, Ref(0.0), Ref(ρ_snow),Ref(param_set))


#our_U = compute_u_from_ρe_int.(ρe_int, Ref(ρ_snow), Ref(z_snow), Ref(0.0), Ref(param_set))
    
#plot1 = plot(time_data, our_U)
uobs = data[:, data[1,:] .== "obs_int (kJ)"][2:end,:][range]
u_model = data[:, data[1,:] .=="U from TS (MFR)"][2:end][range]
uobs[uobs .== ""] .= uobs[4]
#plot!(time_data, uobs)
#plot!(time_data, u_model)
ρcz = (1e3*swe*2100+2100*1700*0.1)./1000
plot1b = plot(time_data, our_T_ave, label = "Our model, FR")
plot!(time_data, uobs ./ ρcz .+ 273.15, label = "Data")
plot!(time_data, u_model ./ ρcz .+ 273.15, label = "UEB, MFR")
plot!(ylabel = "bulk T (K)")
plot!(xticks = ([time_data[1],time_data[250],time_data[end]], [dates[1], dates[250], dates[end]]))
plot!(legend = :bottomright)
plot2a = plot(range, tsurf_pw, label = "our model, EQ")
plot!(range, tsurf_obs, label = "Data")
plot!(xlabel = "date", ylabel = "T surf (K)", legend = :topleft)
plot!(xticks = ([1,250,500], [dates[1], dates[250], dates[500]]))
=#
