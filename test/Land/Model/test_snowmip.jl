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


# The goal of this code is to compare the bulk snow temperature (computed using the average snow internal energy,
# and a bulk liquid fraction (we assume the same liquid fraction throughout the pack)) to obervations. We also
# compare the surface snow temp to observations.

# We specifically search for a data set with (1) periods of no melting, snowing, or raining and (2)
# little vegetation. The first ensures that SWE is conserved and that only radiation, latent, sensible,
# and ground heat fluxes affect the snow internal energy - all of which, with the exception of ground,
# should be measurable from data at the site. The second is important because it allows us to attribute all
# the fluxes measured at the site to entering/exiting the snowpack (rather than e.g. some of the energy being absorbed by
# a canopy, etc).

# Effectively, we are trying to isolate and simplify the energy equation as much as possible to test our "sub grid"
# temperature profile approach.

ClimateMachine.init()
FT = Float64
data = readdlm("/Users/katherinedeck/Downloads/SNOWMIP2_Alptal_openarea_input4CliMA_v3.csv",',')

start = 7161 
indexrange = start:start+350 # no rain or snow in this window, no melting either
Qsurf = data[:, data[1,:] .== "Qsurf (W/m2)"][2:end, :][indexrange] # J/m^2/s
Qsurf = -Qsurf # sign convention is opposite as ours. For us, all fluxes are expressed as F⃗ = f ẑ, ẑ is upwards. 
G = data[:, data[1,:] .== data[1,:][12]][2:end, :][indexrange]# J/m^2/s
G = -G #their convention is downward positive
dates = data[:, data[1,:] .== "Day"][2:end, :][indexrange]


swe = FT(mean((data[:, data[1,:] .== "SWE (m)"][2:end,:][indexrange])))
z_snow = FT(mean((data[:, data[1,:] .== "depsn(m)"][2:end,:][indexrange]))) 
ρ_snow = 1e3*swe ./ z_snow # Convert snow depth and swe to density of snow.
κ_air = FT(0.023) # constants that we could get from ClimaParameters.
κ_ice = FT(2.29)  # constants that we could get from ClimaParameters.
κ_snow = FT(κ_air + (7.75*1e-5 *ρ_snow + 1.105e-6*ρ_snow^2)*(κ_ice-κ_air)) # this is from Bonan's book - conductivity of snow from density.

# Note that these are BULK estimates of ρ_snow and κ_snow. Other models might using different values for the thin surface layer
# and the bulk (if ρ = ρ(z), i.e.)


# actual observed value. Ave = bulk
Tsurf = FT.(data[:, data[1,:] .== "surtsn (K)"][2:end,:][indexrange])
Tave = FT.(data[:, data[1,:] .== "SnowTemp (K)"][2:end,:][indexrange])
t = FT.(0:1800:length(Qsurf)*1800-1)

# We should have built and tested the snow model outside of ClimateMachine for prototyping. Since we did not,
# we have to do things like specify a soil model (which in this case, is prescribed soil T and water content, i.e.
# not solving a PDE since we arent caring about the soil model in this test). 
soil_water_model = PrescribedWaterModel()
soil_heat_model = PrescribedTemperatureModel()
soil_param_functions = nothing

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)

snow_parameters = SnowParameters{FT,FT,FT,FT}(κ_snow,ρ_snow,z_snow)
# Turns data of Qsurf and Qbottom into a spline, so we can evaluate it at any time.
Qsurf_spline = Spline1D(t, Qsurf)
function Q_surf(t::FT, Q::Spline1D) where {FT}
    return Q(t)
end
Qbott_spline = Spline1D(t, G)
function Q_bott(t::FT, Q::Spline1D) where {FT}
    return Q(t)
end
    
# This is how we pass in forcing data to snow model
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

# This is also another side effect of using the full ClimateMachine Land Model,
# rather than just prototyping snow outside of it first. ClimateMachine is set up to solve
# PDEs, so since the snow model is an ODE, we add the rhs as a "source term". Because it represents
# a flux divergence, we gave it that name (Qsurf - Qbottom).
sources = (FluxDivergence{FT}(),)


# This is the full LandModel. We had to include the soil model (no default value), added in the snow model,
# and gave a list of sources, and an initial condition function.
m = LandModel(
    param_set,
    m_soil;
    snow = eq_snow_model,
    source = sources,
    init_state_prognostic = init_land_model!,
)


# OK, here is another ungainly thing we need to do. You might remember that I mentioned that ClimateMachine was
# set up to solve PDEs, where all variables were defined on the same domain. So, our soil model is a column domain.
# we need to give zmin and zmax....but that means our snow model is ALSO a column domain! so effectively we are solving the same
# snow model ODE at every grid point in the column. this is obviously not ideal. ClimaCore (what replaces ClimateMachine) does
# *not* have this feature.
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


# Callbacks are a general term for functions that you evaluate on the state at specified times,
# or when specified events happen. here we are just extracting the state and auxiliary variables
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


# simulated ρe_int (bulk) as a function of time
ρe_int = [dons_arr[k]["snow.ρe_int"][1] for k in 1:N]
# simulated Tave as a function of time - liquid water fraction = 0.0 because no melting in this time period.
clima_Tave = snow_temperature.(ρe_int, Ref(0.0), Ref(ρ_snow),Ref(param_set))
qb = Q_bott.(time_data, Ref(Qbott_spline))
qs = Q_surf.(time_data, Ref(Qsurf_spline))
coeffs = compute_profile_coefficients.(qs,qb, Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
t_profs = get_temperature_profile.(qs,qb, Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
# ^^ temperature profile from 0 to z_snow        
tsurf_pw = [coeffs[k][1] for k in 1:N] # surface
tbottom = [coeffs[k][2] for k in 1:N] # bottom
th = [coeffs[k][4] for k in 1:N] # T at h

# Note that we are solving a super simple ODE = d(ρe_int)/dt = -(Qsurf(t) -Q_bottom(t))/z_snow. so we can just approximate ρe_int(t) with an integral :)
# it's kind of a sanity check
# also it means we dont need ANY of the infrastructure above for this test. we could just compute the integral using a cumulative sum, and then get bulk ρe_int from that.

# so i would recommend prototyping first outside of any LandHydrology.jl infrastructure.
divflux_orig = -(Qsurf.-G)./z_snow
predicted_orig = snow_temperature.(cumsum(divflux_orig*1800) .+ρe_int[1], Ref(0.0), Ref(ρ_snow), Ref(param_set))

plot1 = plot(time_data, clima_Tave, label = "Our Model")
plot!(time_data, Tave, label = "Est from data")
plot!(time_data, predicted_orig, label = "Our Model, predicted")
plot!(title = "bulk T")
plot!(legend = :bottomright)
#plot!(xticks = ([time_data[1],time_data[250],time_data[end]], [dates[1], dates[250], dates[end]]))


plot2 = plot(indexrange, tsurf_pw, label = "Our Model", title = "Tsurf")
plot!(indexrange, Tsurf, label = "Observed ")
plot!(xlabel = "index", ylabel = "T surf (K)", legend = :topleft)
#plot!(xticks = ([1,250,indexrange[end]], [dates[1], dates[250], dates[end]]))

plot(plot1,plot2)
