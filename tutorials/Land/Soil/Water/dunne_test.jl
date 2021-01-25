using MPI
using JLD2, FileIO
using OrderedCollections
using StaticArrays
using Statistics
using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.Runoff
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

const FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

soil_heat_model = PrescribedTemperatureModel();

# Define the porosity, Ksat, and specific storage values for the soil. Note
# that all values must be given in mks units. The soil parameters chosen
# roughly correspond to Yolo light clay.
soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.4,
    Ksat = 6.94e-4 / 60,
    S_s = 5e-4,
);

# Define the boundary conditions. The user can specify two conditions,
# either at the top or at the bottom, and they can either be Dirichlet
# (on `ϑ_l`) or Neumann (on `-K∇h`). Note that fluxes are supplied as
# scalars, inside the code they are multiplied by ẑ. The two conditions
# not supplied must be set to `nothing`.

heaviside(x) = 0.5 * (sign(x) + 1)
sigmoid(x, offset, width) = typeof(x)(exp((x-offset)/width)/(1+exp((x-offset)/width)))
precip_of_t = (t) -> eltype(t)(-((3.3e-4)/120) * (1-sigmoid(t, 200*60,10)))
#precip_of_t = (t) -> eltype(t)(-((3.3e-4)/60) * (1-sigmoid(t, 200*60,10)))
# Define the initial state function. The default for `θ_i` is zero.
ϑ_l0 = (aux) -> eltype(aux)(0.399- 0.025 * sigmoid(aux.z, -0.25,0.02))
#ϑ_l0 = (aux) -> eltype(aux)(0.399- 0.025 * sigmoid(aux.z, -0.5,0.02))
zres = FT(0.04)
bc =  LandDomainBC(
    bottom_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0))),
    surface_bc = LandComponentBC(soil_water = SurfaceDrivenWaterBoundaryConditions(FT;
                                                precip_model = DrivenConstantPrecip{FT}(precip_of_t),
                                                runoff_model = CoarseGridRunoff{FT}(zres)
                                                                                   )),
)

soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = 2.0,  α = 100.0),
    initialϑ_l = ϑ_l0,
);

# Create the soil model - the coupled soil water and soil heat models.
m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);

# We are ignoring sources and sinks here, like runoff or freezing and thawing.
sources = (Pond{FT}(),);

# Define the function that initializes the prognostic variables. This
# in turn calls the functions supplied to `soil_water_model`.
function init_soil_water!(land, state, aux, localgeo, time)
    state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
    state.soil.water.l = eltype(state)(0.0)
end


# Create the land model - in this tutorial, it only includes the soil.
m = LandModel(
    param_set,
    m_soil;
    boundary_conditions = bc,
    source = sources,
    init_state_prognostic = init_soil_water!,
);

# # Specify the numerical configuration and output data.

# Specify the polynomial order and vertical resolution.
N_poly = 1;
nelem_vert = 40;

# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-3);


# # Define a warping function to build an analytic topography (we want a 2D slope, in 3D):
function warp_maxwell_slope(xin, yin, zin; xmax = 400.0, ymax = 320.0, zmax = 0.2)

    myFT = eltype(xin)
    b = myFT(zmax) # y intercept
    scaled_x = xin*myFT(xmax)
    scaled_y = yin*myFT(ymax)

    zdiff = - b / xmax * scaled_x + b # linear function with slope -0.2/400

     ## Linear relaxation towards domain maximum height
     x, y, z = scaled_x, scaled_y, zin + zdiff * heaviside(zin-myFT(-0.001))
     return x, y, z
 end

# Create the driver configuration.
driver_config = ClimateMachine.SingleStackConfiguration(
    "LandModel",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m;
    zmin = zmin,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
   # meshwarp = (x...) -> warp_maxwell_slope(x...),
);

# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60 * 300)
dt = FT(0.05); #5

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# Determine how often you want output.
const n_outputs = 500;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# Create a place to store this output.
mygrid = solver_config.dg.grid;
state_types = (Prognostic(), Auxiliary(), GradientFlux())
dons_arr = Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
time_data = FT[0] # store time data

# We specify a function which evaluates `every_x_simulation_time` and returns
# the state vector, appending the variables we are interested in into
# `dons_arr`.

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

# # Run the integration
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));


# Get the final state and create plots:
dons = dict_of_nodal_states(solver_config, state_types; interp = true)
push!(dons_arr, dons)
push!(time_data, gettime(solver_config.solver));

# Get z-coordinate
z = get_z(solver_config.dg.grid; rm_dupes = true);

N  = length(dons_arr)
tm  = [dons_arr[k]["soil.water.ϑ_l"][end] for k in 1:N]
tf  = [dons_arr[k]["soil.water.K∇h[3]"][end] for k in 1:N]
save("./runoff_test_data/dunne_alt_pond.jld2", "data", dons_arr)

pond_data = load("./runoff_test_data/dunne_alt_pond.jld2")["data"]
ksat_data = load("./runoff_test_data/dunne_alt_ksat.jld2")["data"]
test_data =  load("./runoff_test_data/dunne_alt_test.jld2")["data"]
N = length(test_data)
pond_tm = [pond_data[k]["soil.water.ϑ_l"][end] for k in 1:N]
pond_tf = [pond_data[k]["soil.water.K∇h[3]"][end] for k in 1:N]
pond_L = [pond_data[k]["soil.water.l"][end] for k in 1:N]

ksat_tm = [ksat_data[k]["soil.water.ϑ_l"][end] for k in 1:N]
ksat_tf = [ksat_data[k]["soil.water.K∇h[3]"][end] for k in 1:N]
ksat_L = [ksat_data[k]["soil.water.l"][end] for k in 1:N]


test_tm = [test_data[k]["soil.water.ϑ_l"][end] for k in 1:N]
test_tf = [test_data[k]["soil.water.K∇h[3]"][end] for k in 1:N]
test_L = [test_data[k]["soil.water.l"][end] for k in 1:N]
plot1 = plot(pond_tm, label = "ponding BC", xlabel = "time", ylabel = "surface ϑ_l")
plot!(test_tm, label = "test BC")
plot!(ksat_tm, label = "Ksat BC")

plot2 = plot(pond_tf, label = "ponding BC", xlabel = "time", ylabel = "surface infiltration")
plot!(test_tf, label = "test BC")
plot!(ksat_tf, label = "Ksat BC")

plot3 = plot(pond_L, label = "ponding BC", xlabel = "time", ylabel = "pond height")
plot!(test_L, label = "test BC")
plot!(ksat_L, label = "Ksat BC")
plot4 = plot(pond_data[50]["soil.water.ϑ_l"], pond_data[1]["z"], label = "t = 50, Pond", color = "red", xlabel = "ϑ_l", ylabel = "z", ylim = [-1,0])
plot!(test_data[50]["soil.water.ϑ_l"], test_data[25]["z"], label = "Test", color = "blue")
plot!(ksat_data[50]["soil.water.ϑ_l"], ksat_data[25]["z"], label = "Ksat", color = "green")
plot!(legend = :bottomleft)
plot(plot1,plot2,plot3,plot4,layout = 4)
savefig("./runoff_test_data/dunne_alt_comparison.png")
