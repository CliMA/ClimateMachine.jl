#this runs but have not confirmed output. killed after 45 min
using MPI
using OrderedCollections
using StaticArrays
using Statistics

# - Load CLIMAParameters and ClimateMachine modules

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.Runoff
using ClimateMachine.Land.SoilWaterParameterizations
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
#using ClimateMachine.Mesh.Filters


# - Define the float type desired (`Float64` or `Float32`)
const FT = Float64;

# - Initialize ClimateMachine for CPU
ClimateMachine.init(; disable_gpu = true);

# Load plot helpers:
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

# # Set up the soil model


soil_heat_model = PrescribedTemperatureModel();

# Define the porosity, Ksat, and specific storage values for the soil. Note
# that all values must be given in mks units. The soil parameters chosen
# roughly correspond to Yolo light clay.
soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.4,
    Ksat = 6.94e-6 / 60,
    S_s = 5e-4,
);


# Specify the polynomial order and resolution.
N_poly = 1;
xres = FT(80)
yres = FT(80)
zres = FT(0.05)
# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-3);
xmax = FT(400)
ymax = FT(320)


heaviside(x) = 0.5 * (sign(x) + 1)
sigmoid(x, offset, width) = typeof(x)(exp((x-offset)/width)/(1+exp((x-offset)/width)))
precip_of_t = (t) -> eltype(t)(-((3.3e-4)/60) * (1-sigmoid(t, 300*60,10)))

ϑ_l0 = (aux) -> eltype(aux)(0.399- 0.05 * sigmoid(aux.z, -1,0.02))


bc =  SurfaceDrivenWaterBoundaryConditions(FT;
                                           precip_model = DrivenConstantPrecip{FT}(precip_of_t),
                                           runoff_model = CoarseGridRunoff{FT}(zres)
                                           )

soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = 2.0,  α = 100.0),
    initialϑ_l = ϑ_l0,
    boundaries = bc,
);

# Create the soil model - the coupled soil water and soil heat models.
m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);

# We are ignoring sources and sinks here, like runoff or freezing and thawing.
sources = ();

# Define the function that initializes the prognostic variables. This
# in turn calls the functions supplied to `soil_water_model`.
function init_soil_water!(land, state, aux, localgeo, time)
    state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
end


# Create the land model - in this tutorial, it only includes the soil.
m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_prognostic = init_soil_water!,
);

# # Specify the numerical configuration and output data.



function warp_maxwell_slope(xin, yin, zin; topo_max = 0.2, zmin =  -3, xmax = 400)
    FT = eltype(xin)
    zmax = FT(xin/xmax*topo_max)
    alpha = FT(1.0)- zmax/zmin
    zout = zmin+ (zin-zmin)*alpha
    x, y, z = xin, yin, zout
    return x, y, z
end

topo_max = FT(0.2)
# Create the driver configuration.
driver_config = ClimateMachine.MultiColumnLandModel(
    "LandModel",
    (N_poly, N_poly),
    (xres,yres,zres),
    xmax,
    ymax,
    zmax,
    param_set,
    m;
    zmin = zmin,
    #numerical_flux_first_order = CentralNumericalFluxFirstOrder(),now the default for us
    meshwarp = (x...) -> warp_maxwell_slope(x...;topo_max = topo_max, zmin = zmin, xmax = xmax),
);


# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60 * 350)
dt = FT(0.1); #5

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# Determine how often you want output.
const n_outputs = 500;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);

state_types = (Prognostic(), Auxiliary(), GradientFlux())
all_data = Dict[dict_of_nodal_states(solver_config, state_types; interp = false)]
time_data = FT[0] # store time data

# We specify a function which evaluates `every_x_simulation_time` and returns
# the state vector, appending the variables we are interested in into
# `all_data`.

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    dons = dict_of_nodal_states(solver_config, state_types; interp = false)
    push!(all_data, dons)
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

# # Run the integration
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));


# # Create some plots

output_dir = @__DIR__;

t = time_data ./ (60);

