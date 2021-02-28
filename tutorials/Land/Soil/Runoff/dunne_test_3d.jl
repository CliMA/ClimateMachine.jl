using MPI
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

soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.4,
    Ksat = 6.94e-4 / 60,
    S_s = 5e-4,
);
heaviside(x) = 0.5 * (sign(x) + 1)
precip_of_t = (t) -> eltype(t)(-((3.3e-4)/60)*heaviside(200*60-t))
function he(z)
    z_interface = -0.5
    ν = 0.4
    S_s = 5e-4
    α = 100.0
    n = 2.0
    m = 0.5
    if z < z_interface
        return -S_s * (z - z_interface) + ν
    else
        return ν * (1 + (α * (z - z_interface))^n)^(-m)
    end
end
ϑ_l0 = (aux) -> eltype(aux)(he(aux.z))

N_poly = 1;
xres = FT(80)
yres = FT(80)
zres = FT(0.2)
# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-5);
xmax = FT(400)
ymax = FT(320)

bc =  LandDomainBC(
    bottom_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0))),
    surface_bc = LandComponentBC(soil_water = SurfaceDrivenWaterBoundaryConditions(FT;
                                                precip_model = DrivenConstantPrecip{FT}(precip_of_t),
                                                runoff_model = CoarseGridRunoff{FT}(zres)
                                                                                   )),
    lateral_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)))
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
    boundary_conditions = bc,
    source = sources,
    init_state_prognostic = init_soil_water!,
);



# # Define a warping function to build an analytic topography (we want a 2D slope, in 3D):

function warp_maxwell_slope(xin, yin, zin; topo_max = 0.2, zmin =  -5, xmax = 400)
    FT = eltype(xin)
    zmax = FT(xin/xmax*topo_max)
    alpha = FT(1.0)- zmax/zmin
    zout = zmin+ (zin-zmin)*alpha
    x, y, z = xin, yin, zout
    return x, y, z
end


#function inverse_warp_maxwell_slope(xin, yin, zin; topo_max = 0.2, zmin =  -5, xmax = 400)
#    FT = eltype(xin)
#    zmax = FT(xin/xmax*topo_max)
#    alpha = FT(1.0)- zmax/zmin
#    zout = (zin-zmin)/alpha+zmin
#    return zout
# end


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
timeend = FT(60* 300)
dt = FT(1);

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);


# Determine how often you want output.
const n_outputs = 500;
const every_x_simulation_time = ceil(Int, timeend / n_outputs);
state_types = (Prognostic(), Auxiliary(), GradientFlux())
dons_arr = Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
time_data = FT[0] # store time data

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));
