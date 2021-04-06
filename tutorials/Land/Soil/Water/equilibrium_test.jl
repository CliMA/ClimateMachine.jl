# # Hydrostatic Equilibrium test for Richards Equation

# This tutorial shows how to use `ClimateMachine` code to solve
# Richards equation in a column of soil. We choose boundary
# conditions of zero flux at the top and bottom of the column,
# and then run the simulation long enough to see that the system
# is approaching hydrostatic equilibrium, where the gradient of the
# pressure head is equal and opposite the gradient of the
# gravitational head. Note that the [`SoilWaterModel`](@ref
# ClimateMachine.Land.SoilWaterModel) includes
# a prognostic equation for the volumetric ice fraction,
# as ice is a form of water that must be accounted for to ensure
# water mass conservation. If freezing and thawing are not turned on
# (the default), the amount of ice in the model is zero for all space and time
# (again by default). 

# The equations are:

# ``
# \frac{ ∂ ϑ_l}{∂ t} = ∇ ⋅ K (T, ϑ_l, θ_i; ν, ...) ∇h( ϑ_l, z; ν, ...).
# ``

# ``
# \frac{ ∂ θ_i}{∂ t} = 0
# ``

# Here

# ``t`` is the time (s),

# ``z`` is the location in the vertical (m),

# ``T`` is the temperature of the soil (K),

# ``K`` is the hydraulic conductivity (m/s),

# ``h`` is the hydraulic head (m),

# ``ϑ_l`` is the augmented volumetric liquid water fraction,

# ``θ_i`` is the volumetric ice fraction, and

# ``ν, ...`` denotes parameters relating to soil type, such as porosity.


# We will solve this equation in an effectively 1-d domain with ``z ∈ [-10,0]``,
# and with the following boundary and initial conditions:

# ``- K ∇h(t, z = 0) = 0 ẑ ``

# `` -K ∇h(t, z = -10) = 0 ẑ``

# `` ϑ(t = 0, z) = ν-0.001 ``

# `` θ_i(t = 0, z) = 0.0. ``

# where ``\nu`` is the porosity.

# A word about the hydraulic conductivity: please see the
# [`hydraulic functions`](./hydraulic_functions.md) tutorial
# for options regarding this function. The user can choose to make it depend
# on the temperature and the amount of ice in the soil; the default, which we use
# here, is that `K` only depends on the liquid moisture content.

# Lastly, our formulation of this equation allows for a continuous solution in both
# saturated and unsaturated areas, following [Woodward00a](@cite).

# # Preliminary setup

# - Load external packages

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


# - Define the float type desired (`Float64` or `Float32`)
const FT = Float64;

# - Initialize ClimateMachine for CPU
ClimateMachine.init(; disable_gpu = true);

# Load plot helpers:
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

# # Set up the soil model

# We want to solve Richards equation alone, without simultaneously
# solving the heat equation. Because of that, we choose a
# [`PrescribedTemperatureModel`](@ref
# ClimateMachine.Land.PrescribedTemperatureModel).
# The user can supply a function for temperature,
# depending on time and space; if this option is desired, one could also
# choose to model the temperature dependence of viscosity, or to drive
# a freeze/thaw cycle, for example. If the user simply wants to model
# Richards equation for liquid water, the defaults will allow for that.
# Here we ignore the effects of temperature and freezing and thawing,
# using the defaults.

soil_heat_model = PrescribedTemperatureModel();

# Define the porosity, Ksat, and specific storage values for the soil. Note
# that all values must be given in mks units. The soil parameters chosen
# roughly correspond to Yolo light clay.
soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.495,
    Ksat = 0.0443 / (3600 * 100),
    S_s = 1e-3,
);

# Define the boundary conditions. The user can specify two conditions,
# either at the top or at the bottom, and they can either be Dirichlet
# (on `ϑ_l`) or Neumann (on `-K∇h`). Note that fluxes are supplied as
# scalars, inside the code they are multiplied by ẑ.

surface_flux = (aux, t) -> eltype(aux)(0.0)
bottom_flux = (aux, t) -> eltype(aux)(0.0);

# Our problem is effectively 1D, so we do not need to specify lateral boundary
# conditions.
bc = LandDomainBC(
    bottom_bc = LandComponentBC(soil_water = Neumann(bottom_flux)),
    surface_bc = LandComponentBC(soil_water = Neumann(surface_flux)),
);

# Define the initial state function. The default for `θ_i` is zero.
ϑ_l0 = (aux) -> eltype(aux)(0.494);

# Create the SoilWaterModel. The defaults are a temperature independent
# viscosity, and no impedance factor due to ice. We choose to make the
# hydraulic conductivity a function of the moisture content `ϑ_l`,
# and employ the vanGenuchten hydraulic model with `n` = 2.0. The van
# Genuchten parameter `m` is calculated from `n`, and we use the default
# value for `α`.
soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = 2.0),
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

# # Specify the numerical configuration and output data.

# Specify the polynomial order and vertical resolution.
N_poly = 2;
nelem_vert = 20;

# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-10);

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
);

# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60 * 60 * 24 * 36)
dt = FT(100);

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# Determine how often you want output.
const n_outputs = 6;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# Create a place to store this output.
state_types = (Prognostic(), Auxiliary(), GradientFlux())
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

# Get z-coordinate
z = get_z(solver_config.dg.grid; rm_dupes = true);

# # Create some plots
# We'll plot the moisture content vs depth in the soil, as well as
# the expected profile of `ϑ_l` in hydrostatic equilibrium.
# For `ϑ_l` values above porosity, the soil is
# saturated, and the pressure head changes from being equal to the matric
# potential to the pressure generated by compression of water and the soil
# matrix. The profile can be solved
# for analytically by (1) solving for the form that `ϑ_l(z)` must take
# in both the saturated and unsaturated zones to satisfy the steady-state
# requirement with zero flux boundary conditions, (2) requiring that
# at the interface between saturated and unsaturated zones, the water content
# equals porosity, and (3) solving for the location of the interface by
# requiring that the integrated water content at the end matches that
# at the beginning (yielding an interface location of `z≈-0.56m`).
output_dir = @__DIR__;

t = time_data ./ (60 * 60 * 24);

plot(
    dons_arr[1]["soil.water.ϑ_l"],
    dons_arr[1]["z"],
    label = string("t = ", string(t[1]), "days"),
    xlim = [0.47, 0.501],
    ylabel = "z",
    xlabel = "ϑ_l",
    legend = :bottomleft,
    title = "Equilibrium test",
);
plot!(
    dons_arr[2]["soil.water.ϑ_l"],
    dons_arr[2]["z"],
    label = string("t = ", string(t[2]), "days"),
);
plot!(
    dons_arr[7]["soil.water.ϑ_l"],
    dons_arr[7]["z"],
    label = string("t = ", string(t[7]), "days"),
);
function expected(z, z_interface)
    ν = 0.495
    S_s = 1e-3
    α = 2.6
    n = 2.0
    m = 0.5
    if z < z_interface
        return -S_s * (z - z_interface) + ν
    else
        return ν * (1 + (α * (z - z_interface))^n)^(-m)
    end
end
plot!(expected.(dons_arr[1]["z"], -0.56), dons_arr[1]["z"], label = "expected");

plot!(
    1e-3 .+ dons_arr[1]["soil.water.ϑ_l"],
    dons_arr[1]["z"],
    label = "porosity",
);
# save the output.
savefig(joinpath(output_dir, "equilibrium_test_ϑ_l_vG.png"))
# ![](equilibrium_test_ϑ_l_vG.png)
# # References
# - [Woodward00a](@cite)
