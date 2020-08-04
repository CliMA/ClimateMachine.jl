# # Hydrostatic Equilibirum test for Richard's Equation

# This tutorial shows how to use ClimateMachine code to solve
# Richard's equation in a column of soil. We choose boundary
# conditions of zero flux at the top and bottom of the column,
# and then run the simulation long enough to see that the system
# is approach hydrostatic equilibrium, where the gradient of the
# pressure head is equal and opposite the gradient of the
# gravitational head.

# Note that freezing and thawing are turned off in this example. That
# means that `θ_ice`, initialized to zero by default, will remain zero.

# # Preliminary setup

# - load external packages

using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Plots

# - load CLIMAParameters and ClimateMachine modules

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


# - define the float type desired (`Float64` or `Float32`)
FT = Float64;

# - initialize ClimateMachine for CPU
ClimateMachine.init(; disable_gpu = true);


# # Set up the soil model

# We want to solveRichard's equation alone, i.e. do not solve
# the heat equation. The default is a constant temperature, but this
# only affects Richard's equation if one chooses a temperature dependent
# viscosity of water (see below).

soil_heat_model = PrescribedTemperatureModel{FT}()

# Define the porosity, Ksat, and specific storage values for the soil. Note
# that all values must be givne in mks.
soil_param_functions = SoilParamFunctions(
    porosity = 0.495,
    Ksat = 0.0443 / (3600 * 100),
    S_s = 1e-3,
)

# Define the boundary conditions. The user can specify two conditions,
# either at the top or at the bottom, and they can either be Dirichlet
# (on `ϑ_l`) or Neumann (on `-K∇h`). Note that fluxes are supplied as
# scalars, inside the code they are multiplied by ẑ. The two conditions
# not supplied must be set to `nothing`.

surface_flux = (aux, t) -> FT(0.0)
bottom_flux = (aux, t) -> FT(0.0)
surface_state = nothing
bottom_state = nothing

# Define the initial state function.
ϑ_l0 = (aux) -> FT(0.494)

# Create the SoilWaterModel. The defaults are a temperature independent
# viscosity, and no impedance factor due to ice. We choose to make the
# hydraulic conductivity a function of the moisture content `ϑ_l`,
# and employ the vanGenuchten hydraulic model with `n` = 2.0. The van
# Genuchten parameter `m` is calculated form `n`, and we use the default
# value for `α`.
soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = 2.0),
    initialϑ_l = ϑ_l0,
    dirichlet_bc = Dirichlet(
        surface_state = surface_state,
        bottom_state = bottom_state,
    ),
    neumann_bc = Neumann(
        surface_flux = surface_flux,
        bottom_flux = bottom_flux,
    ),
)

# Create the soil model - the coupled soil water and soil heat models.
m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)

# We are ignoring sources and sinks here.
sources = ()

# Define the function that initializes the prognostic variables. This
# in turn calls the functions supplied to soil_water_model. The default
# initialization for ice is zero volumetric ice fraction. This should not
# be altered unless freeze thaw is added.
function init_soil_water!(land, state, aux, coordinates, time)
    FT = eltype(state)
    state.soil.water.ϑ_l = FT(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_ice = FT(land.soil.water.initialθ_ice(aux))
end;



# Create the land model - in this tutorial, it only includes the soil.
m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_prognostic = init_soil_water!,
)

# # Specify the numerical configuration and output data.

# Specify the polynomial order and vertical resolution.
N_poly = 5;
nelem_vert = 20;

# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-10)

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
timeend = FT(60 * 60 * 24 * 4)
dt = FT(5)

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# Determine how often you want output.
const n_outputs = 5;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# Create a place to store this output.
all_data = Dict([k => Dict() for k in 0:n_outputs]...)

# Obtain the initial state, and create the callback
# function to obtain the output from the simulation.
mygrid = solver_config.dg.grid;
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;

t = ODESolvers.gettime(solver_config.solver)
state_vars = SingleStackUtils.get_vars_from_nodal_stack(
    mygrid,
    Q,
    vars_state(m, Prognostic(), FT),
)
grads = SingleStackUtils.get_vars_from_nodal_stack(
    mygrid,
    solver_config.dg.state_gradient_flux,
    vars_state(m, GradientFlux(), FT),
)
aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
    mygrid,
    aux,
    vars_state(m, Auxiliary(), FT),
)
all_vars = OrderedDict(state_vars..., aux_vars..., grads...);
all_vars["t"] = [t]
all_data[0] = all_vars

step = [1];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
) do (init = false)
    t = ODESolvers.gettime(solver_config.solver)
    grads = SingleStackUtils.get_vars_from_nodal_stack(
        mygrid,
        solver_config.dg.state_gradient_flux,
        vars_state(m, GradientFlux(), FT),
    )
    state_vars = SingleStackUtils.get_vars_from_nodal_stack(
        mygrid,
        Q,
        vars_state(m, Prognostic(), FT),
    )
    aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
        mygrid,
        aux,
        vars_state(m, Auxiliary(), FT),
    )
    all_vars = OrderedDict(state_vars..., aux_vars..., grads...)
    all_vars["t"] = [t]
    all_data[step[1]] = all_vars

    step[1] += 1
    nothing
end;

# # Run the integration
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

# # Get the final state back.
t = ODESolvers.gettime(solver_config.solver)
state_vars = SingleStackUtils.get_vars_from_nodal_stack(
    mygrid,
    Q,
    vars_state(m, Prognostic(), FT),
)
grads = SingleStackUtils.get_vars_from_nodal_stack(
    mygrid,
    solver_config.dg.state_gradient_flux,
    vars_state(m, GradientFlux(), FT),
)
aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
    mygrid,
    aux,
    vars_state(m, Auxiliary(), FT),
)
all_vars = OrderedDict(state_vars..., aux_vars..., grads...);
all_vars["t"] = [t]
all_data[n_outputs] = all_vars

t = [all_data[k]["t"] for k in 0:n_outputs]


# # Create some plots. 
slope = -1e-3

# The initial state.
plot(
    all_data[0]["soil.water.ϑ_l"],
    all_data[0]["z"],
    label = string("t = ", string(t[1][1])),
    xlim = [0.47, 0.501],
    ylabel = "z",
    xlabel = "ϑ_l",
    legend = :bottomleft,
    title = "Equilibrium test",
)

# A middle state.
plot!(
    all_data[3]["soil.water.ϑ_l"],
    all_data[3]["z"],
    label = string("t = ", string(t[4][1])),
)
# The final state
plot!(
    all_data[5]["soil.water.ϑ_l"],
    all_data[5]["z"],
    label = string("t = ", string(t[6][1])),
)
# The expected slope in hydrostatic equilibrium.
plot!(
    (all_data[0]["z"] .+ 10.0) .* slope .+ all_data[5]["soil.water.ϑ_l"][1],
    all_data[0]["z"],
    label = "expected",
)

# The porosity of the soil.
plot!(
    1e-3 .+ all_data[0]["soil.water.ϑ_l"],
    all_data[0]["z"],
    label = "porosity",
)
# save the output.
savefig("./equilibrium_test_ϑ_l_vG.png")
