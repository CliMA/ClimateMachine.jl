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
# (again by default). *It does not make sense to change this default*, since the
# liquid water equation would have no knowledge of the amount of ice in the soil.

# The equations are:

# ``
# \frac{ ∂ ϑ_l}{∂ t} = ∇ ⋅ K (T,θ_l, θ_i; ν, ...) ∇h( ϑ_l, z; ν, ...).
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

# ``- K ∇h(t, z = 0) = 0 ẑ ``

# `` -K ∇h(t, z = -10) = 0 ẑ``

# `` ϑ(t = 0, z) = ν-0.001 ``

# `` θ_i(t = 0, z) = 0.0. ``

# where ``\nu`` is the porosity.

# A word about the hydraulic conductivity: please see the
# [`hydraulic functions`](./hydraulic_functions.md) tutorial
# for options regarding this function. The user can choose to make it depend
# on the temperature and the amount of ice in the soil; the default, which we use
# here, is that `K` only depends on the liquid moisture content.

# # Preliminary setup

# Load external packages

using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Plots
using Logging
disable_logging(Logging.Warn)

# Load CLIMAParameters and ClimateMachine modules

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
using ClimateMachine.SystemSolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state


# - define the float type desired (`Float64` or `Float32`)
FT = Float64;
# - initialize ClimateMachine for CPU
ClimateMachine.init(; disable_gpu = true);

# Load a function that will create an interpolation of the
# simulation output, to be used in plotting:
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(
    clima_dir,
    "tutorials",
    "Land",
    "Soil",
    "interpolation_helper.jl",
));

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
# (on `ϑ_l`) or Neumann (on `K∇h`). Note that fluxes are supplied as
# scalars, inside the code they are multiplied by ẑ. The two conditions
# not supplied must be set to `nothing`. 
surface_flux = (aux, t) -> eltype(aux)(0.0)
bottom_flux = (aux, t) -> eltype(aux)(0.0)
surface_state = nothing
bottom_state = nothing;

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
    dirichlet_bc = Dirichlet(
        surface_state = surface_state,
        bottom_state = bottom_state,
    ),
    neumann_bc = Neumann(
        surface_flux = surface_flux,
        bottom_flux = bottom_flux,
    ),
);

# Create the soil model - the coupled soil water and soil heat models.
m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);

# We are ignoring sources and sinks here, like freezing and thawing.
sources = ();

# Define the function that initializes the prognostic variables. This
# in turn calls the functions supplied to `soil_water_model`.

function init_soil_water!(land, state, aux, coordinates, time)
    myFT = eltype(state)
    state.soil.water.ϑ_l = myFT(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_i = myFT(land.soil.water.initialθ_i(aux))
end;

# Create the land model - in this tutorial, it only includes the soil.
m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_prognostic = init_soil_water!,
);

# # Specify the numerical configuration and output data.

# Specify the polynomial order and vertical resolution.
N_poly = 5
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
timeend = FT(60 * 60 * 24 * 4)
dt = FT(200);

# Create the solver configuration, and specify numerical details related to the
# implicit ODE solver. 
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt)

dg = solver_config.dg
Q = solver_config.Q

vdg = DGModel(
    driver_config.bl,
    driver_config.grid,
    driver_config.numerical_flux_first_order,
    driver_config.numerical_flux_second_order,
    driver_config.numerical_flux_gradient,
    state_auxiliary = dg.state_auxiliary,
    direction = VerticalDirection(),
)

linearsolver = BatchedGeneralizedMinimalResidual(
    dg,
    Q;
    max_subspace_size = 30,
    atol = -1.0,
    rtol = 1e-9,
)

nonlinearsolver = JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-9)

ode_solver = ARK548L2SA2KennedyCarpenter(
    dg,
    vdg,
    NonLinearBackwardEulerSolver(
        nonlinearsolver;
        isadjustable = true,
        preconditioner_update_freq = 100,
    ),
    Q;
    dt = dt,
    t0 = 0,
    split_explicit_implicit = false,
    variant = NaiveVariant(),
)

solver_config.solver = ode_solver;

# Determine how often you want output.
const n_outputs = 4
const every_x_simulation_time = ceil(Int, timeend / n_outputs);
# Create a place to store this output.
all_data = Dict([k => Dict() for k in 0:(n_outputs - 1)]...);

# Obtain the initial state at the nodal points, and create
# the callback function to obtain the output from the simulation.

mygrid = solver_config.dg.grid;
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;
grads = solver_config.dg.state_gradient_flux

K∇h_vert_ind = varsindex(vars_state(m, GradientFlux(), FT), :soil, :water)[3]
ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)

t = ODESolvers.gettime(solver_config.solver)
ϑ_l = Q[:, ϑ_l_ind, :][:];
# Gradients are not defined at t= 0, so initialize with `nothing`.
z = aux[:, z_ind, :][:]
all_vars =
    Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l, "K∇h_vert" => [nothing])
all_data[0] = all_vars;

# We also create an additional cartesian grid upon which an interpolated solution
# of the DG output is evaluated. This is useful because the DG output is multi-valued
# at element boundaries.
zres = FT(0.2)
boundaries = [
    FT(0) FT(0) zmin
    FT(1) FT(1) zmax
]
resolution = (FT(2), FT(2), zres)
thegrid = solver_config.dg.grid
intrp_brck = create_interpolation_grid(boundaries, resolution, thegrid)
step = [1];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
) do (init = false)
    t = ODESolvers.gettime(solver_config.solver)
    iQ, iaux, igrads = interpolate_variables((Q, aux, grads), intrp_brck)
    ϑ_l = iQ[:, ϑ_l_ind, :][:]
    K∇h_vert = igrads[:, K∇h_vert_ind, :][:]
    all_vars = Dict{String, Array}(
        "t" => [t],
        "ϑ_l" => ϑ_l,
        "K∇h_vert" => K∇h_vert,
    )
    all_data[step[1]] = all_vars

    step[1] += 1
    nothing
end;

# # Run the integration
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

# We'll plot the moisture content vs depth in the soil, as well as
# the expected slope of `ϑ_l` in hydrostatic equilibrium when the soil
# is saturated. For `ϑ_l` values above porosity, the soil is
# saturated, and the pressure head changes from being equal to the matric
# potential to the pressure generated by compression of water and the soil
# matrix. The pressure head is continuous at porosity, but the derivative
# is not.

iz = iaux[:, z_ind, :][:]
t = [all_data[k]["t"][1] for k in 0:n_outputs]
t = ceil.(Int64, t ./ 86400)

plot(
    all_data[0]["ϑ_l"],
    z,
    label = string("t = ", string(t[1])),
    xlim = [0.47, 0.501],
    ylabel = "z",
    xlabel = "ϑ_l",
    legend = :bottomleft,
    title = "Equilibrium test",
)
plot!(all_data[1]["ϑ_l"], iz, label = string("t = ", string(t[2])))
plot!(all_data[2]["ϑ_l"], iz, label = string("t = ", string(t[3])))
plot!(all_data[3]["ϑ_l"], iz, label = string("t = ", string(t[4])))
plot!(all_data[4]["ϑ_l"], iz, label = string("t = ", string(t[5])))

slope = -soil_param_functions.S_s
plot!((iz .- zmin) .* slope .+ all_data[4]["ϑ_l"][1], iz, label = "expected")
plot!(soil_param_functions.porosity .+ zeros(length(z)), z, label = "porosity")
savefig("equilibrium_test_ϑ_l.png")

# ![](equilibrium_test_ϑ_l.png)

# We can also look at how the flux changes. Notice that the flux is zero
# at the top and bottom of the domain, which we specified in our boundary
# conditions. Over time, the flux is tending towards zero, as the system approaches
# equilibrium.
# After 1 day:
plot(
    -all_data[1]["K∇h_vert"],
    iz,
    label = string("t = ", string(t[2])),
    ylabel = "z",
    xlabel = "-K∇h",
    legend = :bottomright,
    title = "Equilibrium test",
)
plot!(-all_data[2]["K∇h_vert"], z, label = string("t = ", string(t[3])))
plot!(-all_data[3]["K∇h_vert"], z, label = string("t = ", string(t[4])))
plot!(-all_data[4]["K∇h_vert"], z, label = string("t = ", string(t[5])))
savefig("equilibrium_test_flux.png")

# ![](equilibrium_test_flux.png)
