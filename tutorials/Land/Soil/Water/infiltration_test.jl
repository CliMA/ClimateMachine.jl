# # Infiltration of water into sandy loam.

# This tutorial shows how to use ClimateMachine code to solve
# Richard's equation in a column of soil, assuming a specified
# infiltration flux of water at the surface and free drainage.

# A word about ice: the dynamic water model includes as state
# variables `ϑ_l` and `θ_ice`. However, the right hand side of the ice
# equation is zero unless freeze/thaw source terms are explicitly turned on.
# In this case, liquid water and ice can be transformed into each other while
# conserving the total water mass. If freezing and thawing are not turned on
# (the default), the amount of ice in the model is zero for all space and time
# (again by default). *It does not make sense to change this default*, since the
# liquid water equation would have no knowledge of the amount of ice in the soil.

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
FT = Float64
# - initialize ClimateMachine for CPU
ClimateMachine.init(; disable_gpu = true)
# # Set up the soil model

# We want to solve Richard's equation alone, without simultaneously
# solving the heat equation. One can choose to essentially ignore temperature
# (the default), or to choose a prescribed temperature function with which to drive
# Richard's equation (via the temperature dependence of viscosity, or via freeze/thaw).
# Here we ignore the effects of temperature and freezing and thawing. 

soil_heat_model = PrescribedTemperatureModel{FT}()

# Define the porosity, Ksat, and specific storage values for the soil. Note
# that all values must be given in mks units. The soil parameters chosen
# roughly correspond to Yolo light clay.
Ksat = FT(4.42 / 3600 / 100) # m/s
ν = FT(0.41)
soil_param_functions =
    SoilParamFunctions{FT}(porosity = ν, Ksat = Ksat, S_s = 1e-3)

# Define the boundary conditions. The user can specify two conditions,
# either at the top or at the bottom, and they can either be Dirichlet
# (on `ϑ_l`) or Neumann (on `K∇h`). Note that fluxes are supplied as
# scalars, inside the code they are multiplied by ẑ. The two conditions
# not supplied must be set to `nothing`. 
zero_value = FT(0.0)
# A positive value of `K∇h` implies flux downwards.
surface_flux = (aux, t) -> Ksat
# Free drainage implies ∇h = ẑ
bottom_flux = (aux, t) -> aux.soil.water.K
surface_state = nothing
bottom_state = nothing

# Define the initial state function. Here we show how to make the initial condition
# a function of space. 
function ϑ_l0(aux)
    z = aux.z
    if z > -0.5
        output = ν + 1e-3
    else
        output = 0.9 * ν
    end

    return output
end

# Create the SoilWaterModel. The defaults are a temperature independent
# viscosity, and no impedance factor due to ice. We choose to make the
# hydraulic conductivity a function of the moisture content `ϑ_l`,
# and employ the vanGenuchten hydraulic model with `n` = 2.0. The van
# Genuchten parameter `m` is calculated from `n`, and we use the default
# value for `α`.
soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(α = 7.5, n = 1.89),
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
# initialization for ice is zero volumetric ice fraction. ***This should not
# be altered unless freeze thaw is added.***
function init_soil_water!(land, state, aux, coordinates, time)
    FT = eltype(state)
    state.soil.water.ϑ_l = FT(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_ice = FT(land.soil.water.initialθ_ice(aux))
end
# Create the land model - in this tutorial, it only includes the soil.
m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_prognostic = init_soil_water!,
)

# # Specify the numerical configuration and output data.

# Specify the polynomial order and vertical resolution.
N_poly = 5
nelem_vert = 10
# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-3)
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
)
# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60 * 60 * 1)
dt = FT(0.05)

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt)

# Determine how often you want output.
const n_outputs = 4
const every_x_simulation_time = ceil(Int, timeend / n_outputs)
# Create a place to store this output.
all_data = Dict([k => Dict() for k in 0:n_outputs]...)

# Obtain the initial state at the nodal points, and create
# the callback function to obtain the output from the simulation.
# There are (N_poly+1)^3 nodal points. However, our simulation is effectively 1-d,
# as we only care about the values of the state at unique values of z (the values
# at different x and y will agree as long as they are at the same z). We therefore
# select only certain indices in z.
mygrid = solver_config.dg.grid;
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;
ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)

v = round.(Int, aux[:, z_ind, 5] * 100) / 100
indices = [i[1] for i in indexin(unique(v), v)]

t = ODESolvers.gettime(solver_config.solver)
ϑ_l = Q[indices, ϑ_l_ind, :][:]
z = aux[indices, z_ind, :][:]
all_vars = Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l)
all_data[0] = all_vars

step = [1];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
) do (init = false)
    t = ODESolvers.gettime(solver_config.solver)
    ϑ_l = Q[indices, ϑ_l_ind, :][:]
    all_vars = Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l)
    all_data[step[1]] = all_vars

    step[1] += 1
    nothing
end;

# # Run the integration
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

# Get the final state
t = ODESolvers.gettime(solver_config.solver)
ϑ_l = Q[indices, ϑ_l_ind, :][:]
all_vars = Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l)
all_data[n_outputs] = all_vars

# # Make some plots

t = [all_data[k]["t"][1] for k in 0:n_outputs]
t = ceil.(Int64, t ./ 60)

# The initial state.
plot(
    all_data[0]["ϑ_l"],
    z,
    label = string("t = ", string(t[1])),
    #xlim = [0.47, 0.501],
    ylabel = "z",
    xlabel = "ϑ_l",
    legend = :bottomleft,
)

# Middle states
plot!(all_data[1]["ϑ_l"], z, label = string("t = ", string(t[2])))
plot!(all_data[2]["ϑ_l"], z, label = string("t = ", string(t[3])))
plot!(all_data[3]["ϑ_l"], z, label = string("t = ", string(t[4])))
# The final state
plot!(all_data[4]["ϑ_l"], z, label = string("t = ", string(t[5])))

# save the output:
savefig("./infiltration_test_ϑ_l.png")

# ![](infiltration_test_ϑ_l.png)
