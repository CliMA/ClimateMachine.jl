# # Hydrostatic Equilibirum test for Richard's Equation

# This tutorial shows how to use ClimateMachine code to solve
# Richard's equation in a column of soil. We choose boundary
# conditions of zero flux at the top and bottom of the column,
# and then run the simulation long enough to see that the system
# is approach hydrostatic equilibrium, where the gradient of the
# pressure head is equal and opposite the gradient of the
# gravitational head.

# Note that by default we only support liquid water in this example.
# If you would like to include ice in a dynamical water model, phase transitions
# must be turned on by including freezing and thawing as a source term.

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
soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.495,
    Ksat = 0.0443 / (3600 * 100),
    S_s = 1e-3,
)

# Define the boundary conditions. The user can specify two conditions,
# either at the top or at the bottom, and they can either be Dirichlet
# (on `ϑ_l`) or Neumann (on `-K∇h`). Note that fluxes are supplied as
# scalars, inside the code they are multiplied by ẑ. The two conditions
# not supplied must be set to `nothing`. 
zero_value = FT(0.0)
surface_flux = (aux, t) -> zero_value
bottom_flux = (aux, t) -> zero_value
surface_state = nothing
bottom_state = nothing

# Define the initial state function.
initial_state = FT(0.494)
ϑ_l0 = (aux) -> initial_state

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
nelem_vert = 20
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
)
# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60 * 60 * 24 * 4)
dt = FT(5)

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
grads = solver_config.dg.state_gradient_flux

K∇h_vert_ind = varsindex(vars_state(m, GradientFlux(), FT), :soil, :water)[3]
ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)

v = round.(Int, aux[:, z_ind, 5] * 100) / 100
indices = [i[1] for i in indexin(unique(v), v)]

t = ODESolvers.gettime(solver_config.solver)
ϑ_l = Q[indices, ϑ_l_ind, :][:]
# gradients are not defined at t= 0, so initialize with NaNs
K∇h_vert = zeros(length(ϑ_l)) .+ FT(NaN)
z = aux[indices, z_ind, :][:]
all_vars = Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l, "K∇h_vert" => K∇h_vert)
all_data[0] = all_vars

step = [1];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
) do (init = false)
    t = ODESolvers.gettime(solver_config.solver)
    ϑ_l = Q[indices, ϑ_l_ind, :][:]
    K∇h_vert = grads[indices, K∇h_vert_ind, :][:]
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
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

# # Make some plots

t = [all_data[k]["t"][1] for k in 0:n_outputs]
t = ceil.(Int64, t ./ 86400)

# The initial state.
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

# Middle states
plot!(all_data[1]["ϑ_l"], z, label = string("t = ", string(t[2])))
plot!(all_data[2]["ϑ_l"], z, label = string("t = ", string(t[3])))
plot!(all_data[3]["ϑ_l"], z, label = string("t = ", string(t[4])))
# The final state
plot!(all_data[4]["ϑ_l"], z, label = string("t = ", string(t[5])))
# The expected slope in hydrostatic equilibrium.
slope = -soil_param_functions.S_s
plot!((z .- zmin) .* slope .+ all_data[4]["ϑ_l"][1], z, label = "expected")

# The porosity of the soil.
plot!(soil_param_functions.porosity .+ zeros(length(z)), z, label = "porosity")
# save the output.
savefig("./equilibrium_test_ϑ_l.png")


# Likewhile for the flux
# after 1 day
plot(
    -all_data[1]["K∇h_vert"],
    z,
    label = string("t = ", string(t[2])),
    ylabel = "z",
    xlabel = "-K∇h",
    legend = :bottomright,
    title = "Equilibrium test",
)

# Middle states

plot!(-all_data[2]["K∇h_vert"], z, label = string("t = ", string(t[3])))
plot!(-all_data[3]["K∇h_vert"], z, label = string("t = ", string(t[4])))
# The final state
plot!(-all_data[4]["K∇h_vert"], z, label = string("t = ", string(t[5])))

# save the output.
savefig("./equilibrium_test_flux.png")

# Finally, let's look at the divergence of the flux
final_flux = -all_data[4]["K∇h_vert"]
div_flux =
    (final_flux[2:end] - final_flux[1:(end - 1)]) ./ (z[2:end] - z[1:(end - 1)])
zval = (z[2:end] .+ z[1:(end - 1)]) ./ 2.0
zval = zval[iszero.(isinf.(div_flux))]
div_flux = div_flux[iszero.(isinf.(div_flux))]
plot(
    log10.(abs.(div_flux)),
    zval,
    label = "t = 4",
    xlabel = "Log10|∇⋅K∇h|",
    ylabel = "z",
)
savefig("./equilibrium_test_div_flux.png")
