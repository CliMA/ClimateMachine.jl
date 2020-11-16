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
# saturated and unsaturated areas, following [1].

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
#using ClimateMachine.Mesh.Filters


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
    porosity = 0.4,
    Ksat = 6.94e-6 / 60, #0.0443 / (3600 * 100) why do we divide by 100 in original model?
    S_s = 5e-4, #1e-3,
);

#Ksat = 6.94e-5 / 60 yields error :
#ERROR: LoadError: TaskFailedException:
#Effective saturation is negative

# Define the boundary conditions. The user can specify two conditions,
# either at the top or at the bottom, and they can either be Dirichlet
# (on `ϑ_l`) or Neumann (on `-K∇h`). Note that fluxes are supplied as
# scalars, inside the code they are multiplied by ẑ. The two conditions
# not supplied must be set to `nothing`.

heaviside(x) = 0.5 * (sign(x) + 1)
surface_flux = (aux, t) -> eltype(aux)(((3.3e-4)/60) * heaviside(200*60-t)) # check units -- 
# m3 right? eltype(aux)(0.0)
bottom_flux = (aux, t) -> eltype(aux)(0.0)
surface_state = nothing
bottom_state = nothing

# Define the initial state function. The default for `θ_i` is zero.
ϑ_l0 = (aux) -> eltype(aux)(0.2 + 0.2 * heaviside((-1.0)-aux.z))

# ϑ_l0 = (aux) -> eltype(aux)(0.0 + 0.05 * heaviside((-1.0)-aux.z))
# yields a negative effective saturation : ERROR: LoadError: TaskFailedException:
# Effective saturation is negative
# ϑ_l0 = (aux) -> eltype(aux)(0.4 + 0.05 * heaviside((-1.0)-aux.z))
# this is near saturation, does not yield an error!


# Create the SoilWaterModel. The defaults are a temperature independent
# viscosity, and no impedance factor due to ice. We choose to make the
# hydraulic conductivity a function of the moisture content `ϑ_l`,
# and employ the vanGenuchten hydraulic model with `n` = 2.0. The van
# Genuchten parameter `m` is calculated from `n`, and we use the default
# value for `α`.
soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = 2.0,  α = 1.0), # changed alpha to 1, n to 2
    #but need to verify units for alpha (KMDyes)
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

# Specify the polynomial order and resolution.
N_poly = 5;
xres = FT(80)
yres = FT(80)
zres = FT(0.2)
# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-5);
xmax = FT(400)
ymax = FT(320)


# The atmosphere is dry and the flow impinges against a witch of Agnesi mountain of heigh $h_{m}=1$m
# and base parameter $a=10000m$ and centered on $x_{c} = 120km$ in a 2D domain
# $\Omega = 240km \times 50km$. The mountain is defined as
#
# ```math
# z = \frac{h_m}{1 + \frac{x - x_c}{a}}
# ```
#
# The 2D problem is setup in 3D by using 1 element in the y direction.
# To damp the upward moving gravity waves, a Reyleigh absorbing layer is added at $z = 15000 m$.
# function setmax(f, xmax, ymax, zmax)
#     function setmaxima(xin, yin, zin)
#         return f(xin, yin, zin; xmax = xmax, ymax = ymax, zmax = zmax)
#     end
#     return setmaxima
# end
# # Define a warping function to build an analytic topography (we want a 2D slope, in 3D):
function warp_maxwell_slope(xin, yin, zin; xmax = 400, topo_max = 0.2)
    zdiff = - topo_max / xmax * xin + topo_max
    x, y, z = xin, yin, zin + zdiff * heaviside(zin-eltype(zin)(-0.001))
    return x, y, z
 end

topo_max = FT(0.2)
# Create the driver configuration.
driver_config = ClimateMachine.MultiColumnLandModel(
    "LandModel",
    N_poly,
    (xres,yres,zres),
    xmax,
    ymax,
    zmax,
    param_set,
    m;
    zmin = zmin,
    #numerical_flux_first_order = CentralNumericalFluxFirstOrder(),now the default for us
    meshwarp = (x...) -> warp_maxwell_slope(x...;xmax = xmax, topo_max = topo_max),
);

# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60*4) # * 300)
dt = FT(3); #5

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# Determine how often you want output.
const n_outputs = 5;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# Create a place to store this output.
mygrid = solver_config.dg.grid;
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;
x_ind = varsindex(vars_state(m, Auxiliary(), FT), :x)
y_ind = varsindex(vars_state(m, Auxiliary(), FT), :y)
z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)

x = aux[:, x_ind, :][:]
y = aux[:, y_ind, :][:]
z = aux[:, z_ind, :][:]
ϑ_l = Q[:, ϑ_l_ind, :][:]

all_data = [Dict{String, Array}("ϑ_l" => ϑ_l, "x" => x, "y" =>y, "z" => z)]
time_data = FT[0] # store time data

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    x = aux[:, x_ind, :][:]
    y = aux[:, y_ind, :][:]
    z = aux[:, z_ind, :][:]
    ϑ_l = Q[:, ϑ_l_ind, :][:]
    dons = Dict{String, Array}("ϑ_l" => ϑ_l, "x" => x, "y" =>y, "z" => z)
    push!(all_data, dons)
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

# # Run the integration
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

# Get the final state and create plots:
x = aux[:, x_ind, :][:]
y = aux[:, y_ind, :][:]
z = aux[:, z_ind, :][:]
ϑ_l = Q[:, ϑ_l_ind, :][:]
dons = Dict{String, Array}("ϑ_l" => ϑ_l, "x" => x, "y" =>y, "z" => z)
push!(all_data, dons)
push!(time_data, gettime(solver_config.solver))


# # Create some plots

output_dir = @__DIR__;

t = time_data ./ (60);

plot(
    all_data[1]["ϑ_l"],
    all_data[1]["z"],
    label = string("t = ", string(t[1]), "min"),
    xlim = [0.0, 0.501],
    ylabel = "z",
    xlabel = "ϑ_l",
    legend = :bottomleft,
    title = "Maxwell Infiltration",
);
plot!(
    all_data[4]["ϑ_l"],
    all_data[4]["z"],
    label = string("t = ", string(t[4]), "min"),
);
plot!(
    all_data[6]["ϑ_l"],
    all_data[6]["z"],
    label = string("t = ", string(t[6]), "min"),
);

# save the output.
savefig(joinpath(output_dir, "maxwell_test_infiltation.png"))
# # References
