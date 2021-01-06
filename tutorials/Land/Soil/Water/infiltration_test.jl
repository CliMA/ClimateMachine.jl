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
sigmoid(x, offset, width) = typeof(x)(exp((x-offset)/width)/(1+exp((x-offset)/width)))
precip_of_t = (t) -> eltype(t)(-((3.3e-4)/60) * (1-sigmoid(t, 200*60,10)))#heaviside(200*60-t))
# Define the initial state function. The default for `θ_i` is zero.
ϑ_l0 = (aux) -> eltype(aux)(0.399- 0.05 * sigmoid(aux.z, -1.0,0.02))#heaviside((-0.5)-aux.z))
zres = FT(0.05)
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
    hydraulics = vanGenuchten{FT}(n = 2.0,  α = 100.0), # changed alpha to 1, n to 2
    #but need to verify units for alpha (KMDyes)
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
N_poly = 1;
nelem_vert = 40;

# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-3);

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
    #meshwarp = (x...) -> warp_maxwell_slope(x...),#setmax(warp_agnesi, xmax, ymax, zmax),
);

# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60*100) # * 300)
dt = FT(3); #5

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# Determine how often you want output.
const n_outputs = 500;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# Create a place to store this output.
mygrid = solver_config.dg.grid;
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;
grads = solver_config.dg.state_gradient_flux

x_ind = varsindex(vars_state(m, Auxiliary(), FT), :x)
y_ind = varsindex(vars_state(m, Auxiliary(), FT), :y)
z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
K∇h_vert_ind = varsindex(vars_state(m, GradientFlux(), FT), :soil, :water)[3]

x = aux[:, x_ind, :][:]
y = aux[:, y_ind, :][:]
z = aux[:, z_ind, :][:]
ϑ_l = Q[:, ϑ_l_ind, :][:]
K∇h_vert = zeros(length(ϑ_l)) .+ FT(NaN)

all_data = [Dict{String, Array}("ϑ_l" => ϑ_l, "K∇h" => K∇h_vert, "x" => x, "y" =>y, "z" => z)]
time_data = FT[0] # store time data

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    x = aux[:, x_ind, :][:]
    y = aux[:, y_ind, :][:]
    z = aux[:, z_ind, :][:]
    ϑ_l = Q[:, ϑ_l_ind, :][:]
    K∇h_vert = grads[:, K∇h_vert_ind, :][:]

    dons = Dict{String, Array}("ϑ_l" => ϑ_l, "K∇h" => K∇h_vert, "x" => x, "y" =>y, "z" => z)
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
K∇h_vert = grads[:, K∇h_vert_ind, :][:]

dons = Dict{String, Array}("ϑ_l" => ϑ_l, "K∇h" => K∇h_vert, "x" => x, "y" =>y, "z" => z)
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
xs
