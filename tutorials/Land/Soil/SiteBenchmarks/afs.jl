# # Amaragosa Flats Shallow Site

# # Preliminary setup

using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Dierckx
using DelimitedFiles
using Plots

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

const FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

# # Set up the soil model

soil_heat_model = PrescribedTemperatureModel();
function Ks(z::F) where {F}
    if z < F(-0.75)
        return F(1e-9)
    else
        return F(1e-8)
    end
end

wpf = WaterParamFunctions(FT; Ksat = (aux) -> Ks(aux.z), S_s = 1e-4, θ_r = 0.15);
soil_param_functions = SoilParamFunctions(FT; porosity = 0.8, water = wpf);

file2 = "./data/amargosa/ET-AFS 30 min-Table 1.csv"
data2 = readdlm(file2, ',')
s = 7
columns = data2[6,:]
timestamp = data2[s:end,1]
t = Array(1:1:length(timestamp)) .* 30.0 * 60.0
ET = Float64.(data2[s:end, 2]) # this is total ET
P = Float64.(data2[s:end, 12])
I = (ET) .*1e-3 ./ 60.0 ./ 30.0
I_function = Spline1D(t, I)
#bottom_state = (aux, t) -> eltype(aux)(0.8);
bottom_flux = (aux, t) -> aux.soil.water.K * eltype(aux)(-1)
N_poly = 1;
nelem_vert = 30;

# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-2);
Δ = FT((zmax-zmin)/nelem_vert/2)
bc = LandDomainBC(
bottom_bc = LandComponentBC(
    soil_water = Neumann(bottom_flux)
),
surface_bc = LandComponentBC(
    soil_water = SurfaceDrivenWaterBoundaryConditions(FT;
                                                      precip_model = DrivenConstantPrecip{FT}(I_function),
                                                      runoff_model = CoarseGridRunoff{FT}(Δ),
                                                  )
)
)
                                                                         

# Define the initial state function. The default for `θ_i` is zero.
function f(z::F) where {F}
    max = F(0)
    min = F(-2)
    change = F(-0.5)
    inter1 = F(0.2)
    slope1 = (F(0.6)-inter1)/(change-max)
    inter2 = F(0.6)
    slope2 = (F(0.8)-inter2)/(min-F(-1))
    if z > F(-1)
        if z < F(-0.5)
            initial = inter2
        else
            initial = slope1*(z-max)+inter1
        end
    else
        initial = slope2*(z-F(-1))+inter2
    end
    
    return initial
end

    
ϑ_l0 = aux -> f(aux.z)


soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten(FT; n = 1.1, α = 8.0),
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
timeend = FT(60 * 60 * 24 * 120) + t0
dt = FT(60);

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# Determine how often you want output.
n_outputs = 120;

every_x_simulation_time = ceil(Int, (timeend-t0) / n_outputs);

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

