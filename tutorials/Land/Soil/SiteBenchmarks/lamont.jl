# # Lamont
# to do - compare STAMP to swats, get soil characeteristics for the soil here. make sure IC are correct.
# # Preliminary setup

using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Dierckx
using DelimitedFiles
using Plots
using Dates
using NCDatasets

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
using ClimateMachine.SystemSolvers
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
θ_r = FT(0.0)
ν = FT(0.49)
vgn = FT(1.4)
vgα = FT(2.0)
loam_ks = 1 ./3600 ./ 100 # from Bonan
wpf = WaterParamFunctions(FT; Ksat = loam_ks, S_s = 1e-4, θ_r = θ_r);
soil_param_functions = SoilParamFunctions(FT; porosity = ν, water = wpf);

### Read in flux data
cutoff1 = DateTime(2016,04,01)
cutoff2 = DateTime(2016,07,01)
filepath = "data/lamont/arms_flux/sgparmbeatmC1.c1.20160101.003000.nc"
ds = Dataset(filepath)
times = ds["time"][:]
p = ((times .<=cutoff2) .+ (times .>= cutoff1)) .== 2
precip_rate = ds["precip_rate_sfc"][p] # mm/hr
lhf_baebbr = ds["latent_heat_flux_baebbr"][p]
times = ds["time"][p]
close(ds)
keep1 = (typeof.(lhf_baebbr) .!= Missing)
lhf = lhf_baebbr[keep1]
keep2 = (typeof.(precip_rate) .!= Missing)
precip_rate = precip_rate[keep2] ./1000 ./ 3600
Lv = 2.5008e6
ρ = 1e3
evap_rate = lhf ./ Lv ./ρ

foo = (times .-times[1])./1000
seconds = [k.value for k in foo]

# Create interpolating function for evap_rate. same for P. then we have a net water flux
E = Spline1D(seconds[keep1], evap_rate)
P  = Spline1D(seconds[keep2], precip_rate)
function net_water_flux(t::Real, P::Spline1D, E::Spline1D)
    net = -P(t) +E(t)
    return net
end
incident = (t) -> net_water_flux(t, P, E)


bottom_flux = (aux, t) -> aux.soil.water.K * eltype(aux)(-1)
surface_flux = (aux, t) -> -incident(t)
N_poly = 1;
nelem_vert = 20;

# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-1);
Δ = FT((zmax-zmin)/nelem_vert/2)
bc = LandDomainBC(
bottom_bc = LandComponentBC(
    soil_water = Neumann(bottom_flux)
),
surface_bc = LandComponentBC(
    soil_water = SurfaceDrivenWaterBoundaryConditions(FT;
                                                      precip_model = DrivenConstantPrecip{FT}(incident),
                                                      runoff_model = CoarseGridRunoff{FT}(Δ),
                                                      )
)
)

depths = [5, 10,20,50,75] .* (-0.01) # m
data = readdlm("./data/lamont/stamps_swc_depth.txt", '\t', String)
ts = DateTime.(data[:,1], "yyyymmdd")
soil_data = tryparse.(Float64, data[:, 2:end-1])
keep = ((ts .<= cutoff2) .+ (ts .>= cutoff1)) .==2
swc = FT.(soil_data[keep,:][1,:]).*0.01
θ = Spline1D(reverse(depths), reverse(swc), k=1)


    
ϑ_l0 = aux -> θ(aux.z)


soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten(FT; n = vgn, α = vgα),
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
timeend = FT(60 *60 * 24*60)+t0
dt = FT(100);

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

    dg = solver_config.dg
    Q = solver_config.Q

    vdg = DGModel(
        driver_config;
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

    """
    N(q)(Q) = Qhat  => F(Q) = N(q)(Q) - Qhat
    F(Q) == 0
    ||F(Q^i) || / ||F(Q^0) || < tol
    """
    nonlinearsolver =
        JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-6)

    ode_solver = ARK2GiraldoKellyConstantinescu(
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

    solver_config.solver = ode_solver

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
N = length(dons_arr)

mask = z .== depths[1]
l1 = [dons_arr[k]["soil.water.ϑ_l"][mask][1] for k in 1:N]
mask = z .== depths[3]
l3 = [dons_arr[k]["soil.water.ϑ_l"][mask][1] for k in 1:N]
mask = z .== depths[2]
l2 = [dons_arr[k]["soil.water.ϑ_l"][mask][1] for k in 1:N]
mask = z .== depths[4]
l4 = [dons_arr[k]["soil.water.ϑ_l"][mask][1] for k in 1:N]
mask = z .== depths[5]
l5 = [dons_arr[k]["soil.water.ϑ_l"][mask][1] for k in 1:N]

T = typeof(cutoff2 - cutoff1)
steps = T.(time_data*1000)
times = cutoff1 .+ steps
soil_data = soil_data ./ 100
plot1 = plot(times,l1, label = "", color = "red", title = "Layer 1")
scatter!(ts[keep], soil_data[keep,1], ms = 2, color = "blue", label = "")
plot!(ylim = [0,0.4])

plot2 = plot(times,l2, label = "", color = "red", title= "Layer 2")
scatter!(ts[keep], soil_data[keep,2], ms = 2, color = "blue", label = "")
plot!(ylim = [0,0.4])
plot3 = plot(times,l3, label = "", color = "red", title = "Layer 3")
scatter!(ts[keep], soil_data[keep,3], ms = 2, color = "blue", label = "")
plot!(ylim = [0,0.4])
plot4 = plot(times,l4, label = "", color = "red", title = "Layer 4")
scatter!(ts[keep], soil_data[keep,4], ms = 2, color = "blue", label = "")
plot!(ylim = [0,0.4])

plot5 = plot(times,l5, label = "", color = "red", title = "Layer 5")
scatter!(ts[keep], soil_data[keep,5], ms = 2, color = "blue", label = "")
plot!(ylim = [0,0.4])
