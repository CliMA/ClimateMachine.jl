# Test that Richard's equation agrees with solution from Bonan's book,
# simulation 8.2
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Dierckx
using Test
using Pkg.Artifacts
using DelimitedFiles

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
using ClimateMachine.SystemSolvers
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state
using ClimateMachine.ArtifactWrappers

ClimateMachine.init()
FT = Float64

function init_soil_water!(land, state, aux, localgeo, time)
    state.soil.water.ϑ_l = eltype(aux)(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_i = eltype(aux)(land.soil.water.initialθ_i(aux))
end

soil_heat_model = PrescribedTemperatureModel()

soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.4,
    Ksat = 6.94e-6/ (60), #m/s
    S_s = 5e-4,
)
# Keep in mind that what is passed is aux⁻
# Fluxes are multiplied by ẑ (normal to the surface, -normal to the bottom,
# where normal point outs of the domain.)
sigmoid(x, offset, width) = typeof(x)(exp((x-offset)/width)/(1+exp((x-offset)/width)))
bottom_flux_value = FT(0.0)
surface_value = FT(soil_param_functions.porosity)

surface_state = (aux, t) -> surface_value
bottom_flux = (aux, t) -> bottom_flux_value
ϑ_l0 = (aux) -> eltype(aux)(0.4- 0.025 * sigmoid(aux.z, -1.0,0.02))

bc = GeneralBoundaryConditions(
    Dirichlet(surface_state = surface_state, bottom_state = nothing),
    Neumann(surface_flux = nothing, bottom_flux = bottom_flux),
)
vg_a = FT(100.0)
vg_n = FT(2.0)
soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = vg_n, α = vg_a),
    initialϑ_l = ϑ_l0,
    boundaries = bc,
)

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
sources = ()
m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_prognostic = init_soil_water!,
)


N_poly = 1
nelem_vert = 30

# Specify the domain boundaries
zmax = FT(0)
zmin = FT(-2)

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

t0 = FT(0)
timeend = FT(10)#FT(60 * 60)

dt = FT(1)

solver_config = ClimateMachine.SolverConfiguration(
    t0,
    timeend,
    driver_config,
    ode_dt = dt,
)

dg = solver_config.dg
Q = solver_config.Q
vdg = DGModel(
    driver_config;
    state_auxiliary = dg.state_auxiliary,
    direction = VerticalDirection(),
)

linearsolver = BatchedGeneralizedMinimalResidual(
    vdg,
    Q;
    max_subspace_size = 30,
    atol = -1.0,
    rtol = 1e-9,
)
nonlinearsolver =
    JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-9)

# nonlinearsolver = JacobianFreeNewtonKrylovAlgorithm(
#             BatchedGeneralizedMinimalResidualAlgorithm(;
#                 preconditioner = ColumnwiseLUPreconditioner(vdg, Q, 100),
#                 M = 30, atol = eps(Float64), rtol = 1e-9,
#             );
#             atol = 1e-9, rtol = 1e-9, autodiff=false)

# nonlinearsolver = StandardPicardAlgorithm(; atol = 1e-9, rtol = 1e-9, maxiters = 1000)
# nonlinearsolver = AndersonAccelerationAlgorithm(
#             StandardPicardAlgorithm(; atol = 1e-9, rtol = 1e-9, maxiters = 1000);
#             depth = 1
#         )


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
    t0 = t0,
    split_explicit_implicit = false,
    variant = NaiveVariant(),
)
solver_config.solver = ode_solver

n_outputs = 5
every_x_simulation_time = timeend / n_outputs
time_data = FT[0]
data_avg = Dict[Dict([k => Dict() for k in 0:n_outputs]...),]
data_avg[1] = get_horizontal_mean(
    driver_config.grid,
    solver_config.Q,
    vars_state(m, Prognostic(), FT),
)
data_var = Dict[Dict([k => Dict() for k in 0:n_outputs]...),]
data_var[1] = get_horizontal_variance(
    driver_config.grid,
    solver_config.Q,
    vars_state(m, Prognostic(), FT),
)
callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    state_vars_avg = get_horizontal_mean(
        driver_config.grid,
        solver_config.Q,
        vars_state(m, Prognostic(), FT),
    )
    state_vars_var = get_horizontal_variance(
        driver_config.grid,
        solver_config.Q,
        vars_state(m, Prognostic(), FT),
    )
    push!(time_data, gettime(solver_config.solver))
    push!(data_avg, state_vars_avg)
    push!(data_var, state_vars_var)
    nothing
end

mygrid = solver_config.dg.grid 
dons_arr = Dict[dict_of_nodal_states(solver_config; interp = true)]  # store initial condition at ``t=0``
solvetime = @time ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))
push!(dons_arr, dict_of_nodal_states(solver_config; interp = true));

current_profile = dons_arr[2]["soil.water.ϑ_l"][:]
current_profile = dons_arr[1]["z"][:]
# the_truth_interp_function = #
# the_truth = the_truth_interp_function(current_z)
# rmse = sqrt.((the_truth .- current_profile).^2.0)
# check rmse < 1e-4


# as long as the solution is "good" rmse < 1e-4, what is the tradeoff
#between time to solution (N steps will vary)
# or # of f calls

#Npoly = 1

# const clima_dir = dirname(dirname(pathof(ClimateMachine)))
# include(joinpath(clima_dir, "docs", "plothelpers.jl"))
# output_dir = @__DIR__
# mkpath(output_dir)
# z_scale = 100 # convert from meters to cm
# z_label = "z [cm]"
# z = get_z(driver_config.grid; z_scale = z_scale)
# export_plot(
#     z,
#     time_data,
#     data_avg,
#     ("soil.water.ϑ_l"),
#     joinpath(output_dir, "solution_vs_time_ϑ_l_avg.png");
#     xlabel = "Horizontal mean of ϑ_l",
#     ylabel = z_label,
# );
# export_plot(
#     z,
#     time_data,
#     data_var,
#     ("soil.water.ϑ_l"),
#     joinpath(output_dir, "solution_vs_time_ϑ_l_var.png");
#     xlabel = "Horizontal variance of ϑ_l",
#     ylabel = z_label,
# );
