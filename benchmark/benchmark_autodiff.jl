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

using Logging
disable_logging(Logging.Warn)

ClimateMachine.init()
FT = Float64

function init_soil_water!(land, state, aux, coordinates, time)
    myfloat = eltype(aux)
    state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_i = myfloat(land.soil.water.initialθ_i(aux))
end

soil_heat_model = PrescribedTemperatureModel()

soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.495,
    Ksat = 0.0443 / (3600 * 100),
    S_s = 1e-3,
)
# Keep in mind that what is passed is aux⁻
# Fluxes are multiplied by ẑ (normal to the surface, -normal to the bottom,
# where normal point outs of the domain.)
surface_value = FT(0.494)
bottom_flux_multiplier = FT(1.0)
initial_moisture = FT(0.24)

surface_state = (aux, t) -> surface_value
bottom_flux = (aux, t) -> aux.soil.water.K * bottom_flux_multiplier
ϑ_l0 = (aux) -> initial_moisture

soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = Haverkamp{FT}(),
    initialϑ_l = ϑ_l0,
    dirichlet_bc = Dirichlet(
        surface_state = surface_state,
        bottom_state = nothing,
    ),
    neumann_bc = Neumann(surface_flux = nothing, bottom_flux = bottom_flux),
)

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
sources = ()
m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_prognostic = init_soil_water!,
)


N_poly = 5
nelem_vert = 10

# Specify the domain boundaries
zmax = FT(0)
zmin = FT(-1)

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
timeend = FT(60 * 60 * 24)

function ϑ_l_solve(diffmode, dt)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )

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

    nonlinearsolver =
        JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-9)

    ode_solver = ARK548L2SA2KennedyCarpenter(
        dg,
        vdg,
        NonLinearBackwardEulerSolver(
            nonlinearsolver;
            isadjustable = true,
            preconditioner_update_freq = 100,
            mode = diffmode(),
        ),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = false,
        variant = NaiveVariant(),
    )

    solver_config.solver = ode_solver

    mygrid = solver_config.dg.grid
    Q = solver_config.Q
    aux = solver_config.dg.state_auxiliary

    time_info = @timed ClimateMachine.invoke!(solver_config)
    ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
    ϑ_l = Array(Q[:, ϑ_l_ind, :][:])

    return ϑ_l, time_info
end

dt_fine = FT(10) # step size for "exact" solution
step_sizes = FT.([50, 100.1, 150, 200, 250, 300])
N = length(step_sizes)
MSE = zeros(2, N)
runtimes = zeros(2, N)
allocs = zeros(2, N)

using Serialization: serialize

ϑ_l_exact,_ = ϑ_l_solve(AutoDiffMode, dt_fine)

for (i, diffmode) in enumerate([AutoDiffMode, FiniteDiffMode])
    for (j, dt) in enumerate(step_sizes)
        println(diffmode, dt)
        try
            ϑ_l, time_info = ϑ_l_solve(diffmode, dt)
            serialize("benchmark/solution_$(diffmode)_$(dt)", ϑ_l)
            MSE[i, j] = mean((ϑ_l_exact .- ϑ_l) .^ 2.0)
            runtimes[i, j] = time_info.time
            allocs[i, j] = time_info.malloc
        catch e
            MSE[i, j] = Inf
            runtimes[i. j] = Inf
        end
    end
end

serialize("benchmark/exactsolution", ϑ_l_exact)
serialize("benchmark/MSE", MSE)
serialize("benchmark/runtimes", runtimes)

