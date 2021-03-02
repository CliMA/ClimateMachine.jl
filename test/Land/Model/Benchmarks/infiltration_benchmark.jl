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

using JLD2
using Plots

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
nelem_vert = 50

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

function setupsolver(nlalg, driver_config, t0, timeend, dt)
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

    nlsolver = nlalg(Q, vdg)
    ode_solver = ARK548L2SA2KennedyCarpenter(
        dg,
        vdg,
        NonLinearBackwardEulerSolver(
            nlsolver;
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
    return solver_config
end

tol = 1e-12
algs = (
    N_FD_30 = (Q, f) -> JacobianFreeNewtonKrylovAlgorithm(
                BatchedGeneralizedMinimalResidualAlgorithm(;
                    preconditioner = ColumnwiseLUPreconditioner(f, Q, 100),
                    M = 30, atol = tol, rtol = tol,
                );
            maxiters = Int(1e3), atol = tol, rtol = tol, autodiff=false),
    N_AD_30 = (Q, f) -> JacobianFreeNewtonKrylovAlgorithm(
                BatchedGeneralizedMinimalResidualAlgorithm(;
                    preconditioner = ColumnwiseLUPreconditioner(f, Q, 100),
                    M = 30, atol = tol, rtol = tol,
                );
            maxiters = Int(1e3), atol = tol, rtol = tol, autodiff=true),
    # OldFD_30 = (Q, f) -> JacobianFreeNewtonKrylovSolver(Q, BatchedGeneralizedMinimalResidual(
    #             f,
    #             Q;
    #             max_subspace_size = 30,
    #             atol = tol,
    #             rtol = tol,
    #         ); M = Int(1e3), tol = tol),
    # P = (Q, f) -> StandardPicardAlgorithm(; atol = tol, rtol = tol, maxiters = Int(1e4)),
    # PA_1_1 = (Q, f) -> AndersonAccelerationAlgorithm(
    #             StandardPicardAlgorithm(; atol = tol, rtol = tol, maxiters = Int(1e4));
    #             depth = 1
    #         ),
    PA_2_1 = (Q, f) -> AndersonAccelerationAlgorithm(
            StandardPicardAlgorithm(; atol = tol, rtol = tol, maxiters = Int(1e4));
            depth = 2
        ),
    PA_3_1 = (Q, f) -> AndersonAccelerationAlgorithm(
            StandardPicardAlgorithm(; atol = tol, rtol = tol, maxiters = Int(1e4));
            depth = 3
        ),
    PA_4_1 = (Q, f) -> AndersonAccelerationAlgorithm(
            StandardPicardAlgorithm(; atol = tol, rtol = tol, maxiters = Int(1e4));
            depth = 4
        ),
    # PA_1_8 = (Q, f) -> AndersonAccelerationAlgorithm(
    #         StandardPicardAlgorithm(; atol = tol, rtol = tol, maxiters = Int(1e4));
    #         depth = 1, ω = 0.8,
    #     ),
    PA_2_8 = (Q, f) -> AndersonAccelerationAlgorithm(
            StandardPicardAlgorithm(; atol = tol, rtol = tol, maxiters = Int(1e4));
            depth = 2, ω = 0.8,
        ),
    PA_3_8 = (Q, f) -> AndersonAccelerationAlgorithm(
            StandardPicardAlgorithm(; atol = tol, rtol = tol, maxiters = Int(1e4));
            depth = 3, ω = 0.8,
        ),
    PA_4_8 = (Q, f) -> AndersonAccelerationAlgorithm(
            StandardPicardAlgorithm(; atol = tol, rtol = tol, maxiters = Int(1e4));
            depth = 4, ω = 0.8,
        ),
    N_FD_15 = (Q, f) -> JacobianFreeNewtonKrylovAlgorithm(
            BatchedGeneralizedMinimalResidualAlgorithm(;
                preconditioner = ColumnwiseLUPreconditioner(f, Q, 100),
                M = 15, maxrestarts = 21, atol = tol, rtol = tol,
            );
        maxiters = Int(1e3), atol = tol, rtol = tol, autodiff=false),
    N_AD_15 = (Q, f) -> JacobianFreeNewtonKrylovAlgorithm(
                BatchedGeneralizedMinimalResidualAlgorithm(;
                    preconditioner = ColumnwiseLUPreconditioner(f, Q, 100),
                    M = 15, maxrestarts = 21, atol = tol, rtol = tol,
                );
            maxiters = Int(1e3), atol = tol, rtol = tol, autodiff=true),
    # OldFD_15 = (Q, f) -> JacobianFreeNewtonKrylovSolver(Q, BatchedGeneralizedMinimalResidual(
    #             f,
    #             Q;
    #             max_subspace_size = 15,
    #             atol = tol,
    #             rtol = tol,
    #         ); M = Int(1e3), tol = tol),
)

# precompile stuff to not screw with timing - must be Newton AD/FD + GMRES, Picard, Picard + Anderson
for key in keys(algs)[1:4]
    @info string(key)
    solver_config = setupsolver(algs[key], driver_config, 0.0, 1.0, 1.0)
    @elapsed ClimateMachine.invoke!(solver_config)
    GC.gc()
end

t0 = FT(0)
timeend = FT(60 * 60)

dts = FT[0.5, 1, 2, 5, 7.5, 10, 15, 20]

function benchmark_solver!(nlalg, driver_config, t0, timeend, dt, solvetimes, rmse, finalstates, i, j)
    solver_config = setupsolver(nlalg, driver_config, t0, timeend, dt)

    dons_arr = Dict[dict_of_nodal_states(solver_config; interp = true)]  # store initial condition at ``t=t0``
    solvetimes[i, j] = @elapsed ClimateMachine.invoke!(solver_config)
    push!(dons_arr, dict_of_nodal_states(solver_config; interp = true)); # store final condition at ``t=timeend``

    # interpolation
    current_profile = dons_arr[2]["soil.water.ϑ_l"][:]
    simulation_z = dons_arr[2]["z"][:]
    true_profile = true_moisture_continuous.(simulation_z)
    rmse[i, j] = sqrt(sum((true_profile .- current_profile).^2.0))
    finalstates[i, j] = dons_arr[2]
end

# load "true" data and create interpolation function
@load joinpath(@__DIR__, "infiltration_truth.jld2") truedata # a dict_of_nodal_states
true_moisture_continuous = Spline1D(truedata["z"][:], truedata["soil.water.ϑ_l"][:])

N = length(algs)
M = length(dts)
rmse = zeros(N, M)
solvetimes = zeros(N, M)
finalstates = Array{Any, 2}(nothing, N, M)
for (i, nlalg) in enumerate(algs)
    for (j, dt) in enumerate(dts)
        @info "Solver: $(string(keys(algs)[i])), dt: $dt"
        benchmark_solver!(nlalg, driver_config, t0, timeend, dt, solvetimes, rmse, finalstates, i, j)
        GC.gc()
    end
end
save some plots and data
labels = [label for label in string.(keys(algs))]
labels = reshape(labels, (1, N))
@save joinpath(@__DIR__, "benchmark_data.jld2") solvetimes rmse finalstates dts labels
timeplot = plot(dts, [solvetimes[i, :] for i in 1:N],
    title="Solve Times", xlabel="dt", ylabel="total solve time",
    markershape=:circle, legend=:outertopright, label = labels)
png(timeplot, joinpath(@__DIR__, "solvetimes"))
errplot = plot(dts, [rmse[i, :] for i in 1:N],
    title="RMSE", xlabel="dt", ylabel="rmse",
    markershape=:circle, legend=:outertopright, label = labels)
png(errplot, joinpath(@__DIR__, "rmse"))


# as long as the solution is "good" rmse < 1e-4, what is the tradeoff
#between time to solution (N steps will vary)
# or # of f calls
