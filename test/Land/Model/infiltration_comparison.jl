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

using Logging: disable_logging, Warn
disable_logging(Warn)

ClimateMachine.init()
FT = Float64

function init_soil_water!(land, state, aux, localgeo, time)
    state.soil.water.ϑ_l = land.soil.water.initialϑ_l(aux)
    state.soil.water.θ_i = land.soil.water.initialθ_i(aux)
end

soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.4,
    Ksat = 6.94e-6 / 60,
    S_s = 5e-4,
)

sigmoid(x, offset, width) =
    (exp((x - offset) / width) / (one(typeof(x)) + exp((x - offset) / width)))
ϑ_l0 = (aux) ->
    (FT = eltype(aux); FT(0.4) - FT(0.025) * sigmoid(aux.z, FT(-1), FT(0.02)))
# Fluxes are multiplied by ẑ (normal to the surface, -normal to the bottom).
bottom_flux_value = FT(0)
surface_state_value = FT(soil_param_functions.porosity)
surface_state = (aux, t) -> surface_state_value
bottom_flux = (aux, t) -> bottom_flux_value
bc = GeneralBoundaryConditions(
    Dirichlet(surface_state = surface_state, bottom_state = nothing),
    Neumann(surface_flux = nothing, bottom_flux = bottom_flux),
)
soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = FT(2), α = FT(100)),
    initialϑ_l = ϑ_l0,
    boundaries = bc,
)

soil_heat_model = PrescribedTemperatureModel()

m = LandModel(
    param_set,
    SoilModel(soil_param_functions, soil_water_model, soil_heat_model),
    source = (),
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

timestart = FT(0)
timeend = FT(60)
dt = FT(60)

old_newton_fd(Q, vdg, M, tol, freq) =
    NonLinearBackwardEulerSolver(
        JacobianFreeNewtonKrylovSolver(
            Q,
            BatchedGeneralizedMinimalResidual(
                vdg,
                Q;
                max_subspace_size = M,
                atol = tol,
                rtol = tol,
            );
            tol = tol,
        );
        preconditioner_update_freq = freq,
    )
new_newton_fd(Q, vdg, M, tol, freq) =
    NonLinearBackwardEulerSolver(
        JacobianFreeNewtonKrylovAlgorithm(
            BatchedGeneralizedMinimalResidualAlgorithm(;
                preconditioner = freq > 0 ?
                    ColumnwiseLUPreconditioner(vdg, Q, freq) : NoPreconditioner(),
                M = M,
                atol = tol,
                rtol = tol,
            );
            atol = tol,
            rtol = tol,
        );
        preconditioner_update_freq = freq,
    )

function HEVI_setup(nonlinear_solver_fun, M, tol, freq)
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
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

    ode_solver = ARK548L2SA2KennedyCarpenter(
        dg,
        vdg,
        nonlinear_solver_fun(Q, vdg, M, tol, freq),
        Q;
        dt = dt,
        t0 = timestart,
        split_explicit_implicit = false,
        variant = NaiveVariant(),
    )
    solver_config.solver = ode_solver

    return solver_config
end

repeats = 50
for M in (1, 2, 20, 200), tol in (1e0, 1e-6, 1e-12), freq in (-1, 5)
    comparison = true
    time_old = 0.
    time_new = 0.
    bytes_old = 0.
    bytes_new = 0.
    for i in 1:(repeats + 1)
        solver_config_old = HEVI_setup(old_newton_fd, M, tol, freq)
        solver_config_new = HEVI_setup(new_newton_fd, M, tol, freq)
        stats_old = @timed ClimateMachine.invoke!(solver_config_old)
        stats_new = @timed ClimateMachine.invoke!(solver_config_new)
        if i > 1
            comparison &= solver_config_old.Q == solver_config_new.Q
            time_old += stats_old.time
            time_new += stats_new.time
            bytes_old += stats_old.bytes
            bytes_new += stats_new.bytes
        end
    end
    time_old /= repeats
    time_new /= repeats
    bytes_old /= repeats
    bytes_new /= repeats
    println("$M, $tol, $freq: $comparison; ($time_old, $time_new), ($bytes_old, $bytes_new); $(time_new/time_old), $(bytes_new/bytes_old)")
end

# Sample output:
# 1, 1.0, -1: true; (0.006874286080000001, 0.00679876792), (567440.0, 571136.0); 0.9890143995869312, 1.0065134639785704
# 1, 1.0, 5: true; (0.006926923960000002, 0.00724784012), (567440.0, 595020.8); 1.0463288123058878, 1.0486056675595659
# 1, 1.0e-6, -1: true; (0.06423785213999998, 0.06463986399999999), (6.160864e6, 5.2074272e6); 1.0062581771744774, 0.845243004877238
# 1, 1.0e-6, 5: true; (0.06777153574000001, 0.06672135393999999), (5.8934048e6, 5.50315776e6); 0.9845040873202437, 0.9337824138603206
# 1, 1.0e-12, -1: true; (2.2083850298200005, 2.28433828798), (2.5621448768e8, 1.950126464e8); 1.034393123089677, 0.7611304425671734
# 1, 1.0e-12, 5: true; (2.84137919208, 2.923891963920001), (3.3413527648e8, 2.6041010752e8); 1.0290396903271466, 0.7793553265711144
# 2, 1.0, -1: true; (0.007183657900000002, 0.0071269601199999985), (567440.0, 571136.0); 0.99210739420094, 1.0065134639785704
# 2, 1.0, 5: true; (0.0069839481400000005, 0.006884829979999999), (567440.0, 571136.0); 0.9858077182113781, 1.0065134639785704
# 2, 1.0e-6, -1: true; (0.07368981999999999, 0.07317009018), (7.3379584e6, 6.2469056e6); 0.9929470608016142, 0.8513138477318158
# 2, 1.0e-6, 5: true; (0.06535899394, 0.06520448592000001), (5.8916896e6, 5.5104e6); 0.9976360098176874, 0.9352834881185866
# 2, 1.0e-12, -1: true; (2.3592368880000008, 2.41944965196), (2.681896064e8, 2.0696607104e8); 1.0255221356813573, 0.7717154807681614
# 2, 1.0e-12, 5: true; (1.8542241920200002, 1.8965811280199998), (2.105570432e8, 1.670204832e8); 1.0228434814852976, 0.7932315189350075
# 20, 1.0, -1: true; (0.007080092020000001, 0.006934565859999999), (567440.0, 571136.0); 0.9794457247746334, 1.0065134639785704
# 20, 1.0, 5: true; (0.00687277398, 0.0073108460600000025), (567440.0, 602790.4); 1.0637402133803333, 1.0622980403214437
# 20, 1.0e-6, -1: true; (0.0728839521, 0.07182684188000002), (7.2339168e6, 6.1492288e6); 0.9854959810830569, 0.850055228724776
# 20, 1.0e-6, 5: true; (0.06521166594, 0.06453847002), (5.8936544e6, 5.4977856e6); 0.9896767562935842, 0.9328313516313409
# 20, 1.0e-12, -1: true; (0.26537025196, 0.26730751816000003), (2.85111968e7, 2.29658432e7); 1.0073002387633563, 0.8055026017006763
# 20, 1.0e-12, 5: true; (0.28639890203999996, 0.28732737606000003), (2.91546144e7, 2.53198656e7); 1.003241890989758, 0.8684685467834554
# 200, 1.0, -1: true; (0.007213292040000001, 0.0068273597200000005), (579420.8, 571136.0); 0.9464970615552672, 0.9857015833742937
# 200, 1.0, 5: true; (0.006920083880000003, 0.006815986100000001), (567440.0, 571136.0); 0.9849571505482962, 1.0065134639785704
# 200, 1.0e-6, -1: true; (0.0732111239, 0.07161358000000002), (7.2261792e6, 6.1371392e6); 0.9781789458363994, 0.8492924172154491
# 200, 1.0e-6, 5: true; (0.06459200012, 0.06451483192), (5.8837088e6, 5.5018176e6); 0.9988052978719247, 0.9350934566986048
# 200, 1.0e-12, -1: true; (0.26800593396000005, 0.26792264604000005), (2.85153888e7, 2.29633792e7); 0.999689231060039, 0.8052977766166737
# 200, 1.0e-12, 5: true; (0.28964569594, 0.2922204639999999), (2.91499424e7, 2.5316896e7); 1.00888937103534, 0.868505867099072

# The new solver is roughly as fast as the old one, but with fewer memory
# leaks. Great success! Still, we could probably speed it up even more.