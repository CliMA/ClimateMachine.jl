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

# Sample output with 50 repeats from before batched GMRES optimizations:
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

# Same output after batched GMRES optimizations:
# 1, 1.0, -1: true; (0.00701758208, 0.0069684121600000004), (567440.0, 571136.0); 0.9929933245611572, 1.0065134639785704
# 1, 1.0, 5: true; (0.006742780099999999, 0.006700428160000001), (567440.0, 571136.0); 0.993718920182493, 1.0065134639785704
# 1, 1.0e-6, -1: true; (0.05915858791999999, 0.05790489208000002), (6.1761664e6, 5.1604416e6); 0.9788078809167091, 0.8355412185785667
# 1, 1.0e-6, 5: true; (0.06267377806, 0.06207745594000001), (5.8890656e6, 5.4925536e6); 0.9904853012143434, 0.9326697939992382
# 1, 1.0e-12, -1: true; (2.11634639214, 2.0895835559400004), (2.5621162944e8, 1.9176436288e8); 0.987354226935914, 0.7484608067913937
# 1, 1.0e-12, 5: true; (2.7567841580600003, 2.7256855499599997), (3.3413521024e8, 2.5648736768e8); 0.9887192444830772, 0.7676155036033834
# 2, 1.0, -1: true; (0.006815921980000001, 0.006749087899999999), (567440.0, 571136.0); 0.9901944182758966, 1.0065134639785704
# 2, 1.0, 5: true; (0.00683913796, 0.006768379899999997), (567440.0, 571136.0); 0.9896539504812091, 1.0065134639785704
# 2, 1.0e-6, -1: true; (0.07058067597999998, 0.06925090608000002), (7.3489024e6, 6.2168576e6); 0.9811595754569314, 0.8459572956092054
# 2, 1.0e-6, 5: true; (0.062320043799999995, 0.06166937396000002), (5.8851104e6, 5.4931616e6); 0.989559220431742, 0.9333999239844335
# 2, 1.0e-12, -1: true; (2.2449771720399996, 2.23132987792), (2.6819115584e8, 2.034635968e8); 0.9939209653042492, 0.7586514035585283
# 2, 1.0e-12, 5: true; (1.7985937100199998, 1.7920219999600004), (2.105621248e8, 1.645329472e8); 0.9963461953506296, 0.7813985889260934
# 20, 1.0, -1: true; (0.00668715012, 0.006643519760000002), (567440.0, 571136.0); 0.9934754926662245, 1.0065134639785704
# 20, 1.0, 5: true; (0.006823652120000001, 0.00675729792), (567440.0, 571136.0); 0.9902758524565579, 1.0065134639785704
# 20, 1.0e-6, -1: true; (0.07055684400000001, 0.068510984), (7.2339936e6, 6.134848e6); 0.9710040885615574, 0.8480582565071665
# 20, 1.0e-6, 5: true; (0.064295828, 0.061760117940000005), (5.9130336e6, 5.478256e6); 0.9605618258777227, 0.9264713124579573
# 20, 1.0e-12, -1: true; (0.25501391396, 0.24955720012000002), (2.85149984e7, 2.26296128e7); 0.9786022897524882, 0.7936038600654455
# 20, 1.0e-12, 5: true; (0.27613093415999995, 0.27204235586), (2.91494176e7, 2.5103504e7); 0.9851933347763532, 0.8612008769602312
# 200, 1.0, -1: true; (0.006782680000000001, 0.006730561999999999), (567440.0, 571136.0); 0.9923160166777731, 1.0065134639785704
# 200, 1.0, 5: true; (0.006798834060000003, 0.006967666000000001), (567440.0, 583929.6); 1.0248324842921666, 1.0290596362611024
# 200, 1.0e-6, -1: true; (0.07037478591999997, 0.06820597805999999), (7.231824e6, 6.1108864e6); 0.96918203257534, 0.844999325204817
# 200, 1.0e-6, 5: true; (0.06215345800000002, 0.06144463404000001), (5.883728e6, 5.4849504e6); 0.9885955828877612, 0.932223651399249
# 200, 1.0e-12, -1: true; (0.25570818196000006, 0.24896009407999997), (2.85163232e7, 2.26242688e7); 0.9736101996100552, 0.793379589694088
# 200, 1.0e-12, 5: true; (0.27547026197999996, 0.27112767600000004), (2.91516576e7, 2.51010848e7); 0.9842357358330199, 0.8610517159751492

# We definitely still have unnecessary allocations happening, but they are no
# longer as significant. Moreover, the batched GMRES optimizations have made
# our new solver consistently faster than the original solver.