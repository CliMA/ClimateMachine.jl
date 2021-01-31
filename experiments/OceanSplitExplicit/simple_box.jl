using Test
using MPI

using ClimateMachine
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers

using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies

using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.SystemSolvers

using ClimateMachine.VariableTemplates
using ClimateMachine.BalanceLaws

using ClimateMachine.Ocean
using ClimateMachine.Ocean.SplitExplicit01
using ClimateMachine.Ocean.OceanProblems

using ClimateMachine:
    ConfigSpecificInfo, DriverConfiguration, OceanSplitExplicitConfigType

using CLIMAParameters
using CLIMAParameters.Planet: grav

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

struct OceanSplitExplicitSpecificInfo <: ConfigSpecificInfo
    model_2D::BalanceLaw
    grid_2D::DiscontinuousSpectralElementGrid
    dg::DGModel
end

function OceanSplitExplicitConfiguration(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    (Nˣ, Nʸ, Nᶻ)::NTuple{3, Int},
    param_set::AbstractParameterSet,
    model_3D;
    FT = Float64,
    array_type = ClimateMachine.array_type(),
    solver_type = SplitExplicitSolverType{FT}(90.0 * 60.0, 240.0),
    mpicomm = MPI.COMM_WORLD,
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
    fv_reconstruction = nothing,
    periodicity = (false, false, false),
    boundary = ((1, 1), (1, 1), (2, 3)),
)

    (polyorder_horz, polyorder_vert) = isa(N, Int) ? (N, N) : N

    xrange = range(FT(0); length = Nˣ + 1, stop = model_3D.problem.Lˣ)
    yrange = range(FT(0); length = Nʸ + 1, stop = model_3D.problem.Lʸ)
    zrange = range(FT(-model_3D.problem.H); length = Nᶻ + 1, stop = 0)

    brickrange_2D = (xrange, yrange)
    brickrange_3D = (xrange, yrange, zrange)

    topology_2D = BrickTopology(
        mpicomm,
        brickrange_2D;
        periodicity = (periodicity[1], periodicity[2]),
        boundary = (boundary[1], boundary[2]),
    )
    topology_3D = StackedBrickTopology(
        mpicomm,
        brickrange_3D;
        periodicity = periodicity,
        boundary = boundary,
    )

    grid_2D = DiscontinuousSpectralElementGrid(
        topology_2D,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = polyorder_horz,
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topology_3D,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horz, polyorder_vert),
    )

    model_2D = BarotropicModel(model_3D)

    dg_2D = DGModel(
        model_2D,
        grid_2D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
    )

    Q_2D = init_ode_state(dg_2D, FT(0); init_on_cpu = true)

    vert_filter = CutoffFilter(grid_3D, polyorder_vert - 1)
    exp_filter = ExponentialFilter(grid_3D, 1, 8)

    flowintegral_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.FlowIntegralModel(model_3D),
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
    )

    tendency_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.TendencyIntegralModel(model_3D),
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
    )

    conti3d_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.Continuity3dModel(model_3D),
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
    )
    conti3d_Q = init_ode_state(conti3d_dg, FT(0); init_on_cpu = true)

    ivdc_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.IVDCModel(model_3D),
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient;
        direction = VerticalDirection(),
    )
    # Not sure this is needed since we set values later,
    # but we'll do it just in case!
    ivdc_Q = init_ode_state(ivdc_dg, FT(0); init_on_cpu = true)
    ivdc_RHS = init_ode_state(ivdc_dg, FT(0); init_on_cpu = true)

    ivdc_bgm_solver = BatchedGeneralizedMinimalResidual(
        ivdc_dg,
        ivdc_Q;
        max_subspace_size = 10,
    )

    modeldata = (
        dg_2D = dg_2D,
        Q_2D = Q_2D,
        vert_filter = vert_filter,
        exp_filter = exp_filter,
        flowintegral_dg = flowintegral_dg,
        tendency_dg = tendency_dg,
        conti3d_dg = conti3d_dg,
        conti3d_Q = conti3d_Q,
        ivdc_dg = ivdc_dg,
        ivdc_Q = ivdc_Q,
        ivdc_RHS = ivdc_RHS,
        ivdc_bgm_solver = ivdc_bgm_solver,
    )

    dg_3D = DGModel(
        model_3D,
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient;
        modeldata = modeldata,
    )


    return DriverConfiguration(
        OceanSplitExplicitConfigType(),
        name,
        (polyorder_horz, polyorder_vert),
        FT,
        array_type,
        solver_type,
        param_set,
        model_3D,
        mpicomm,
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        fv_reconstruction,
        nothing, # filter
        OceanSplitExplicitSpecificInfo(model_2D, grid_2D, dg_3D),
    )
end


function config_simple_box(
    name,
    resolution,
    dimensions,
    boundary_conditions;
    dt_slow = 90.0 * 60.0,
    dt_fast = 240.0,
)

    problem = OceanGyre{FT}(
        dimensions...;
        τₒ = 0.1,
        λʳ = 10 // 86400,
        θᴱ = 10,
        BC = boundary_conditions,
    )

    add_fast_substeps = 2
    numImplSteps = 5
    numImplSteps > 0 ? ivdc_dt = dt_slow / FT(numImplSteps) : ivdc_dt = dt_slow
    model_3D = OceanModel{FT}(
        param_set,
        problem;
        cʰ = 1,
        κᶜ = FT(0.1),
        add_fast_substeps = add_fast_substeps,
        numImplSteps = numImplSteps,
        ivdc_dt = ivdc_dt,
    )

    N, Nˣ, Nʸ, Nᶻ = resolution
    resolution = (Nˣ, Nʸ, Nᶻ)

    config = OceanSplitExplicitConfiguration(
        name,
        N,
        resolution,
        param_set,
        model_3D;
        solver_type = SplitExplicitSolverType{FT}(dt_slow, dt_fast),
    )

    return config
end

function run_simple_box(driver_config, timespan; refDat = ())

    timestart, timeend = timespan
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        init_on_cpu = true,
        ode_dt = driver_config.solver_type.dt_slow,
    )

    ## Create a callback to report state statistics for main MPIStateArrays
    ## every ntFreq timesteps.
    nt_freq = 1 # floor(Int, 1 // 10 * solver_config.timeend / solver_config.dt)
    cb = ClimateMachine.StateCheck.sccreate(
        [
            (solver_config.Q, "oce Q_3D"),
            (solver_config.dg.state_auxiliary, "oce aux"),
            (solver_config.dg.modeldata.Q_2D, "baro Q_2D"),
            (solver_config.dg.modeldata.dg_2D.state_auxiliary, "baro aux"),
        ],
        nt_freq;
        prec = 12,
    )

    result = ClimateMachine.invoke!(solver_config; user_callbacks = [cb])

    ## Check results against reference if present
    ClimateMachine.StateCheck.scprintref(cb)
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cb, refDat)
    end
end
