include("split_explicit.jl")

function SplitConfig(
    name,
    resolution,
    dimensions,
    coupling,
    rotation = Fixed();
    boundary_conditions = (
        OceanBC(Impenetrable(FreeSlip()), Insulating()),
        OceanBC(Penetrable(FreeSlip()), Insulating()),
    ),
    solver = SplitExplicitSolver,
    dt_slow = 90 * 60,
)
    mpicomm = MPI.COMM_WORLD
    ArrayType = ClimateMachine.array_type()

    N, Nˣ, Nʸ, Nᶻ = resolution
    Lˣ, Lʸ, H = dimensions

    xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
    yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
    zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)

    brickrange_2D = (xrange, yrange)
    topl_2D = BrickTopology(
        mpicomm,
        brickrange_2D,
        periodicity = (true, true),
        boundary = ((0, 0), (0, 0)),
    )
    grid_2D = DiscontinuousSpectralElementGrid(
        topl_2D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    brickrange_3D = (xrange, yrange, zrange)
    topl_3D = StackedBrickTopology(
        mpicomm,
        brickrange_3D;
        periodicity = (true, true, false),
        boundary = ((0, 0), (0, 0), (1, 2)),
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topl_3D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    problem = SimpleBox{FT}(
        dimensions...;
        BC = boundary_conditions,
        rotation = rotation,
    )

    dg_3D, dg_2D = setup_models(
        solver,
        problem,
        grid_3D,
        grid_2D,
        param_set,
        coupling,
        dt_slow,
    )

    return SplitConfig(name, dg_3D, dg_2D, solver, mpicomm, ArrayType)
end

function setup_models(
    ::Type{SplitExplicitSolver},
    problem,
    grid_3D,
    grid_2D,
    param_set,
    coupling,
    _,
)

    model_3D = HydrostaticBoussinesqModel{FT}(
        param_set,
        problem;
        coupling = coupling,
        cʰ = FT(1),
        αᵀ = FT(0),
        κʰ = FT(0),
        κᶻ = FT(0),
    )

    model_2D = ShallowWaterModel{FT}(
        param_set,
        problem,
        ShallowWater.ConstantViscosity{FT}(model_3D.νʰ),
        nothing;
        coupling = coupling,
        c = FT(1),
    )

    vert_filter = CutoffFilter(grid_3D, polynomialorder(grid_3D) - 1)
    exp_filter = ExponentialFilter(grid_3D, 1, 8)

    integral_model = DGModel(
        VerticalIntegralModel(model_3D),
        grid_3D,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    dg_2D = DGModel(
        model_2D,
        grid_2D,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q_2D = init_ode_state(dg_2D, FT(0); init_on_cpu = true)

    modeldata = (
        dg_2D = dg_2D,
        Q_2D = Q_2D,
        vert_filter = vert_filter,
        exp_filter = exp_filter,
        integral_model = integral_model,
    )

    dg_3D = DGModel(
        model_3D,
        grid_3D,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        modeldata = modeldata,
    )

    return dg_3D, dg_2D

end
