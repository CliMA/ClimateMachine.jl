include("split_explicit.jl")

function SplitConfig(name, resolution, dimensions, coupling, rotation = Fixed())
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

    problem = SimpleBox{FT}(Lˣ, Lʸ, H; rotation = rotation)

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

    return SplitConfig(
        name,
        model_3D,
        model_2D,
        grid_3D,
        grid_2D,
        mpicomm,
        ArrayType,
    )
end
