include("../CNSE.jl")
include("ThreeDimensionalCompressibleNavierStokesEquations.jl")

function Config(
    name,
    resolution,
    domain,
    params;
    numerical_flux_first_order = RusanovNumericalFlux(),
    Nover = 0,
    periodicity = (true, true, true),
    boundary = ((0, 0), (0, 0), (0, 0)),
    boundary_conditons = (),
)
    mpicomm = MPI.COMM_WORLD
    ArrayType = ClimateMachine.array_type()

    xrange =
        range(-domain.Lˣ / 2; length = resolution.Nˣ + 1, stop = domain.Lˣ / 2)
    yrange =
        range(-domain.Lʸ / 2; length = resolution.Nʸ + 1, stop = domain.Lʸ / 2)
    zrange =
        range(-domain.Lᶻ / 2; length = resolution.Nᶻ + 1, stop = domain.Lᶻ / 2)

    brickrange = (xrange, yrange, zrange)

    topl = BrickTopology(
        mpicomm,
        brickrange,
        periodicity = periodicity,
        boundary = boundary,
    )

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = resolution.N + Nover,
    )

    model = ThreeDimensionalCompressibleNavierStokes.CNSE3D{FT}(
        (domain.Lˣ, domain.Lʸ, domain.Lᶻ),
        ClimateMachine.Ocean.NonLinearAdvectionTerm(),
        ThreeDimensionalCompressibleNavierStokes.ConstantViscosity{FT}(
            μ = params.μ,
            ν = params.ν,
            κ = params.κ,
        ),
        nothing,
        ThreeDimensionalCompressibleNavierStokes.Buoyancy{FT}(
            α = params.α,
            g = params.g,
        ),
        boundary_conditons;
        cₛ = params.cₛ,
        ρₒ = params.ρₒ,
    )

    dg = DGModel(
        model,
        grid,
        numerical_flux_first_order,
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    return Config(name, dg, Nover, mpicomm, ArrayType)
end

import ClimateMachine.Ocean: ocean_init_aux!

function ocean_init_aux!(
    ::ThreeDimensionalCompressibleNavierStokes.CNSE3D,
    aux,
    geom,
)
    @inbounds begin
        aux.x = geom.coord[1]
        aux.y = geom.coord[2]
        aux.z = geom.coord[3]
    end

    return nothing
end
