include("../CNSE.jl")
include("TwoDimensionalCompressibleNavierStokesEquations.jl")

function Config(
    name,
    resolution,
    domain,
    params;
    numerical_flux_first_order = RusanovNumericalFlux(),
    Nover = 0,
    periodicity = (true, true),
    boundary = ((0, 0), (0, 0)),
    boundary_conditons = (),
)
    mpicomm = MPI.COMM_WORLD
    ArrayType = ClimateMachine.array_type()

    xrange =
        range(-domain.Lˣ / 2; length = resolution.Nˣ + 1, stop = domain.Lˣ / 2)
    yrange =
        range(-domain.Lʸ / 2; length = resolution.Nʸ + 1, stop = domain.Lʸ / 2)

    brickrange = (xrange, yrange)

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

    model = TwoDimensionalCompressibleNavierStokes.CNSE2D{FT}(
        (domain.Lˣ, domain.Lʸ),
        ClimateMachine.Ocean.NonLinearAdvectionTerm(),
        TwoDimensionalCompressibleNavierStokes.ConstantViscosity{FT}(
            ν = params.ν,
            κ = params.κ,
        ),
        nothing,
        nothing,
        boundary_conditons;
        g = params.g,
        c = params.c,
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

import ClimateMachine.Ocean: ocean_init_state!, ocean_init_aux!

function ocean_init_state!(
    ::TwoDimensionalCompressibleNavierStokes.CNSE2D,
    state,
    aux,
    localgeo,
    t,
)
    ϵ = 0.1 # perturbation magnitude
    l = 0.5 # Gaussian width
    k = 0.5 # Sinusoidal wavenumber

    x = aux.x
    y = aux.y

    # The Bickley jet
    U = cosh(y)^(-2)

    # Slightly off-center vortical perturbations
    Ψ = exp(-(y + l / 10)^2 / (2 * (l^2))) * cos(k * x) * cos(k * y)

    # Vortical velocity fields (ũ, ṽ) = (-∂ʸ, +∂ˣ) ψ̃
    u = Ψ * (k * tan(k * y) + y / (l^2))
    v = -Ψ * k * tan(k * x)

    ρ = 1
    state.ρ = ρ
    state.ρu = ρ * @SVector [U + ϵ * u, ϵ * v]
    state.ρθ = ρ * sin(k * y)

    return nothing
end

function ocean_init_aux!(
    ::TwoDimensionalCompressibleNavierStokes.CNSE2D,
    aux,
    geom,
)
    @inbounds begin
        aux.x = geom.coord[1]
        aux.y = geom.coord[2]
    end

    return nothing
end
