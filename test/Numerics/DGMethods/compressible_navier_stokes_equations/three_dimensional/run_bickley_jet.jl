#!/usr/bin/env julia --project

include("box.jl")
ClimateMachine.init()

const FT = Float64

#################
# Initial State #
#################
import ClimateMachine.Ocean: ocean_init_state!

function ocean_init_state!(
    model::ThreeDimensionalCompressibleNavierStokes.CNSE3D,
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
    z = aux.z

    # The Bickley jet
    U = sech(y)^2
    V = 0
    W = 0

    # Slightly off-center vortical perturbations
    Ψ₁ = exp(-(y + l / 10)^2 / (2 * (l^2))) * cos(k * x) * cos(k * y)
    Ψ₂ = exp(-(z + l / 10)^2 / (2 * (l^2))) * cos(k * y) * cos(k * z)

    # Vortical velocity fields (u, v, w) = (-∂ʸ, +∂ˣ, 0) Ψ₁ + (0, -∂ᶻ, +∂ʸ)Ψ₂ 
    u = Ψ₁ * (k * tan(k * y) + y / (l^2) + 1 / (10 * l))
    v = -Ψ₁ * k * tan(k * x) + Ψ₂ * (k * tan(k * z) + z / (l^2) + 1 / (10 * l))
    w = -Ψ₂ * k * tan(k * y)

    ρ = model.ρₒ
    state.ρ = ρ
    state.ρu = ρ * @SVector [U + ϵ * u, V + ϵ * v, W + ϵ * w]
    state.ρθ = ρ * sin(k * y)

    return nothing
end

#################
# RUN THE TESTS #
#################

vtkpath =
    abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk_bickley_3D"))

let
    # simulation times
    timeend = FT(200) # s
    dt = FT(0.002) # s
    nout = Int(1000)

    # Domain Resolution
    N = 4
    Nˣ = 13
    Nʸ = 13
    Nᶻ = 13

    # Domain size
    Lˣ = 4 * FT(π)  # m
    Lʸ = 4 * FT(π)  # m
    Lᶻ = 4 * FT(π)  # m

    # model params
    cₛ = sqrt(10) # m/s
    ρₒ = 1 # kg/m³
    μ = 0 # 1e-6,   # m²/s
    ν = 0 # 1e-6,   # m²/s
    κ = 0 # 1e-6,   # m²/s
    α = 0   # 1/K
    g = 0   # m/s²

    resolution = (; N, Nˣ, Nʸ, Nᶻ)
    domain = (; Lˣ, Lʸ, Lᶻ)
    timespan = (; dt, nout, timeend)
    params = (; cₛ, ρₒ, μ, ν, κ, α, g)

    config = Config(
        "rusanov_overintegration",
        resolution,
        domain,
        params;
        numerical_flux_first_order = RoeNumericalFlux(),
        Nover = 1,
        periodicity = (true, true, true),
        boundary = ((0, 0), (0, 0), (0, 0)),
        boundary_conditons = (ClimateMachine.Ocean.OceanBC(
            Impenetrable(FreeSlip()),
            Insulating(),
        ),),
    )

    tic = Base.time()

    run_CNSE(config, resolution, timespan; TimeStepper = SSPRK22Heuns)

    toc = Base.time()
    time = toc - tic
    println(time)
end
