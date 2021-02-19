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
    x = aux.x
    y = aux.y
    z = aux.z

    Uₒ = 1

    u = Uₒ * sin(x) * cos(y) * cos(z)
    v = -Uₒ * cos(x) * sin(y) * cos(z)

    ρ = model.ρₒ
    state.ρ = ρ
    state.ρu = ρ * @SVector [u, v, -0]
    state.ρθ = ρ * sin(0.5 * z)

    return nothing
end

#################
# RUN THE TESTS #
#################

vtkpath =
    abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk_taylor_green"))

let
    # simulation times
    timeend = FT(200) # s
    dt = FT(0.01) # s
    nout = Int(100)

    # Domain Resolution
    N = 1
    Nˣ = 16
    Nʸ = 16
    Nᶻ = 16

    # Domain size
    Lˣ = 4 * FT(π)  # m
    Lʸ = 4 * FT(π)  # m
    Lᶻ = 4 * FT(π)  # m

    # model params
    cₛ = sqrt(10) # m/s
    ρₒ = 1 # kg/m³
    μ = 0 # 1e-6,   # m²/s
    ν = 1e-3   # m²/s
    κ = 1e-3   # m²/s
    α = 0   # 1/K
    g = 0   # m/s²

    resolution = (; N, Nˣ, Nʸ, Nᶻ)
    domain = (; Lˣ, Lʸ, Lᶻ)
    timespan = (; dt, nout, timeend)
    params = (; cₛ, ρₒ, μ, ν, κ, α, g)

    config = Config(
        "roeflux_overintegration",
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
