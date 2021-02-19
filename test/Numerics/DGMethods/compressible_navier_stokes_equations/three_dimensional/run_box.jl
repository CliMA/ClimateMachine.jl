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

    ρ = model.ρₒ
    state.ρ = ρ
    state.ρu = ρ * @SVector [-0, -0, -0]
    state.ρθ = ρ * 5

    return nothing
end

#################
# RUN THE TESTS #
#################

vtkpath = abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk_box_3D"))

let
    # simulation times
    timeend = FT(200) # s
    dt = FT(0.05) # s
    nout = Int(200)

    # Domain Resolution
    N = 1
    Nˣ = 8
    Nʸ = 8
    Nᶻ = 8

    # Domain size
    Lˣ = 4 * FT(π)  # m
    Lʸ = 4 * FT(π)  # m
    Lᶻ = 4 * FT(π)  # m

    # model params
    cₛ = sqrt(10) # m/s
    ρₒ = 1 # kg/m³
    μ = 0 # 1e-6,   # m²/s
    ν = 1e-2   # m²/s
    κ = 1e-2   # m²/s
    α = 2e-4   # 1/K
    g = 10     # m/s²

    resolution = (; N, Nˣ, Nʸ, Nᶻ)
    domain = (; Lˣ, Lʸ, Lᶻ)
    timespan = (; dt, nout, timeend)
    params = (; cₛ, ρₒ, μ, ν, κ, α, g)

    BC = (
        ClimateMachine.Ocean.OceanBC(Impenetrable(NoSlip()), Insulating()),
        ClimateMachine.Ocean.OceanBC(
            Impenetrable(KinematicStress(
                (state, aux, t) -> (@SVector [0.01 / state.ρ, -0, -0]),
            )),
            TemperatureFlux((state, aux, t) -> (0.1)),
        ),
    )

    config = Config(
        "heat_the_box",
        resolution,
        domain,
        params;
        numerical_flux_first_order = RoeNumericalFlux(),
        Nover = 1,
        periodicity = (true, true, false),
        boundary = ((0, 0), (0, 0), (1, 2)),
        boundary_conditons = BC,
    )

    tic = Base.time()

    run_CNSE(config, resolution, timespan; TimeStepper = SSPRK22Heuns)

    toc = Base.time()
    time = toc - tic
    println(time)
end
