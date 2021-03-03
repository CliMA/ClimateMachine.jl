#!/usr/bin/env julia --project

include("config_sphere.jl")
ClimateMachine.init()

const FT = Float64

#################
# Initial State #
#################
function cnse_init_state!(model::CNSE3D, state, aux, localgeo, t)
    x = aux.x
    y = aux.y
    z = aux.z

    ρ = model.ρₒ
    state.ρ = ρ
    state.ρu = @SVector [-0, -0, -0]
    state.ρθ = ρ

    return nothing
end

#################
# RUN THE TESTS #
#################

vtkpath = abspath(joinpath(
    ClimateMachine.Settings.output_dir,
    "vtk_sphere_iso_p4_roeOI",
))

let
    # simulation times
    timeend = FT(10) # s
    dt = FT(0.0001) # s
    nout = Int(1000)

    # Domain Resolution
    N = 4
    Nʰ = 8
    Nᶻ = 6
    """
    N = 1
    Nʰ = 18
    Nᶻ = 30
    """

    # Domain size
    min_height = 0.5
    max_height = 1.0
    # min_height = 1
    # max_height = 1.01

    # model params
    cₛ = sqrt(10) # m/s
    ρₒ = 1 # kg/m³
    μ = 0 # 1e-6,   # m²/s
    ν = 0 # 1e-3   # m²/s
    κ = 0 # 1e-3   # m²/s

    resolution = (; N, Nʰ, Nᶻ)
    domain = (; min_height, max_height)
    timespan = (; dt, nout, timeend)
    params = (; cₛ, ρₒ, μ, ν, κ)

    config = Config(
        "roeOI",
        resolution,
        domain,
        params;
        numerical_flux_first_order = RoeNumericalFlux(),
        Nover = 2,
        boundary = (1, 1),
        boundary_conditons = (FluidBC(Impenetrable(FreeSlip()), Insulating()),),
    )

    tic = Base.time()

    run_CNSE(config, resolution, timespan; TimeStepper = SSPRK22Heuns)

    toc = Base.time()
    time = toc - tic
    println(time)
end
