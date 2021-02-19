#!/usr/bin/env julia --project

include("sphere.jl")
ClimateMachine.init()

const FT = Float64
vtkpath = nothing

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
    state.ρu = @SVector [-0, -0, -0]
    state.ρθ = ρ

    return nothing
end

#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    include("refvals_sphere.jl")

    # simulation times
    timeend = FT(0.01) # s
    dt = FT(0.0001) # s
    nout = Int(10)
    timespan = (; dt, nout, timeend)

    # Domain Resolution
    polyorder_1 = (name = "first_order", N = 1, Nʰ = 18, Nᶻ = 30)
    polyorder_4 = (name = "fourth_order", N = 4, Nʰ = 8, Nᶻ = 6)
    resolutions = [polyorder_1, polyorder_4]

    # Domain size
    min_height = 0.5
    max_height = 1.0
    # min_height = 1
    # max_height = 1.01
    domain = (; min_height, max_height)

    # model params
    cₛ = sqrt(10) # m/s
    ρₒ = 1 # kg/m³
    μ = 0 # 1e-6,   # m²/s
    ν = 0 # 1e-3   # m²/s
    κ = 0 # 1e-3   # m²/s
    params = (; cₛ, ρₒ, μ, ν, κ)

    for resolution in resolutions
        @testset "$(resolution.name)" begin
            config = Config(
                resolution.name,
                resolution,
                domain,
                params;
                numerical_flux_first_order = RoeNumericalFlux(),
                Nover = 1,
                boundary = (1, 1),
                boundary_conditons = (ClimateMachine.Ocean.OceanBC(
                    Impenetrable(FreeSlip()),
                    Insulating(),
                ),),
            )

            println("starting test " * resolution.name)
            tic = Base.time()

            run_CNSE(
                config,
                resolution,
                timespan;
                TimeStepper = SSPRK22Heuns,
                refDat = getproperty(refVals, Symbol(resolution.name)),
            )

            toc = Base.time()
            time = toc - tic
            println(time)
        end
    end
end
