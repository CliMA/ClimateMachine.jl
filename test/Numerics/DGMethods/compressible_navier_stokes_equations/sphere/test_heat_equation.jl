#!/usr/bin/env julia --project

include("../shared_source/boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")

ClimateMachine.init()

#! format: off
refVals = (
[
 [ "state",     "ρ",   9.99999999999998778755e-01,  1.00000000000000111022e+00,  1.00000000000000000000e+00,  4.03070839356576998483e-16 ],
 [ "state", "ρu[1]",  -6.93674606129580362185e-18,  1.19513727285696382782e-17,  9.77991462267996185035e-19,  2.78015892816367900233e-18 ],
 [ "state", "ρu[2]",  -1.21340560458612968582e-17,  1.22630936333135748291e-17, -1.30065979768613433360e-21,  2.78919703417897612909e-18 ],
 [ "state", "ρu[3]",  -1.00545092917566730233e-17,  1.00043577527427175213e-17,  3.76131929192945135398e-19,  2.47195369518944690265e-18 ],
 [ "state",    "ρθ",  -3.00959038541689510859e-02, -3.00957518135526284897e-02, -3.00958465374152675520e-02,  3.12498428220800724718e-08 ],
],
[
 [ "state",     "ρ",    12,    12,    12,     0 ],
 [ "state", "ρu[1]",     0,     0,     0,     0 ],
 [ "state", "ρu[2]",     0,     0,     0,     0 ],
 [ "state", "ρu[3]",     0,     0,     0,     0 ],
 [ "state",    "ρθ",    12,    12,    12,     0 ],
],
)
#! format: on

#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    ########
    # Setup physical and numerical domains
    ########
    Ω = AtmosDomain(radius = 1, height = 0.2)

    grid = DiscretizedDomain(
        Ω;
        elements = (vertical = 1, horizontal = 5),
        polynomial_order = (vertical = 1 + 0, horizontal = 3 + 0),
        overintegration_order = (vertical = 1, horizontal = 1),
    )

    ########
    # Define physical parameters and parameterizations
    ########
    parameters = (
        ρₒ = 1,        # reference density
        cₛ = 1e-2,     # sound speed
        R = Ω.radius, # [m]
        ω = 0.0,      # [s⁻¹]
        K = 7.848e-6, # [s⁻¹]
        n = 4,        # dimensionless
        Ω = 0.0,      # [s⁻¹] 2π/86400
    )

    physics = FluidPhysics(;
        orientation = SphericalOrientation(),
        advection = NonLinearAdvectionTerm(),
        dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0.0, κ = 1e-3),
        coriolis = nothing,
        buoyancy = Buoyancy{Float64}(α = 2e-4, g = 0),
    )

    ########
    # Define timestepping parameters
    ########
    Δt = 0.1 * min_node_distance(grid.numerical)^2 / physics.dissipation.κ
    start_time = 0
    end_time = 20.0 * Δt

    method = SSPRK22Heuns

    timestepper = TimeStepper(method = method, timestep = Δt)

    callbacks = (Info(), StateCheck(10))

    ########
    # Define boundary conditions (west east are the ones that are enforced for a sphere)
    ########
    ρu_bcs = (bottom = Impenetrable(FreeSlip()), top = Impenetrable(FreeSlip()))
    ρθ_bcs = (
        bottom = TemperatureFlux(
            flux = (p, state, aux, t) -> p.Q,
            params = (Q = 1e-3,), # positive means removing heat from system
        ),
        top = TemperatureFlux(
            flux = (p, state, aux, t) -> p.Q,
            params = (Q = 1e-3,), # positive means removing heat from system
        ),
    )
    BC = (ρθ = ρθ_bcs, ρu = ρu_bcs)

    ########
    # Define initial conditions
    ########

    # Earth Spherical Representation
    # longitude: λ ∈ [-π, π), λ = 0 is the Greenwich meridian
    # latitude:  ϕ ∈ [-π/2, π/2], ϕ = 0 is the equator
    # radius:    r ∈ [Rₑ - hᵐⁱⁿ, Rₑ + hᵐᵃˣ]
    # Rₑ = Radius of sphere; hᵐⁱⁿ, hᵐᵃˣ ≥ 0

    ρ₀(p, λ, ϕ, r) = p.ρₒ
    ρuʳᵃᵈ(p, λ, ϕ, r) = 0.0
    ρuˡᵃᵗ(p, λ, ϕ, r) = 0.0
    ρuˡᵒⁿ(p, λ, ϕ, r) = 0.0
    ρθ₀(p, λ, ϕ, r) = 0.0

    # Cartesian Representation (boiler plate really)
    ρ₀ᶜᵃʳᵗ(p, x...) = ρ₀(p, lon(x...), lat(x...), rad(x...))
    ρu⃗₀ᶜᵃʳᵗ(p, x...) = (
        ρuʳᵃᵈ(p, lon(x...), lat(x...), rad(x...)) * r̂(x...) +
        ρuˡᵃᵗ(p, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...) +
        ρuˡᵒⁿ(p, lon(x...), lat(x...), rad(x...)) * λ̂(x...)
    )
    ρθ₀ᶜᵃʳᵗ(p, x...) = ρθ₀(p, lon(x...), lat(x...), rad(x...))

    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρθ = ρθ₀ᶜᵃʳᵗ)

    ########
    # Create the things
    ########

    model = SpatialModel(
        balance_law = Fluid3D(),
        physics = physics,
        numerics = (flux = RoeNumericalFlux(),),
        grid = grid,
        boundary_conditions = BC,
        parameters = parameters,
    )

    simulation = Simulation(
        model = model,
        initial_conditions = initial_conditions,
        timestepper = timestepper,
        callbacks = callbacks,
        time = (; start = start_time, finish = end_time),
    )

    ########
    # Run the model
    ########
    tic = Base.time()

    @testset "State Check" begin
        evolve!(simulation, model, refDat = refVals)
    end

    toc = Base.time()
    time = toc - tic
    println(time)

    ## Check the budget
    θ̄ = weightedsum(simulation.state, 5)

    @testset "Exact Geometry" begin
        # Exact (with exact geometry)
        R = grid.domain.radius
        H = grid.domain.height

        θ̄ᴱ = 0
        θ̄ᴱ -= 4π * R^2 * end_time * BC.ρθ.bottom.params.Q
        θ̄ᴱ -= 4π * (R + H)^2 * end_time * BC.ρθ.top.params.Q

        δθ̄ᴱ = abs(θ̄ - θ̄ᴱ) / abs(θ̄ᴱ)
        println("The relative error w.r.t. exact geometry ", δθ̄ᴱ)

        @test isapprox(0, δθ̄ᴱ; atol = 1e-9)
    end

    # TODO: make Approx Geom test MPI safe
    #=
    @testset "Approx Geometry" begin
        # Exact (with approximated geometry)
        n_e = convention(grid.resolution.elements, Val(3))
        n_ijk = convention(grid.resolution.polynomial_order, Val(3))
        n_ijk =
            n_ijk .+ convention(grid.resolution.overintegration_order, Val(3))
        n_ijk = n_ijk .+ (1, 1, 1)

        Mᴴ = reshape(
            grid.numerical.vgeo[:, grid.numerical.MHid, :],
            (n_ijk..., n_e[3], 6 * n_e[1] * n_e[2]),
        )
        Aᴮᵒᵗ = sum(Mᴴ[:, :, 1, 1, :])
        Aᵀᵒᵖ = sum(Mᴴ[:, :, end, end, :])

        θ̄ᴬ = 0
        θ̄ᴬ -= Aᴮᵒᵗ * end_time * BC.ρθ.bottom.params.Q
        θ̄ᴬ -= Aᵀᵒᵖ * end_time * BC.ρθ.top.params.Q

        δθ̄ᴬ = abs(θ̄ - θ̄ᴬ) / abs(θ̄ᴬ)
        println("The relative error w.r.t. approximated geometry ", δθ̄ᴬ)

        @test isapprox(0, δθ̄ᴬ; atol = 1e-15)
    end
    =#

end
