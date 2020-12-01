using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using LinearAlgebra
using Printf
using Test

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")

struct Pseudo1D{n1, n2, α, β, μ, δ} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(
    ::Pseudo1D{n1, n2, α, β},
    aux::Vars,
    geom::LocalGeometry,
) where {n1, n2, α, β}
    # Direction of flow is n1 (resp n2) with magnitude α
    aux.advection.u = hcat(α * n1, α * n2)

    # diffusion of strength β in the n1 and n2 directions
    aux.diffusion.D = hcat(β * n1 * n1', β * n2 * n2')
end

function initial_condition!(
    ::Pseudo1D{n1, n2, α, β, μ, δ},
    state,
    aux,
    localgeo,
    t,
) where {n1, n2, α, β, μ, δ}
    ξn1 = dot(n1, localgeo.coord)
    ξn2 = dot(n2, localgeo.coord)
    ρ1 = exp(-(ξn1 - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
    ρ2 = exp(-(ξn2 - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
    state.ρ = (ρ1, ρ2)
end

Dirichlet_data!(P::Pseudo1D, x...) = initial_condition!(P, x...)

function Neumann_data!(
    ::Pseudo1D{n1, n2, α, β, μ, δ},
    ∇state,
    aux,
    x,
    t,
) where {n1, n2, α, β, μ, δ}
    ξn1 = dot(n1, x)
    ξn2 = dot(n2, x)
    ∇ρ1 =
        -(
            2n1 * (ξn1 - μ - α * t) / (4 * β * (δ + t)) *
            exp(-(ξn1 - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
        )
    ∇ρ2 =
        -(
            2n2 * (ξn2 - μ - α * t) / (4 * β * (δ + t)) *
            exp(-(ξn2 - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
        )
    ∇state.ρ = hcat(∇ρ1, ∇ρ2)
end

function test_run(mpicomm, dim, polynomialorders, level, ArrayType, FT)

    n_hd =
        dim == 2 ? SVector{3, FT}(1, 0, 0) :
        SVector{3, FT}(1 / sqrt(2), 1 / sqrt(2), 0)

    n_vd = dim == 2 ? SVector{3, FT}(0, 1, 0) : SVector{3, FT}(0, 0, 1)

    α = FT(1)
    β = FT(1 // 100)
    μ = FT(-1 // 2)
    δ = FT(1 // 10)

    # Grid/topology information
    base_num_elem = 4
    Ne = 2^(level - 1) * base_num_elem
    brickrange = ntuple(j -> range(FT(-1); length = Ne + 1, stop = 1), dim)
    periodicity = ntuple(j -> false, dim)
    bc = ntuple(j -> (1, 2), dim)

    topl = StackedBrickTopology(
        mpicomm,
        brickrange;
        periodicity = periodicity,
        boundary = bc,
    )

    dt = (α / 4) / (Ne * maximum(polynomialorders)^2)
    timeend = 1
    @info "time step" dt

    @info @sprintf """Test parameters:
    ArrayType                   = %s
    FloatType                   = %s
    Dimension                   = %s
    Horizontal polynomial order = %s
    Vertical polynomial order   = %s
      """ ArrayType FT dim polynomialorders[1] polynomialorders[end]

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorders,
    )

    # Model being tested
    model = AdvectionDiffusion{dim}(
        Pseudo1D{n_hd, n_vd, α, β, μ, δ}(),
        num_equations = 2,
    )

    # Main DG discretization
    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        direction = EveryDirection(),
    )

    # Initialize all relevant state arrays and create solvers
    Q = init_ode_state(dg, FT(0))

    eng0 = norm(Q, dims = (1, 3))
    @info @sprintf """Starting
    norm(Q₀) = %.16e""" eng0[1]

    solver = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)
    solve!(Q, solver; timeend = timeend)

    # Reference solution
    engf = norm(Q, dims = (1, 3))
    Q_ref = init_ode_state(dg, FT(timeend))

    engfe = norm(Q_ref, dims = (1, 3))
    errf = norm(Q_ref .- Q, dims = (1, 3))

    metrics = @. (engf, engf / eng0, engf - eng0, errf, errf / engfe)

    @info @sprintf """Finished
    Horizontal field:
      norm(Q)                 = %.16e
      norm(Q) / norm(Q₀)      = %.16e
      norm(Q) - norm(Q₀)      = %.16e
      norm(Q - Qe)            = %.16e
      norm(Q - Qe) / norm(Qe) = %.16e
    Vertical field:
      norm(Q)                 = %.16e
      norm(Q) / norm(Q₀)      = %.16e
      norm(Q) - norm(Q₀)      = %.16e
      norm(Q - Qe)            = %.16e
      norm(Q - Qe) / norm(Qe) = %.16e
      """ first.(metrics)... last.(metrics)...

    return errf
end

"""
    main()

Run this test problem
"""
function main()

    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    # Dictionary keys: dim, level, polynomial order, FT, and direction
    expected_result = Dict()

    # Dim 2, degree 4 in the horizontal, Float64
    expected_result[2, 1, 4, Float64, HorizontalDirection] = 0.0467837436192571
    expected_result[2, 2, 4, Float64, HorizontalDirection] =
        0.004066556782723549
    expected_result[2, 3, 4, Float64, HorizontalDirection] =
        5.3144336694234015e-5
    expected_result[2, 4, 4, Float64, HorizontalDirection] =
        3.978000110046181e-7

    # Dim 2, degree 2 in the vertical, Float64
    expected_result[2, 1, 2, Float64, VerticalDirection] = 0.15362016594121006
    expected_result[2, 2, 2, Float64, VerticalDirection] = 0.04935353328794371
    expected_result[2, 3, 2, Float64, VerticalDirection] = 0.015530511948609192
    expected_result[2, 4, 2, Float64, VerticalDirection] = 0.0006275095484456197

    # Dim 2, degree 2 in the horizontal, Float64
    expected_result[2, 1, 2, Float64, HorizontalDirection] = 0.15362016594121003
    expected_result[2, 2, 2, Float64, HorizontalDirection] = 0.04935353328794369
    expected_result[2, 3, 2, Float64, HorizontalDirection] =
        0.015530511948609204
    expected_result[2, 4, 2, Float64, HorizontalDirection] =
        0.0006275095484455967

    # Dim 2, degree 4 in the vertical, Float64
    expected_result[2, 1, 4, Float64, VerticalDirection] = 0.04678374361925714
    expected_result[2, 2, 4, Float64, VerticalDirection] = 0.0040665567827235
    expected_result[2, 3, 4, Float64, VerticalDirection] = 5.3144336694109365e-5
    expected_result[2, 4, 4, Float64, VerticalDirection] = 3.978000109805811e-7

    # Dim 3, degree 4 in the horizontal, Float64
    expected_result[3, 1, 4, Float64, HorizontalDirection] =
        0.017475667486259432
    expected_result[3, 2, 4, Float64, HorizontalDirection] =
        0.0012502148161420109
    expected_result[3, 3, 4, Float64, HorizontalDirection] =
        6.999081063570052e-5
    expected_result[3, 4, 4, Float64, HorizontalDirection] =
        2.8724182090419642e-6

    # Dim 3, degree 2 in the vertical, Float64
    expected_result[3, 1, 2, Float64, VerticalDirection] = 0.2172517221280645
    expected_result[3, 2, 2, Float64, VerticalDirection] = 0.06979643612684193
    expected_result[3, 3, 2, Float64, VerticalDirection] = 0.02196346062832051
    expected_result[3, 4, 2, Float64, VerticalDirection] = 0.0008874325139302493

    # Dim 3, degree 2 in the horizontal, Float64
    expected_result[3, 1, 2, Float64, HorizontalDirection] = 0.10343354980172516
    expected_result[3, 2, 2, Float64, HorizontalDirection] = 0.03415137756593495
    expected_result[3, 3, 2, Float64, HorizontalDirection] =
        0.0035959803480493553
    expected_result[3, 4, 2, Float64, HorizontalDirection] =
        0.0002714157844893719

    # Dim 3, degree 4 in the vertical, Float64
    expected_result[3, 1, 4, Float64, VerticalDirection] = 0.06616220472493903
    expected_result[3, 2, 4, Float64, VerticalDirection] = 0.005750979754288175
    expected_result[3, 3, 4, Float64, VerticalDirection] = 7.515744171591452e-5
    expected_result[3, 4, 4, Float64, VerticalDirection] = 5.625741705890895e-7

    # Dim 2, degree 4 in the horizontal, Float32
    expected_result[2, 1, 4, Float32, HorizontalDirection] = 0.046783954f0
    expected_result[2, 2, 4, Float32, HorizontalDirection] = 0.004066328f0
    expected_result[2, 3, 4, Float32, HorizontalDirection] = 5.327546f-5

    # Dim 2, degree 2 in the vertical, Float32
    expected_result[2, 1, 2, Float32, VerticalDirection] = 0.15362015f0
    expected_result[2, 2, 2, Float32, VerticalDirection] = 0.04935346f0
    expected_result[2, 3, 2, Float32, VerticalDirection] = 0.015530386f0

    # Dim 2, degree 2 in the horizontal, Float32
    expected_result[2, 1, 2, Float32, HorizontalDirection] = 0.1536202f0
    expected_result[2, 2, 2, Float32, HorizontalDirection] = 0.04935346f0
    expected_result[2, 3, 2, Float32, HorizontalDirection] = 0.015530357f0

    # Dim 2, degree 4 in the vertical, Float32
    expected_result[2, 1, 4, Float32, VerticalDirection] = 0.04678398f0
    expected_result[2, 2, 4, Float32, VerticalDirection] = 0.0040662177f0
    expected_result[2, 3, 4, Float32, VerticalDirection] = 5.3401447f-5

    # Dim 3, degree 4 in the horizontal, Float32
    expected_result[3, 1, 4, Float32, HorizontalDirection] = 0.01747554f0
    expected_result[3, 2, 4, Float32, HorizontalDirection] = 0.0012502924f0
    expected_result[3, 3, 4, Float32, HorizontalDirection] = 7.00218f-5

    # Dim 3, degree 2 in the vertical, Float32
    expected_result[3, 1, 2, Float32, VerticalDirection] = 0.21725166f0
    expected_result[3, 2, 2, Float32, VerticalDirection] = 0.06979626f0
    expected_result[3, 3, 2, Float32, VerticalDirection] = 0.021963252f0

    # Dim 3, degree 2 in the horizontal, Float32
    expected_result[3, 1, 2, Float32, HorizontalDirection] = 0.10343349f0
    expected_result[3, 2, 2, Float32, HorizontalDirection] = 0.034151305f0
    expected_result[3, 3, 2, Float32, HorizontalDirection] = 0.0035958516f0

    # Dim 3, degree 4 in the vertical, Float32
    expected_result[3, 1, 4, Float32, VerticalDirection] = 0.06616244f0
    expected_result[3, 2, 4, Float32, VerticalDirection] = 0.005750495f0
    expected_result[3, 3, 4, Float32, VerticalDirection] = 7.538217f-5

    @testset "Variable degree DG: advection diffusion model" begin
        for FT in (Float32, Float64)
            numlevels =
                integration_testing ||
                ClimateMachine.Settings.integration_testing ?
                (FT == Float64 ? 4 : 3) : 1
            for dim in 2:3
                for polynomialorders in ((4, 2), (2, 4))
                    result = Dict()
                    for level in 1:numlevels
                        result[level] = test_run(
                            mpicomm,
                            dim,
                            polynomialorders,
                            level,
                            ArrayType,
                            FT,
                        )
                        horiz_poly = polynomialorders[1]
                        vert_poly = polynomialorders[2]
                        @test result[level][1] ≈ FT(expected_result[
                            dim,
                            level,
                            horiz_poly,
                            FT,
                            HorizontalDirection,
                        ])
                        @test result[level][2] ≈ FT(expected_result[
                            dim,
                            level,
                            vert_poly,
                            FT,
                            VerticalDirection,
                        ])
                    end
                    @info begin
                        msg = ""
                        for l in 1:(numlevels - 1)
                            rate = @. log2(result[l]) - log2(result[l + 1])
                            msg *= @sprintf(
                                "\n  rates for level %d Horizontal = %e",
                                l,
                                rate[1]
                            )
                            msg *= @sprintf(", Vertical = %e\n", rate[2])
                        end
                        msg
                    end
                end
            end
        end
    end
end

main()
