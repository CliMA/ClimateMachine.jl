using ClimateMachine
using MPI
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.VTK: writemesh
using Logging
using LinearAlgebra
using Random
using StaticArrays
using ClimateMachine.DGmethods:
    DGModel,
    Vars,
    vars_state_conservative,
    number_state_conservative,
    init_ode_state
using ClimateMachine.ColumnwiseLUSolver:
    banded_matrix, banded_matrix_vector_product!
using ClimateMachine.DGmethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxSecondOrder,
    CentralNumericalFluxGradient
using ClimateMachine.MPIStateArrays: MPIStateArray, euclidean_distance

using Test

include("../DGmethods/advection_diffusion/advection_diffusion_model.jl")

struct Pseudo1D{n, α, β, μ, δ} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(
    ::Pseudo1D{n, α, β},
    aux::Vars,
    geom::LocalGeometry,
) where {n, α, β}
    # Direction of flow is n with magnitude α
    aux.u = α * n

    # diffusion of strength β in the n direction
    aux.D = β * n * n'
end

function initial_condition!(
    ::Pseudo1D{n, α, β, μ, δ},
    state,
    aux,
    x,
    t,
) where {n, α, β, μ, δ}
    ξn = dot(n, x)
    # ξT = SVector(x) - ξn * n
    state.ρ = exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
end

let
    # boiler plate MPI stuff
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD
    Random.seed!(777 + MPI.Comm_rank(mpicomm))

    # Mesh generation parameters
    N = 4
    Nq = N + 1
    Neh = 10
    Nev = 4

    @testset "$(@__FILE__) DGModel matrix" begin
        for FT in (Float64, Float32)
            for dim in (2, 3)
                for single_column in (false, true)
                    # Setup the topology
                    if dim == 2
                        brickrange = (
                            range(FT(0); length = Neh + 1, stop = 1),
                            range(FT(1); length = Nev + 1, stop = 2),
                        )
                    elseif dim == 3
                        brickrange = (
                            range(FT(0); length = Neh + 1, stop = 1),
                            range(FT(0); length = Neh + 1, stop = 1),
                            range(FT(1); length = Nev + 1, stop = 2),
                        )
                    end
                    topl = StackedBrickTopology(mpicomm, brickrange)

                    # Warp mesh
                    function warpfun(ξ1, ξ2, ξ3)
                        # single column currently requires no geometry warping

                        # Even if the warping is in only the horizontal, the way we
                        # compute metrics causes problems for the single column approach
                        # (possibly need to not use curl-invariant computation)
                        if !single_column
                            ξ1 = ξ1 + sin(2π * ξ1 * ξ2) / 10
                            ξ2 = ξ2 + sin(2π * ξ1) / 5
                            if dim == 3
                                ξ3 = ξ3 + sin(8π * ξ1 * ξ2) / 10
                            end
                        end
                        (ξ1, ξ2, ξ3)
                    end

                    # create the actual grid
                    grid = DiscontinuousSpectralElementGrid(
                        topl,
                        FloatType = FT,
                        DeviceArray = ArrayType,
                        polynomialorder = N,
                        meshwarp = warpfun,
                    )
                    d = dim == 2 ? FT[1, 10, 0] : FT[1, 1, 10]
                    n = SVector{3, FT}(d ./ norm(d))

                    α = FT(1)
                    β = FT(1 // 100)
                    μ = FT(-1 // 2)
                    δ = FT(1 // 10)
                    model = AdvectionDiffusion{dim}(Pseudo1D{n, α, β, μ, δ}())

                    # the nonlinear model is needed so we can grab the state_auxiliary below
                    dg = DGModel(
                        model,
                        grid,
                        RusanovNumericalFlux(),
                        CentralNumericalFluxSecondOrder(),
                        CentralNumericalFluxGradient(),
                    )

                    vdg = DGModel(
                        model,
                        grid,
                        RusanovNumericalFlux(),
                        CentralNumericalFluxSecondOrder(),
                        CentralNumericalFluxGradient();
                        direction = VerticalDirection(),
                        state_auxiliary = dg.state_auxiliary,
                    )

                    A_banded = banded_matrix(
                        vdg,
                        MPIStateArray(dg),
                        MPIStateArray(dg);
                        single_column = single_column,
                    )

                    Q = init_ode_state(dg, FT(0))
                    dQ1 = MPIStateArray(dg)
                    dQ2 = MPIStateArray(dg)

                    vdg(dQ1, Q, nothing, 0; increment = false)
                    Q.data .= dQ1.data

                    vdg(dQ1, Q, nothing, 0; increment = false)
                    banded_matrix_vector_product!(vdg, A_banded, dQ2, Q)
                    @test all(isapprox.(
                        Array(dQ1.realdata),
                        Array(dQ2.realdata),
                        atol = 100 * eps(FT),
                    ))

                    α = FT(1 // 10)
                    function op!(LQ, Q)
                        vdg(LQ, Q, nothing, 0; increment = false)
                        @. LQ = Q + α * LQ
                    end

                    A_banded = banded_matrix(
                        op!,
                        vdg,
                        MPIStateArray(dg),
                        MPIStateArray(dg);
                        single_column = single_column,
                    )

                    Q = init_ode_state(dg, FT(0))
                    dQ1 = MPIStateArray(vdg)
                    dQ2 = MPIStateArray(vdg)

                    op!(dQ1, Q)
                    Q.data .= dQ1.data

                    op!(dQ1, Q)
                    banded_matrix_vector_product!(vdg, A_banded, dQ2, Q)
                    @test all(isapprox.(
                        Array(dQ1.realdata),
                        Array(dQ2.realdata),
                        atol = 100 * eps(FT),
                    ))
                end
            end
        end
    end
end

nothing
