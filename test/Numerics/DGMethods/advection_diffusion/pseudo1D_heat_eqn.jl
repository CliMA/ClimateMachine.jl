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
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
import ClimateMachine.DGMethods.NumericalFluxes:
    normal_boundary_flux_second_order!

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")

struct HeatEqn{n, κ, A} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(
    ::HeatEqn{n},
    aux::Vars,
    geom::LocalGeometry,
) where {n}
    # diffusion of strength 1 in the n direction
    aux.diffusion.D = n * n'
end

# solution is such that
# u(1, t) = 1
# ∇u(0,t) = n
function initial_condition!(
    ::HeatEqn{n, κ, A},
    state,
    aux,
    localgeo,
    t,
) where {n, κ, A}
    ξn = dot(n, localgeo.coord)
    state.ρ = ξn + sum(A .* cos.(κ * ξn) .* exp.(-κ .^ 2 * t))
end
inhomogeneous_data!(::Val{0}, P::HeatEqn, x...) = initial_condition!(P, x...)

function normal_boundary_flux_second_order!(
    ::CentralNumericalFluxSecondOrder,
    bcs,
    ::AdvectionDiffusion{1, dim, HeatEqn{nd, κ, A}},
    fluxᵀn::Vars{S},
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    t,
    _...,
) where {S, dim, nd, κ, A}

    if any_isa(bcs, InhomogeneousBC{0})
        fluxᵀn.ρ = -diff⁻.σ' * n⁻
    elseif any_isa(bcs, InhomogeneousBC{1})
        # Get exact gradient of ρ
        x = aux⁻.coord
        ξn = dot(nd, x)
        ∇ρ = SVector(ntuple(
            i ->
                nd[i] * (1 - sum(A .* κ .* sin.(κ * ξn) .* exp.(-κ .^ 2 * t))),
            Val(3),
        ))

        # Compute flux value
        D = aux⁻.diffusion.D
        fluxᵀn.ρ = -(D * ∇ρ)' * n⁻
    end
end

function test_run(
    mpicomm,
    ArrayType,
    dim,
    topl,
    N,
    timeend,
    FT,
    direction,
    dt,
    n,
    κ = 10 * FT(π) / 2,
    A = 1,
)

    numberofsteps = convert(Int64, cld(timeend, dt))
    dt = timeend / numberofsteps
    @info "time step" dt numberofsteps dt * numberofsteps timeend

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )
    bcs = (InhomogeneousBC{1}(), InhomogeneousBC{0}())
    model = AdvectionDiffusion{dim}(HeatEqn{n, κ, A}(), bcs; advection = false)
    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        direction = direction(),
    )

    Q = init_ode_state(dg, FT(0))

    lsrk = LSRK144NiegemannDiehlBusch(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
    norm(Q₀) = %.16e""" eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            @info @sprintf(
                """Update
                simtime = %.16e
                runtime = %s
                norm(Q) = %.16e""",
                gettime(lsrk),
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end
    callbacks = (cbinfo,)

    solve!(
        Q,
        lsrk;
        numberofsteps = numberofsteps,
        callbacks = callbacks,
        adjustfinalstep = false,
    )

    # Print some end of the simulation information
    engf = norm(Q)
    Qe = init_ode_state(dg, FT(timeend))

    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ engf engf / eng0 engf - eng0 errf errf / engfe
    errf
end

using Test
let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    base_num_elem = 4

    expected_result = Dict()
    expected_result[2, 1, Float64, EveryDirection] = 0.005157483268127576
    expected_result[2, 2, Float64, EveryDirection] = 6.5687731035717e-5
    expected_result[2, 3, Float64, EveryDirection] = 1.6644861275185443e-6
    expected_result[2, 1, Float64, HorizontalDirection] = 0.020515449798977983
    expected_result[2, 2, Float64, HorizontalDirection] = 0.0005686256942296802
    expected_result[2, 3, Float64, HorizontalDirection] = 1.0132022682547854e-5
    expected_result[2, 1, Float64, VerticalDirection] = 0.02051544979897792
    expected_result[2, 2, Float64, VerticalDirection] = 0.0005686256942296017
    expected_result[2, 3, Float64, VerticalDirection] = 1.0132022682754848e-5
    expected_result[3, 1, Float64, EveryDirection] = 0.001260581018671256
    expected_result[3, 2, Float64, EveryDirection] = 2.214908522198975e-5
    expected_result[3, 3, Float64, EveryDirection] = 5.931735594156876e-7
    expected_result[3, 1, Float64, HorizontalDirection] = 0.005157483268127569
    expected_result[3, 2, Float64, HorizontalDirection] = 6.568773103570526e-5
    expected_result[3, 3, Float64, HorizontalDirection] = 1.6644861273865866e-6
    expected_result[3, 1, Float64, VerticalDirection] = 0.020515449798978087
    expected_result[3, 2, Float64, VerticalDirection] = 0.0005686256942297547
    expected_result[3, 3, Float64, VerticalDirection] = 1.0132022682817856e-5

    expected_result[2, 1, Float32, EveryDirection] = 0.005157135
    expected_result[2, 2, Float32, EveryDirection] = 6.5721644e-5
    expected_result[2, 3, Float32, EveryDirection] = 3.280845e-6
    expected_result[2, 1, Float32, HorizontalDirection] = 0.020514594
    expected_result[2, 2, Float32, HorizontalDirection] = 0.0005684704
    expected_result[2, 3, Float32, HorizontalDirection] = 1.02350195e-5
    expected_result[2, 1, Float32, VerticalDirection] = 0.020514673
    expected_result[2, 2, Float32, VerticalDirection] = 0.0005684843
    expected_result[2, 3, Float32, VerticalDirection] = 1.0227403e-5
    expected_result[3, 1, Float32, EveryDirection] = 0.0012602004
    expected_result[3, 2, Float32, EveryDirection] = 2.2415780e-5
    expected_result[3, 3, Float32, EveryDirection] = 1.1309192e-5
    expected_result[3, 1, Float32, HorizontalDirection] = 0.005157044
    expected_result[3, 2, Float32, HorizontalDirection] = 6.66792e-5
    expected_result[3, 3, Float32, HorizontalDirection] = 9.930429e-5
    expected_result[3, 1, Float32, VerticalDirection] = 0.020514654
    expected_result[3, 2, Float32, VerticalDirection] = 0.0005684157
    expected_result[3, 3, Float32, VerticalDirection] = 3.224683e-5

    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32)
            numlevels =
                integration_testing ||
                ClimateMachine.Settings.integration_testing ?
                3 : 1
            result = zeros(FT, numlevels)
            for dim in 2:3
                connectivity = dim == 2 ? :face : :full
                for direction in
                    (EveryDirection, HorizontalDirection, VerticalDirection)
                    if direction <: EveryDirection
                        n =
                            dim == 2 ?
                            SVector{3, FT}(1 / sqrt(2), 1 / sqrt(2), 0) :
                            SVector{3, FT}(
                                1 / sqrt(3),
                                1 / sqrt(3),
                                1 / sqrt(3),
                            )
                    elseif direction <: HorizontalDirection
                        n =
                            dim == 2 ? SVector{3, FT}(1, 0, 0) :
                            SVector{3, FT}(1 / sqrt(2), 1 / sqrt(2), 0)
                    elseif direction <: VerticalDirection
                        n =
                            dim == 2 ? SVector{3, FT}(0, 1, 0) :
                            SVector{3, FT}(0, 0, 1)
                    end
                    for l in 1:numlevels
                        Ne = 2^(l - 1) * base_num_elem
                        brickrange = ntuple(
                            j -> range(FT(0); length = Ne + 1, stop = 1),
                            dim,
                        )
                        periodicity = ntuple(j -> false, dim)
                        bc = ntuple(j -> (1, 2), dim)
                        topl = StackedBrickTopology(
                            mpicomm,
                            brickrange;
                            periodicity = periodicity,
                            boundary = bc,
                            connectivity = connectivity,
                        )
                        dt = 1 / (Ne * polynomialorder^2)^2

                        timeend = 0.01

                        @info (ArrayType, FT, dim, direction)
                        result[l] = test_run(
                            mpicomm,
                            ArrayType,
                            dim,
                            topl,
                            polynomialorder,
                            timeend,
                            FT,
                            direction,
                            dt,
                            n,
                        )


                        @test (
                            result[l] ≈
                            FT(expected_result[dim, l, FT, direction]) ||
                            result[l] <
                            FT(expected_result[dim, l, FT, direction])
                        )
                    end
                    @info begin
                        msg = ""
                        for l in 1:(numlevels - 1)
                            rate = log2(result[l]) - log2(result[l + 1])
                            msg *= @sprintf(
                                "\n  rate for level %d = %e\n",
                                l,
                                rate
                            )
                        end
                        msg
                    end
                end
            end
        end
    end
end

nothing
