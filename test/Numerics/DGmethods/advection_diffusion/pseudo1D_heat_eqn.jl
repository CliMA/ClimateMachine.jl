using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
import ClimateMachine.DGmethods.NumericalFluxes:
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
    # No advection
    aux.u = 0 * n

    # diffusion of strength 1 in the n direction
    aux.D = n * n'
end

# solution is such that
# u(1, t) = 1
# ∇u(0,t) = n
function initial_condition!(
    ::HeatEqn{n, κ, A},
    state,
    aux,
    x,
    t,
) where {n, κ, A}
    ξn = dot(n, x)
    state.ρ = ξn + sum(A .* cos.(κ * ξn) .* exp.(-κ .^ 2 * t))
end
Dirichlet_data!(P::HeatEqn, x...) = initial_condition!(P, x...)

function normal_boundary_flux_second_order!(
    ::CentralNumericalFluxSecondOrder,
    ::AdvectionDiffusion{dim, HeatEqn{nd, κ, A}},
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
    bctype,
    t,
    _...,
) where {S, dim, nd, κ, A}

    if bctype == 1
        fluxᵀn.ρ = -diff⁻.σ' * n⁻
    elseif bctype == 2
        # Get exact gradient of ρ
        x = aux⁻.coord
        ξn = dot(nd, x)
        ∇ρ = SVector(ntuple(
            i -> nd[i] * (1 - sum(A .* κ .* sin.(κ * ξn) .* exp.(-κ .^ 2 * t))),
            Val(3),
        ))

        # Compute flux value
        D = aux⁻.D
        fluxᵀn.ρ = -(D * ∇ρ)' * n⁻
    end
end

function run(
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
    model = AdvectionDiffusion{dim}(HeatEqn{n, κ, A}())
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
    expected_result = Dict()
    expected_result[2, 1, Float64, EveryDirection] = 5.1574832681276104e-03
    expected_result[2, 2, Float64, EveryDirection] = 6.5687731035740145e-05
    expected_result[2, 3, Float64, EveryDirection] = 1.6644861275101170e-06
    expected_result[2, 1, Float64, HorizontalDirection] = 2.0515449798977920e-02
    expected_result[2, 2, Float64, HorizontalDirection] = 5.6862569422960307e-04
    expected_result[2, 3, Float64, HorizontalDirection] = 1.0132022682546759e-05
    expected_result[2, 1, Float64, VerticalDirection] = 2.0515449798977868e-02
    expected_result[2, 2, Float64, VerticalDirection] = 5.6862569422963256e-04
    expected_result[2, 3, Float64, VerticalDirection] = 1.0132022682819374e-05
    expected_result[3, 1, Float64, EveryDirection] = 1.2605810186713056e-03
    expected_result[3, 2, Float64, EveryDirection] = 2.2149085222010158e-05
    expected_result[3, 3, Float64, EveryDirection] = 5.9317355941102274e-07
    expected_result[3, 1, Float64, HorizontalDirection] = 5.1574832681275749e-03
    expected_result[3, 2, Float64, HorizontalDirection] = 6.5687731035720697e-05
    expected_result[3, 3, Float64, HorizontalDirection] = 1.6644861273769655e-06
    expected_result[3, 1, Float64, VerticalDirection] = 2.0515449798978055e-02
    expected_result[3, 2, Float64, VerticalDirection] = 5.6862569422975464e-04
    expected_result[3, 3, Float64, VerticalDirection] = 1.0132022682813954e-05
    expected_result[2, 1, Float32, EveryDirection] = 5.1570897921919823e-03
    expected_result[2, 2, Float32, EveryDirection] = 6.5653577621560544e-05
    expected_result[2, 3, Float32, EveryDirection] = 3.2485229439771501e-06
    expected_result[2, 1, Float32, HorizontalDirection] = 2.0514581352472305e-02
    expected_result[2, 2, Float32, HorizontalDirection] = 5.6842499179765582e-04
    expected_result[2, 3, Float32, HorizontalDirection] = 1.0186861800320912e-05
    expected_result[2, 1, Float32, VerticalDirection] = 2.0514704287052155e-02
    expected_result[2, 2, Float32, VerticalDirection] = 5.6839984608814120e-04
    expected_result[2, 3, Float32, VerticalDirection] = 1.0241863492410630e-05
    expected_result[3, 1, Float32, EveryDirection] = 1.2601461494341493e-03
    expected_result[3, 2, Float32, EveryDirection] = 2.2380427253665403e-05
    expected_result[3, 3, Float32, EveryDirection] = 1.1313175491522998e-05
    expected_result[3, 1, Float32, HorizontalDirection] = 5.1570334471762180e-03
    expected_result[3, 2, Float32, HorizontalDirection] = 6.6673339460976422e-05
    expected_result[3, 3, Float32, HorizontalDirection] = 9.9301614682190120e-05
    expected_result[3, 1, Float32, VerticalDirection] = 2.0514605566859245e-02
    expected_result[3, 2, Float32, VerticalDirection] = 5.6837813463062048e-04
    expected_result[3, 3, Float32, VerticalDirection] = 3.2253094104817137e-05

    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32)
            numlevels = integration_testing ||
            ClimateMachine.Settings.integration_testing ?
                3 : 1
            result = zeros(FT, numlevels)
            for dim in 2:3
                for direction in
                    (EveryDirection, HorizontalDirection, VerticalDirection)
                    if direction <: EveryDirection
                        n = dim == 2 ?
                            SVector{3, FT}(1 / sqrt(2), 1 / sqrt(2), 0) :
                            SVector{3, FT}(
                            1 / sqrt(3),
                            1 / sqrt(3),
                            1 / sqrt(3),
                        )
                    elseif direction <: HorizontalDirection
                        n = dim == 2 ? SVector{3, FT}(1, 0, 0) :
                            SVector{3, FT}(1 / sqrt(2), 1 / sqrt(2), 0)
                    elseif direction <: VerticalDirection
                        n = dim == 2 ? SVector{3, FT}(0, 1, 0) :
                            SVector{3, FT}(0, 0, 1)
                    end
                    for l in 1:numlevels
                        Ne = 2^(l - 1) * base_num_elem
                        brickrange = ntuple(
                            j -> range(FT(0); length = Ne + 1, stop = 1),
                            dim,
                        )
                        periodicity = ntuple(j -> false, dim)
                        bc = ntuple(j -> (2, 1), dim)
                        topl = StackedBrickTopology(
                            mpicomm,
                            brickrange;
                            periodicity = periodicity,
                            boundary = bc,
                        )
                        dt = 1 / (Ne * polynomialorder^2)^2

                        timeend = 0.01

                        @info (ArrayType, FT, dim, direction)
                        result[l] = run(
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
                        @test result[l] ≈
                              FT(expected_result[dim, l, FT, direction])
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
