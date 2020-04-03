using MPI
using Test
using Logging
using StaticArrays
using LinearAlgebra: norm

using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers

const explicit_methods = (
    (LSRK54CarpenterKennedy, 4),
    (LSRK144NiegemannDiehlBusch, 4),
    (SSPRK33ShuOsher, 3),
    (SSPRK34SpiteriRuuth, 3),
    (LSRKEulerMethod, 1),
)

include("onestate_model.jl")

@testset "Onestate RK solvers" begin
    CLIMA.init()
    ArrayType = CLIMA.array_type()

    mpicomm = MPI.COMM_WORLD
    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
        ll == "WARN" ? Logging.Warn :
        ll == "ERROR" ? Logging.Error : Logging.Info
    logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

    FT = Float64

    brickrange = (
        range(FT(0); length = 2, stop = FT(1)),
        range(FT(0); length = 2, stop = FT(1)),
    )
    topl = BrickTopology(mpicomm, brickrange, periodicity = (true, true))
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = 1,
    )

    ω = 100
    λf = -10
    λs = -1
    ξ = 1 // 10
    α = 1
    ηfs = ((1 - ξ) / α) * (λf - λs)
    ηsf = -ξ * α * (λf - λs)
    Ω = @SMatrix [
        λf ηfs
        ηsf λs
    ]

    model = MultiODE{FT}(Ω, ω)

    dg = DGModel(
        model,
        grid,
        CentralNumericalFluxNonDiffusive(),
        CentralNumericalFluxDiffusive(),
        CentralNumericalFluxGradient(),
    )

    @testset "Explicit Methods convergence" begin
        finaltime = 5π / 2
        dts = [2.0^(-k) for k in 2:8]
        errors = similar(dts)
        for (method, expected_order) in explicit_methods
            for (n, dt) in enumerate(dts)
                Q = init_ode_state(dg, FT(0))
                solver = method(dg, Q; dt = dt, t0 = 0.0)
                solve!(Q, solver; timeend = finaltime)
                @show errors[n] =
                    norm(Q.data[1, :, 1] - exactsolution(ω, finaltime))
            end

            @show rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
            @test isapprox(rates[end], expected_order; atol = 0.31)
        end
    end
end
