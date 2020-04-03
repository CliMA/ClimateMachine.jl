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

const mrrk_methods =
    ((LSRK54CarpenterKennedy, 4), (LSRK144NiegemannDiehlBusch, 4))


include("onestate_model.jl")

@testset "Multistate RK solvers" begin
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
    topl = BrickTopology(mpicomm, brickrange, periodicity = (false, false))
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



    @testset "MSRK method (no substeps)" begin
        model_fast = MultiODE{FT}(Ω, ω)
        model_slow = NullODE()
        dg_fast = DGModel(
            model_fast,
            grid,
            CentralNumericalFluxNonDiffusive(),
            CentralNumericalFluxDiffusive(),
            CentralNumericalFluxGradient(),
        )
        dg_slow = DGModel(
            model_slow,
            grid,
            CentralNumericalFluxNonDiffusive(),
            CentralNumericalFluxDiffusive(),
            CentralNumericalFluxGradient(),
        )

        finaltime = 5π / 2
        dts = [2.0^(-k) for k in 2:8]
        error = similar(dts)
        for (slow_method, slow_expected_order) in mrrk_methods
            for (fast_method, fast_expected_order) in mrrk_methods
                for (n, dt) in enumerate(dts)
                    Qᶠ = init_ode_state(dg_fast, FT(0))
                    Qˢ = init_ode_state(dg_slow, FT(0))
                    solver = MultistateRungeKutta(
                        slow_method(dg_slow, Qˢ),
                        fast_method(dg_fast, Qᶠ);
                        dt = dt,
                        t0 = 0.0,
                    )
                    Qvec = (slow = Qˢ, fast = Qᶠ)
                    solve!(Qvec, solver; timeend = finaltime)
                    @show error[n] =
                        norm(Qᶠ.data[1, :, 1] - exactsolution(ω, finaltime))
                end

                @show rate = log2.(error[1:(end - 1)] ./ error[2:end])
                min_order = min(slow_expected_order, fast_expected_order)
                max_order = max(slow_expected_order, fast_expected_order)
                @test (
                    isapprox(rate[end], min_order; atol = 0.31) ||
                    isapprox(rate[end], max_order; atol = 0.31) ||
                    min_order <= rate[end] <= max_order
                )
            end
        end
    end

    @testset "MSMRRK methods (with MultiODE model being slow)" begin
        model_fast = NullODE()
        model_slow = MultiODE{FT}(Ω, ω)
        dg_fast = DGModel(
            model_fast,
            grid,
            CentralNumericalFluxNonDiffusive(),
            CentralNumericalFluxDiffusive(),
            CentralNumericalFluxGradient(),
        )
        dg_slow = DGModel(
            model_slow,
            grid,
            CentralNumericalFluxNonDiffusive(),
            CentralNumericalFluxDiffusive(),
            CentralNumericalFluxGradient(),
        )
        finaltime = 5π / 2
        dts = [2.0^(-k) for k in 2:8]
        error = similar(dts)
        for (slow_method, slow_expected_order) in mrrk_methods
            for (fast_method, fast_expected_order) in mrrk_methods
                for (n, slow_dt) in enumerate(dts)
                    fast_dt = slow_dt
                    Qᶠ = init_ode_state(dg_fast, FT(0))
                    Qˢ = init_ode_state(dg_slow, FT(0))
                    solver = MultistateMultirateRungeKutta(
                        slow_method(dg_slow, Qˢ; dt = slow_dt),
                        fast_method(dg_fast, Qᶠ; dt = fast_dt),
                    )
                    Qvec = (slow = Qˢ, fast = Qᶠ)
                    solve!(Qvec, solver; timeend = finaltime)
                    @show error[n] =
                        norm(Qˢ.data[1, :, 1] - exactsolution(ω, finaltime))
                end


                @show rate = log2.(error[1:(end - 1)] ./ error[2:end])
                min_order = min(slow_expected_order, fast_expected_order)
                max_order = max(slow_expected_order, fast_expected_order)
                @test (
                    isapprox(rate[end], min_order; atol = 0.31) ||
                    isapprox(rate[end], max_order; atol = 0.31) ||
                    min_order <= rate[end] <= max_order
                )
            end
        end
    end

    @testset "MSMRRK methods (with MultiODE model being fast)" begin
        model_fast = MultiODE{FT}(Ω, ω)
        model_slow = NullODE()
        dg_fast = DGModel(
            model_fast,
            grid,
            CentralNumericalFluxNonDiffusive(),
            CentralNumericalFluxDiffusive(),
            CentralNumericalFluxGradient(),
        )
        dg_slow = DGModel(
            model_slow,
            grid,
            CentralNumericalFluxNonDiffusive(),
            CentralNumericalFluxDiffusive(),
            CentralNumericalFluxGradient(),
        )
        finaltime = 5π / 2
        dts = [2.0^(-k) for k in 2:10]
        error = similar(dts)
        for (slow_method, slow_expected_order) in mrrk_methods
            for (fast_method, fast_expected_order) in mrrk_methods
                for (n, fast_dt) in enumerate(dts)
                    slow_dt = ω * fast_dt
                    Qᶠ = init_ode_state(dg_fast, FT(0))
                    Qˢ = init_ode_state(dg_slow, FT(0))
                    solver = MultistateMultirateRungeKutta(
                        slow_method(dg_slow, Qˢ; dt = slow_dt),
                        fast_method(dg_fast, Qᶠ; dt = fast_dt),
                    )
                    Qvec = (slow = Qˢ, fast = Qᶠ)
                    solve!(Qvec, solver; timeend = finaltime)
                    @show error[n] =
                        norm(Qᶠ.data[1, :, 1] - exactsolution(ω, finaltime))
                end


                @show rate = log2.(error[1:(end - 1)] ./ error[2:end])
                min_order = min(slow_expected_order, fast_expected_order)
                max_order = max(slow_expected_order, fast_expected_order)
                @test (
                    isapprox(rate[end], min_order; atol = 1.0) ||
                    isapprox(rate[end], max_order; atol = 1.0) ||
                    min_order <= rate[end] <= max_order
                )
            end
        end
    end
end
