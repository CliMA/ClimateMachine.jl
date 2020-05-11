using Test
using ClimateMachine
using LinearAlgebra

include("ode_tests_common.jl")

ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()

a = 100
b = 1
c = 1 / 100
Δ = sqrt(4 * a * c - b^2)

α1, α2 = 1 / 4, 3 / 4
β1, β2, β3 = 1 / 3, 3 / 6, 1 / 6

function rhs!(dQ, Q, ::Nothing, t; increment)
    if increment
        @. dQ += $cos(t) * (a + b * Q + c * Q^2)
    else
        @. dQ = $cos(t) * (a + b * Q + c * Q^2)
    end
end
function rhs_linear!(dQ, Q, ::Nothing, t; increment)
    if increment
        @. dQ += $cos(t) * b * Q
    else
        @. dQ = $cos(t) * b * Q
    end
end
struct ODETestBasicLinBE <: AbstractBackwardEulerSolver end
ODESolvers.Δt_is_adjustable(::ODETestBasicLinBE) = true
(::ODETestBasicLinBE)(Q, Qhat, α, p, t) = @. Q = Qhat / (1 - α * $cos(t) * b)
function rhs_nonlinear!(dQ, Q, ::Nothing, t; increment)
    if increment
        @. dQ += $cos(t) * (a + c * Q^2)
    else
        @. dQ = $cos(t) * (a + c * Q^2)
    end
end
function rhs_fast!(dQ, Q, ::Any, t; increment)
    if increment
        @. dQ += α1 * $cos(t) * (a + b * Q + c * Q^2)
    else
        @. dQ = α1 * $cos(t) * (a + b * Q + c * Q^2)
    end
end
function rhs_fast_linear!(dQ, Q, ::Nothing, t; increment)
    if increment
        @. dQ += α1 * $cos(t) * Q
    else
        @. dQ = α1 * $cos(t) * Q
    end
end
function rhs_slow!(dQ, Q, ::Nothing, t; increment)
    if increment
        @. dQ += α2 * $cos(t) * (a + b * Q + c * Q^2)
    else
        @. dQ = α2 * $cos(t) * (a + b * Q + c * Q^2)
    end
end
function rhs1!(dQ, Q, ::Any, t; increment)
    if increment
        @. dQ += β1 * $cos(t) * (a + b * Q + c * Q^2)
    else
        @. dQ = β1 * $cos(t) * (a + b * Q + c * Q^2)
    end
end
function rhs2!(dQ, Q, ::Any, t; increment)
    if increment
        @. dQ += β2 * $cos(t) * (a + b * Q + c * Q^2)
    else
        @. dQ = β2 * $cos(t) * (a + b * Q + c * Q^2)
    end
end
function rhs3!(dQ, Q, ::Nothing, t; increment)
    if increment
        @. dQ += β3 * $cos(t) * (a + b * Q + c * Q^2)
    else
        @. dQ = β3 * $cos(t) * (a + b * Q + c * Q^2)
    end
end

function exactsolution(t, q0, t0)
    k = @. 2 * atan((2 * c * q0 + b) / Δ) / Δ - sin(t0)
    solution = @. (Δ * tan((k + sin(t)) * Δ / 2) - b) / (2 * c)
    return ArrayType(solution)
end

q0 = ArrayType === Array ? [1.0] : range(-1.0, 1.0, length = 303)
t0 = 0.1
finaltime = 1.2

Qinit = exactsolution(t0, q0, t0)
Q = similar(Qinit)
Qexact = exactsolution(finaltime, q0, t0)

dts = [0.125, 0.0625]
errors = similar(dts)

@testset "Convergence/limited" begin
    @testset "Explicit methods" begin
        for (method, expected_order) in explicit_methods
            for (n, dt) in enumerate(dts)
                Q .= Qinit
                solver = method(rhs!, Q; dt = dt, t0 = t0)
                solve!(Q, solver; timeend = finaltime)
                errors[n] = norm(Q - Qexact)
            end
            rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
            @test isapprox(rates[end], expected_order; atol = 0.7)
        end
    end

    @testset "IMEX methods" begin
        for (method, order) in imex_methods
            for split_explicit_implicit in (false, true)
                for variant in (LowStorageVariant(), NaiveVariant())
                    for (n, dt) in enumerate(dts)
                        Q .= Qinit
                        rhs_arg! =
                            split_explicit_implicit ? rhs_nonlinear! : rhs!
                        solver = method(
                            rhs_arg!,
                            rhs_linear!,
                            LinearBackwardEulerSolver(
                                DivideLinearSolver();
                                isadjustable = true,
                            ),
                            Q;
                            dt = dt,
                            t0 = t0,
                            split_explicit_implicit = split_explicit_implicit,
                            variant = variant,
                        )
                        solve!(Q, solver; timeend = finaltime)
                        errors[n] = norm(Q - Qexact)
                    end
                    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                    if variant isa LowStorageVariant && split_explicit_implicit
                        expected_order = 2
                    else
                        expected_order = order
                    end
                    @test isapprox(rates[end], expected_order; atol = 0.3)
                end
            end
        end
    end

    @testset "IMEX methods with direct solver" begin
        for (method, order) in imex_methods
            for split_explicit_implicit in (false, true)
                for (n, dt) in enumerate(dts)
                    Q .= Qinit
                    rhs_arg! = split_explicit_implicit ? rhs_nonlinear! : rhs!
                    solver = method(
                        rhs_arg!,
                        rhs_linear!,
                        ODETestBasicLinBE(),
                        Q;
                        dt = dt,
                        t0 = t0,
                        split_explicit_implicit = split_explicit_implicit,
                        variant = NaiveVariant(),
                    )
                    solve!(Q, solver; timeend = finaltime)
                    errors[n] = norm(Q - Qexact)
                end
                rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                expected_order = order
                @test isapprox(rates[end], expected_order; atol = 0.3)
            end
        end
    end

    @testset "MRRK methods with 2 rates" begin
        for (slow_method, slow_expected_order) in slow_mrrk_methods
            for (fast_method, fast_expected_order) in fast_mrrk_methods
                for nsubsteps in (1, 3)
                    for (n, dt) in enumerate(dts)
                        Q .= Qinit
                        solver = MultirateRungeKutta(
                            (
                                slow_method(rhs_slow!, Q; dt = dt),
                                fast_method(rhs_fast!, Q; dt = dt / nsubsteps),
                            );
                            t0 = t0,
                        )
                        solve!(Q, solver; timeend = finaltime)
                        errors[n] = norm(Q - Qexact)
                    end

                    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                    min_order = min(slow_expected_order, fast_expected_order)
                    max_order = max(slow_expected_order, fast_expected_order)
                    @test (
                        isapprox(rates[end], min_order; atol = 0.5) ||
                        isapprox(rates[end], max_order; atol = 0.5) ||
                        min_order <= rates[end] <= max_order
                    )
                end
            end
        end
    end

    @testset "MRRK methods with IMEX" begin
        for (slow_method, slow_expected_order) in slow_mrrk_methods
            for (fast_method, fast_expected_order) in imex_methods
                for (n, dt) in enumerate(dts)
                    Q .= Qinit
                    solver = MultirateRungeKutta(
                        (
                            slow_method(rhs_slow!, Q; dt = dt),
                            fast_method(
                                rhs_fast!,
                                rhs_fast_linear!,
                                LinearBackwardEulerSolver(
                                    DivideLinearSolver();
                                    isadjustable = true,
                                ),
                                Q;
                                dt = dt,
                                split_explicit_implicit = false,
                            ),
                        ),
                        t0 = t0,
                    )
                    solve!(Q, solver; timeend = finaltime)
                    errors[n] = norm(Q - Qexact)
                end
                rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                min_order = min(slow_expected_order, fast_expected_order)
                max_order = max(slow_expected_order, fast_expected_order)
                @test (
                    isapprox(rates[end], min_order; atol = 0.5) ||
                    isapprox(rates[end], max_order; atol = 0.5) ||
                    min_order <= rates[end] <= max_order
                )
            end
        end
    end

    @testset "MRRK methods with 3 rates" begin
        for (rate3_method, rate3_order) in slow_mrrk_methods
            for (rate2_method, rate2_order) in slow_mrrk_methods
                for (rate1_method, rate1_order) in fast_mrrk_methods
                    for nsubsteps in (1, 2)
                        for (n, dt) in enumerate(dts)
                            Q .= Qinit
                            solver = MultirateRungeKutta(
                                (
                                    rate3_method(rhs3!, Q, dt = dt),
                                    rate2_method(rhs2!, Q, dt = dt / nsubsteps),
                                    rate1_method(
                                        rhs1!,
                                        Q;
                                        dt = dt / nsubsteps^2,
                                    ),
                                );
                                dt = dt,
                                t0 = t0,
                            )
                            solve!(Q, solver; timeend = finaltime)
                            errors[n] = norm(Q - Qexact)
                        end
                        rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                        @test 3.8 <= rates[end]
                    end
                end
            end
        end
    end

    @testset "MIS methods" begin
        for (method, expected_order) in mis_methods
            for fast_method in (LSRK54CarpenterKennedy,)
                for (n, dt) in enumerate(dts)
                    Q .= Qinit
                    solver = method(
                        rhs_slow!,
                        rhs_fast!,
                        fast_method,
                        4,
                        Q;
                        dt = dt,
                        t0 = t0,
                    )
                    solve!(Q, solver; timeend = finaltime)
                    errors[n] = norm(Q - Qexact)
                end
                rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                @test isapprox(rates[end], expected_order; atol = 0.6)
            end
        end
    end

    @testset "MRI GARK methods with 2 rates" begin
        for (slow_method, expected_order) in mrigark_erk_methods
            for (fast_method, _) in fast_mrigark_methods
                for nsubsteps in (1, 3)
                    for (n, dt) in enumerate(dts)
                        dt /= 4 # Need a smaller dt to get convergence rate
                        Q .= Qinit
                        fastsolver =
                            fast_method(rhs_fast!, Q; dt = dt / nsubsteps)
                        solver = slow_method(
                            rhs_slow!,
                            fastsolver,
                            Q;
                            dt = dt,
                            t0 = t0,
                        )
                        solve!(Q, solver; timeend = finaltime)

                        errors[n] = norm(Q - Qexact)
                    end

                    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                    @test isapprox(rates[end], expected_order; atol = 0.5)
                end
            end
        end
    end

    @testset "MRI GARK methods with 3 rates" begin
        for (rate3_method, rate3_order) in mrigark_erk_methods
            for (rate2_method, rate2_order) in mrigark_erk_methods
                for (rate1_method, _) in fast_mrigark_methods
                    for nsubsteps in (1, 2)
                        for (n, dt) in enumerate(dts)
                            dt /= 4 # Need a smaller dt to get convergence rate
                            Q .= Qinit
                            solver1 =
                                rate1_method(rhs1!, Q; dt = dt / nsubsteps^2)
                            solver2 = rate2_method(
                                rhs2!,
                                solver1,
                                Q;
                                dt = dt / nsubsteps,
                            )
                            solver3 = rate3_method(
                                rhs3!,
                                solver2,
                                Q;
                                dt = dt,
                                t0 = t0,
                            )
                            solve!(Q, solver3; timeend = finaltime)
                            errors[n] = norm(Q - Qexact)
                        end
                        rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                        expected_order = min(rate3_order, rate2_order)
                        @test isapprox(rates[end], expected_order; atol = 0.5)
                    end
                end
            end
        end
    end

    @testset "MRI GARK implicit methods with 2 rates and linear solver" begin
        for (slow_method, expected_order) in mrigark_irk_methods
            for (fast_method, _) in fast_mrigark_methods
                for nsubsteps in (1, 3)
                    for (n, dt) in enumerate(dts)
                        dt /= 4
                        Q .= Qinit
                        nsteps = ceil(Int, (finaltime - t0) / dt)
                        dt = (finaltime - t0) / nsteps
                        fastsolver =
                            fast_method(rhs_nonlinear!, Q; dt = dt / nsubsteps)
                        solver = slow_method(
                            rhs_linear!,
                            LinearBackwardEulerSolver(
                                DivideLinearSolver();
                                isadjustable = true,
                            ),
                            fastsolver,
                            Q;
                            dt = dt,
                            t0 = t0,
                        )
                        solve!(Q, solver; timeend = finaltime)

                        errors[n] = norm(Q - Qexact)
                    end

                    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                    @test isapprox(rates[end], expected_order; atol = 0.5)
                end
            end
        end
    end

    @testset "MRI GARK implicit methods with 2 rates and custom solver" begin
        for (slow_method, expected_order) in mrigark_irk_methods
            for (fast_method, _) in fast_mrigark_methods
                for nsubsteps in (1, 3)
                    for (n, dt) in enumerate(dts)
                        dt /= 4
                        Q .= Qinit
                        nsteps = ceil(Int, (finaltime - t0) / dt)
                        dt = (finaltime - t0) / nsteps
                        fastsolver =
                            fast_method(rhs_nonlinear!, Q; dt = dt / nsubsteps)
                        solver = slow_method(
                            rhs_linear!,
                            ODETestBasicLinBE(),
                            fastsolver,
                            Q;
                            dt = dt,
                            t0 = t0,
                        )
                        solve!(Q, solver; timeend = finaltime)

                        errors[n] = norm(Q - Qexact)
                    end

                    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                    @test isapprox(rates[end], expected_order; atol = 0.5)
                end
            end
        end
    end
end

@testset "Explicit methods composition of solve!" begin
    halftime = 1.0
    finaltime = 2.0
    dt = 0.075
    for (method, _) in explicit_methods
        Q .= Qinit
        solver1 = method(rhs!, Q; dt = dt, t0 = t0)
        solve!(Q, solver1; timeend = finaltime)

        Q2 = similar(Q)
        Q2 .= Qinit
        solver2 = method(rhs!, Q2; dt = dt, t0 = t0)
        solve!(Q2, solver2; timeend = halftime, adjustfinalstep = false)
        solve!(Q2, solver2; timeend = finaltime)

        @test Q2 == Q
    end
end
