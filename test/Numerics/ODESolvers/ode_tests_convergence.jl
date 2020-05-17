using Test
using ClimateMachine
using StaticArrays
using LinearAlgebra
using KernelAbstractions
using ClimateMachine.MPIStateArrays: device

include("ode_tests_common.jl")

ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()

@testset "ODE Solvers" begin
    @testset "Convergence/extensive" begin
        @testset "1-rate ODE" begin
            function rhs!(dQ, Q, ::Nothing, time; increment)
                if increment
                    dQ .+= Q * cos(time)
                else
                    dQ .= Q * cos(time)
                end
            end
            exactsolution(q0, time) = q0 * exp(sin(time))

            @testset "Explicit methods" begin
                finaltime = 20.0
                dts = [2.0^(-k) for k in 0:7]
                errors = similar(dts)
                q0 =
                    ArrayType === Array ? [1.0] : range(-1.0, 1.0, length = 303)
                for (method, expected_order) in explicit_methods
                    for (n, dt) in enumerate(dts)
                        Q = ArrayType(q0)
                        solver = method(rhs!, Q; dt = dt, t0 = 0.0)
                        solve!(Q, solver; timeend = finaltime)
                        Q = Array(Q)
                        errors[n] =
                            maximum(@. abs(Q - exactsolution(q0, finaltime)))
                    end
                    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                    @test isapprox(rates[end], expected_order; atol = 0.17)
                end
            end
        end

        @testset "Two-rate ODE with a linear stiff part" begin
            c = 100.0
            function rhs_full!(dQ, Q, ::Nothing, time; increment)
                if increment
                    dQ .+= im * c * Q .+ exp(im * time)
                else
                    dQ .= im * c * Q .+ exp(im * time)
                end
            end

            function rhs_nonlinear!(dQ, Q, ::Nothing, time; increment)
                if increment
                    dQ .+= exp(im * time)
                else
                    dQ .= exp(im * time)
                end
            end
            rhs_slow! = rhs_nonlinear!

            function rhs_linear!(dQ, Q, ::Nothing, time; increment)
                if increment
                    dQ .+= im * c * Q
                else
                    dQ .= im * c * Q
                end
            end
            rhs_fast! = rhs_linear!

            function exactsolution(q0, time)
                q0 * exp(im * c * time) +
                (exp(im * time) - exp(im * c * time)) / (im * (1 - c))
            end

            @testset "IMEX methods" begin
                finaltime = pi / 2
                dts = [2.0^(-k) for k in 2:13]
                errors = similar(dts)

                q0 = ArrayType <: Array ? [1.0] : range(-1.0, 1.0, length = 303)
                for (method, expected_order) in imex_methods
                    for split_explicit_implicit in (false, true)
                        for variant in (LowStorageVariant(), NaiveVariant())
                            for (n, dt) in enumerate(dts)
                                Q = ArrayType{ComplexF64}(q0)
                                rhs! =
                                    split_explicit_implicit ? rhs_nonlinear! :
                                    rhs_full!
                                solver = method(
                                    rhs!,
                                    rhs_linear!,
                                    LinearBackwardEulerSolver(
                                        DivideLinearSolver(),
                                        isadjustable = true,
                                    ),
                                    Q;
                                    dt = dt,
                                    t0 = 0.0,
                                    split_explicit_implicit = split_explicit_implicit,
                                    variant = variant,
                                )
                                solve!(Q, solver; timeend = finaltime)
                                Q = Array(Q)
                                errors[n] = maximum(@. abs(
                                    Q - exactsolution(q0, finaltime),
                                ))
                            end

                            rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                            @test errors[1] < 2.0
                            @test isapprox(
                                rates[end],
                                expected_order;
                                atol = 0.1,
                            )
                        end
                    end
                end
            end

            @testset "MRRK methods (no substeps)" begin
                finaltime = pi / 2
                dts = [2.0^(-k) for k in 2:11]
                errors = similar(dts)
                for (slow_method, slow_expected_order) in slow_mrrk_methods
                    for (fast_method, fast_expected_order) in fast_mrrk_methods
                        q0 = ArrayType === Array ? [1.0] :
                            range(-1.0, 1.0, length = 303)
                        for (n, dt) in enumerate(dts)
                            Q = ArrayType{ComplexF64}(q0)
                            solver = MultirateRungeKutta(
                                (
                                    slow_method(rhs_slow!, Q),
                                    fast_method(rhs_fast!, Q),
                                );
                                dt = dt,
                                t0 = 0.0,
                            )
                            solve!(Q, solver; timeend = finaltime)
                            Q = Array(Q)
                            errors[n] = maximum(@. abs(
                                Q - exactsolution(q0, finaltime),
                            ))
                        end

                        rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                        min_order =
                            min(slow_expected_order, fast_expected_order)
                        max_order =
                            max(slow_expected_order, fast_expected_order)
                        @test (
                            isapprox(rates[end], min_order; atol = 0.1) ||
                            isapprox(rates[end], max_order; atol = 0.1) ||
                            min_order <= rates[end] <= max_order
                        )
                    end
                end
            end

            @testset "MRRK methods (with substeps)" begin
                finaltime = pi / 2
                dts = [2.0^(-k) for k in 2:15]
                errors = similar(dts)
                for (slow_method, slow_expected_order) in slow_mrrk_methods
                    for (fast_method, fast_expected_order) in fast_mrrk_methods
                        q0 = ArrayType === Array ? [1.0] :
                            range(-1.0, 1.0, length = 303)
                        for (n, fast_dt) in enumerate(dts)
                            slow_dt = c * fast_dt
                            Q = ArrayType{ComplexF64}(q0)
                            solver = MultirateRungeKutta((
                                slow_method(rhs_slow!, Q; dt = slow_dt),
                                fast_method(rhs_fast!, Q; dt = fast_dt),
                            ))
                            solve!(Q, solver; timeend = finaltime)
                            Q = Array(Q)
                            errors[n] = maximum(@. abs(
                                Q - exactsolution(q0, finaltime),
                            ))
                        end

                        rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                        min_order =
                            min(slow_expected_order, fast_expected_order)
                        max_order =
                            max(slow_expected_order, fast_expected_order)
                        @test (
                            isapprox(rates[end], min_order; atol = 0.1) ||
                            isapprox(rates[end], max_order; atol = 0.1) ||
                            min_order <= rates[end] <= max_order
                        )
                    end
                end
            end

            @testset "MIS methods (with substeps)" begin
                finaltime = pi / 2
                dts = [2.0^(-k) for k in 2:11]
                errors = similar(dts)
                for (mis_method, mis_expected_order) in mis_methods
                    for fast_method in (LSRK54CarpenterKennedy,)
                        q0 = ArrayType === Array ? [1.0] :
                            range(-1.0, 1.0, length = 303)
                        for (n, dt) in enumerate(dts)
                            Q = ArrayType{ComplexF64}(q0)
                            solver = mis_method(
                                rhs_slow!,
                                rhs_fast!,
                                fast_method,
                                4,
                                Q;
                                dt = dt,
                                t0 = 0.0,
                            )
                            solve!(Q, solver; timeend = finaltime)
                            Q = Array(Q)
                            errors[n] = maximum(@. abs(
                                Q - exactsolution(q0, finaltime),
                            ))
                        end
                        rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
                        @test isapprox(
                            rates[end],
                            mis_expected_order;
                            atol = 0.1,
                        )
                    end
                end
            end
        end

        #=
        Test problem (4.2) from RobertsSarsharSandu2018arxiv
        @article{RobertsSarsharSandu2018arxiv,
          title={Coupled Multirate Infinitesimal GARK Schemes for Stiff Systems with
                 Multiple Time Scales},
          author={Roberts, Steven and Sarshar, Arash and Sandu, Adrian},
          journal={arXiv preprint arXiv:1812.00808},
          year={2019}
        }

        Note: The actual rates are all over the place with this test and passing largely
              depends on final dt size
        =#
        @testset "2-rate ODE from RobertsSarsharSandu2018arxiv" begin
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

            function rhs_fast!(dQ, Q, param, t; increment)
                @inbounds begin
                    increment || (dQ .= 0)
                    yf = Q[1]
                    ys = Q[2]
                    gf = (-3 + yf^2 - cos(ω * t)) / 2yf
                    gs = (-2 + ys^2 - cos(t)) / 2ys
                    dQ[1] += Ω[1, 1] * gf + Ω[1, 2] * gs - ω * sin(ω * t) / 2yf
                end
            end

            function rhs_slow!(dQ, Q, param, t; increment)
                @inbounds begin
                    increment || (dQ .= 0)
                    yf = Q[1]
                    ys = Q[2]
                    gf = (-3 + yf^2 - cos(ω * t)) / 2yf
                    gs = (-2 + ys^2 - cos(t)) / 2ys
                    dQ[2] += Ω[2, 1] * gf + Ω[2, 2] * gs - sin(t) / 2ys
                end
            end

            exactsolution(t) = [sqrt(3 + cos(ω * t)); sqrt(2 + cos(t))]

            @testset "MRRK methods (no substeps)" begin
                finaltime = 5π / 2
                dts = [2.0^(-k) for k in 2:8]
                error = similar(dts)
                for (slow_method, slow_expected_order) in slow_mrrk_methods
                    for (fast_method, fast_expected_order) in fast_mrrk_methods
                        for (n, dt) in enumerate(dts)
                            Q = exactsolution(0)
                            solver = MultirateRungeKutta(
                                (
                                    slow_method(rhs_slow!, Q),
                                    fast_method(rhs_fast!, Q),
                                );
                                dt = dt,
                                t0 = 0.0,
                            )
                            solve!(Q, solver; timeend = finaltime)
                            error[n] = norm(Q - exactsolution(finaltime))
                        end

                        rate = log2.(error[1:(end - 1)] ./ error[2:end])
                        min_order =
                            min(slow_expected_order, fast_expected_order)
                        max_order =
                            max(slow_expected_order, fast_expected_order)
                        @test (
                            isapprox(rate[end], min_order; atol = 0.3) ||
                            isapprox(rate[end], max_order; atol = 0.3) ||
                            min_order <= rate[end] <= max_order
                        )
                    end
                end
            end

            @testset "MRRK methods (with substeps)" begin
                finaltime = 5π / 2
                dts = [2.0^(-k) for k in 2:9]
                error = similar(dts)
                for (slow_method, slow_expected_order) in slow_mrrk_methods
                    for (fast_method, fast_expected_order) in fast_mrrk_methods
                        for (n, fast_dt) in enumerate(dts)
                            Q = exactsolution(0)
                            slow_dt = ω * fast_dt
                            solver = MultirateRungeKutta((
                                slow_method(rhs_slow!, Q; dt = slow_dt),
                                fast_method(rhs_fast!, Q; dt = fast_dt),
                            ))
                            solve!(Q, solver; timeend = finaltime)
                            error[n] = norm(Q - exactsolution(finaltime))
                        end

                        rate = log2.(error[1:(end - 1)] ./ error[2:end])
                        min_order =
                            min(slow_expected_order, fast_expected_order)
                        max_order =
                            max(slow_expected_order, fast_expected_order)
                        @test (
                            isapprox(rate[end], min_order; atol = 0.3) ||
                            isapprox(rate[end], max_order; atol = 0.3) ||
                            min_order <= rate[end] <= max_order
                        )
                    end
                end
            end

            @testset "MRRK methods with IMEX fast solver" begin
                function rhs_zero!(dQ, Q, param, t; increment)
                    if !increment
                        dQ .= 0
                    end
                end

                finaltime = 5π / 2
                dts = [2.0^(-k) for k in 2:9]
                error = similar(dts)
                for (slow_method, slow_expected_order) in slow_mrrk_methods
                    for (fast_method, fast_expected_order) in imex_methods
                        for (n, fast_dt) in enumerate(dts)
                            Q = exactsolution(0)
                            slow_dt = ω * fast_dt
                            solver = MultirateRungeKutta((
                                slow_method(rhs_slow!, Q; dt = slow_dt),
                                fast_method(
                                    rhs_fast!,
                                    rhs_zero!,
                                    LinearBackwardEulerSolver(
                                        DivideLinearSolver(),
                                        isadjustable = true,
                                    ),
                                    Q;
                                    dt = fast_dt,
                                    split_explicit_implicit = false,
                                ),
                            ))
                            solve!(Q, solver; timeend = finaltime)
                            error[n] = norm(Q - exactsolution(finaltime))
                        end

                        rate = log2.(error[1:(end - 1)] ./ error[2:end])
                        min_order =
                            min(slow_expected_order, fast_expected_order)
                        max_order =
                            max(slow_expected_order, fast_expected_order)
                        atol = fast_method == ARK2GiraldoKellyConstantinescu ?
                            0.5 : 0.37
                        @test (
                            isapprox(rate[end], min_order; atol = atol) ||
                            isapprox(rate[end], max_order; atol = atol) ||
                            min_order <= rate[end] <= max_order
                        )
                    end
                end
            end

            @testset "MIS methods (with substeps)" begin
                finaltime = 5π / 2
                dts = [2.0^(-k) for k in 2:10]
                error = similar(dts)
                for (mis_method, mis_expected_order) in mis_methods
                    for fast_method in (LSRK54CarpenterKennedy,)
                        for (n, dt) in enumerate(dts)
                            Q = exactsolution(0)
                            solver = mis_method(
                                rhs_slow!,
                                rhs_fast!,
                                fast_method,
                                4,
                                Q;
                                dt = dt,
                                t0 = 0.0,
                            )
                            solve!(Q, solver; timeend = finaltime)
                            error[n] = norm(Q - exactsolution(finaltime))
                        end

                        rate = log2.(error[1:(end - 1)] ./ error[2:end])
                        @test isapprox(
                            rate[end],
                            mis_expected_order;
                            atol = 0.1,
                        )
                    end
                end
            end
        end

        # Simple 3-rate problem based on test of RobertsSarsharSandu2018arxiv
        #
        # NOTE: Since we have no theory to say this ODE solver is accurate, the rates
        #      suggest that things are really only 2nd order.
        #
        # TODO: This is not great, but no theory to say we should be accurate!
        @testset "3-rate ODE" begin
            ω1, ω2, ω3 = 10000, 100, 1
            λ1, λ2, λ3 = -100, -10, -1
            β1, β2, β3 = 2, 3, 4

            ξ12 = λ2 / λ1
            ξ13 = λ3 / λ1
            ξ23 = λ3 / λ2

            α12, α13, α23 = 1, 1, 1

            η12 = ((1 - ξ12) / α12) * (λ1 - λ2)
            η13 = ((1 - ξ13) / α13) * (λ1 - λ3)
            η23 = ((1 - ξ23) / α23) * (λ2 - λ3)

            η21 = ξ12 * α12 * (λ2 - λ1)
            η31 = ξ13 * α13 * (λ3 - λ1)
            η32 = ξ23 * α23 * (λ3 - λ2)

            Ω = @SMatrix [
                λ1 η12 η13
                η21 λ2 η23
                η31 η32 λ3
            ]

            function rhs1!(dQ, Q, param, t; increment)
                @inbounds begin
                    increment || (dQ .= 0)
                    y1, y2, y3 = Q[1], Q[2], Q[3]
                    g = @SVector [
                        (-β1 + y1^2 - cos(ω1 * t)) / 2y1,
                        (-β2 + y2^2 - cos(ω2 * t)) / 2y2,
                        (-β3 + y3^2 - cos(ω3 * t)) / 2y3,
                    ]
                    dQ[1] += Ω[1, :]' * g - ω1 * sin(ω1 * t) / 2y1
                end
            end
            function rhs2!(dQ, Q, param, t; increment)
                @inbounds begin
                    increment || (dQ .= 0)
                    y1, y2, y3 = Q[1], Q[2], Q[3]
                    g = @SVector [
                        (-β1 + y1^2 - cos(ω1 * t)) / 2y1,
                        (-β2 + y2^2 - cos(ω2 * t)) / 2y2,
                        (-β3 + y3^2 - cos(ω3 * t)) / 2y3,
                    ]
                    dQ[2] += Ω[2, :]' * g - ω2 * sin(ω2 * t) / 2y2
                end
            end
            function rhs3!(dQ, Q, param, t; increment)
                @inbounds begin
                    increment || (dQ .= 0)
                    y1, y2, y3 = Q[1], Q[2], Q[3]
                    g = @SVector [
                        (-β1 + y1^2 - cos(ω1 * t)) / 2y1,
                        (-β2 + y2^2 - cos(ω2 * t)) / 2y2,
                        (-β3 + y3^2 - cos(ω3 * t)) / 2y3,
                    ]
                    dQ[3] += Ω[3, :]' * g - ω3 * sin(ω3 * t) / 2y3
                end
            end
            function rhs12!(dQ, Q, param, t; increment)
                rhs1!(dQ, Q, param, t; increment = increment)
                rhs2!(dQ, Q, param, t; increment = true)
            end

            exactsolution(t) = [
                sqrt(β1 + cos(ω1 * t)),
                sqrt(β2 + cos(ω2 * t)),
                sqrt(β3 + cos(ω3 * t)),
            ]

            @testset "MRRK methods (no substeps)" begin
                finaltime = π / 2
                dts = [2.0^(-k) for k in 2:10]
                error = similar(dts)
                for (rate3_method, rate3_order) in slow_mrrk_methods
                    for (rate2_method, rate2_order) in slow_mrrk_methods
                        for (rate1_method, rate1_order) in fast_mrrk_methods
                            for (n, dt) in enumerate(dts)
                                Q = exactsolution(0)
                                solver = MultirateRungeKutta(
                                    (
                                        rate3_method(rhs3!, Q),
                                        rate2_method(rhs2!, Q),
                                        rate1_method(rhs1!, Q),
                                    );
                                    dt = dt,
                                    t0 = 0.0,
                                )
                                solve!(Q, solver; timeend = finaltime)
                                error[n] = norm(Q - exactsolution(finaltime))
                            end

                            rate = log2.(error[1:(end - 1)] ./ error[2:end])
                            min_order =
                                min(rate3_order, rate2_order, rate1_order)
                            max_order =
                                max(rate3_order, rate2_order, rate1_order)
                            @test 2 <= rate[end]
                        end
                    end
                end
            end

            @testset "MRRK methods (with substeps)" begin
                finaltime = π / 2
                dts = [2.0^(-k) for k in 10:17]
                error = similar(dts)
                for (rate3_method, rate3_order) in slow_mrrk_methods
                    for (rate2_method, rate2_order) in slow_mrrk_methods
                        for (rate1_method, rate1_order) in fast_mrrk_methods
                            for (n, dt1) in enumerate(dts)
                                Q = exactsolution(0)
                                dt2 = (ω1 / ω2) * dt1
                                dt3 = (ω2 / ω3) * dt2
                                solver = MultirateRungeKutta((
                                    rate3_method(rhs3!, Q; dt = dt3),
                                    rate2_method(rhs2!, Q; dt = dt2),
                                    rate1_method(rhs1!, Q; dt = dt1),
                                ))
                                solve!(Q, solver; timeend = finaltime)
                                error[n] = norm(Q - exactsolution(finaltime))
                            end

                            rate = log2.(error[1:(end - 1)] ./ error[2:end])
                            min_order =
                                min(rate3_order, rate2_order, rate1_order)
                            max_order =
                                max(rate3_order, rate2_order, rate1_order)

                            @test 2 <= rate[end]
                        end
                    end
                end
            end
        end

        #=
        Test problem (8.2) from Sandu (2019) for MRI-GARK Schemes
            @article{Sandu2019,
                title={A class of multirate infinitesimal gark methods},
                author={Sandu, Adrian},
                journal={SIAM Journal on Numerical Analysis},
                volume={57},
                number={5},
                pages={2300--2327},
                year={2019},
                publisher={SIAM},
                doi={10.1137/18M1205492}
            }
        =#
        @testset "2-rate problem" begin
            ω = 20
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

            function rhs_fast!(dQ, Q, param, t; increment)
                @kernel function knl!(dQ, Q, t, increment)
                    @inbounds begin
                        increment || (dQ .= 0)
                        yf = Q[1]
                        ys = Q[2]
                        gf = (-3 + yf^2 - cos(ω * t)) / 2yf
                        gs = (-2 + ys^2 - cos(t)) / 2ys
                        dQ[1] +=
                            Ω[1, 1] * gf + Ω[1, 2] * gs - ω * sin(ω * t) / 2yf
                    end
                end
                event = Event(device(Q))
                event = knl!(device(Q), 1)(
                    dQ,
                    Q,
                    t,
                    increment;
                    ndrange = 1,
                    dependencies = (event,),
                )
                wait(device(Q), event)
            end


            function rhs_slow!(dQ, Q, param, t; increment)
                @kernel function knl!(dQ, Q, t, increment)
                    @inbounds begin
                        increment || (dQ .= 0)
                        yf = Q[1]
                        ys = Q[2]
                        gf = (-3 + yf^2 - cos(ω * t)) / 2yf
                        gs = (-2 + ys^2 - cos(t)) / 2ys
                        dQ[2] += Ω[2, 1] * gf + Ω[2, 2] * gs - sin(t) / 2ys
                    end
                end
                event = Event(device(Q))
                event = knl!(device(Q), 1)(
                    dQ,
                    Q,
                    t,
                    increment;
                    ndrange = 1,
                    dependencies = (event,),
                )
                wait(device(Q), event)
            end

            struct ODETestConvNonLinBE <: AbstractBackwardEulerSolver end
            ODESolvers.Δt_is_adjustable(::ODETestConvNonLinBE) = true
            function (::ODETestConvNonLinBE)(Q, Qhat, α, p, t)
                @kernel function knl!(Q, Qhat, α, p, t)
                    @inbounds begin
                        # Slow RHS has zero tendency of yf so just copy Qhat
                        Q[1] = yf = Qhat[1]

                        gf = (-3 + yf^2 - cos(ω * t)) / 2yf

                        # solves: Q = Qhat[2] + α * rhs_slow(Q, t)
                        # (simplifies to a quadratic equation)
                        a = 2 - α * Ω[2, 2]
                        b = -2 * (Qhat[2] + α * Ω[2, 1] * gf)
                        c = α * (Ω[2, 2] * (2 + cos(t)) + sin(t))
                        Q[2] = (-b + sqrt(b^2 - 4 * a * c)) / (2a)
                    end
                end
                event = Event(device(Q))
                event = knl!(device(Q), 1)(
                    Q,
                    Qhat,
                    α,
                    p,
                    t;
                    ndrange = 1,
                    dependencies = (event,),
                )
                wait(device(Q), event)
            end

            exactsolution(t) =
                ArrayType([sqrt(3 + cos(ω * t)); sqrt(2 + cos(t))])

            finaltime = 1
            dts = [2.0^(-k) for k in 2:7]
            error = similar(dts)
            @testset "Explicit" begin
                for (mri_method, mri_expected_order) in mrigark_erk_methods
                    for (fast_method, fast_expected_order) in
                        fast_mrigark_methods
                        for (n, slow_dt) in enumerate(dts)
                            Q = exactsolution(0)
                            fast_dt = slow_dt / ω
                            fastsolver = fast_method(rhs_fast!, Q; dt = fast_dt)
                            solver = mri_method(
                                rhs_slow!,
                                fastsolver,
                                Q,
                                dt = slow_dt,
                            )
                            solve!(Q, solver; timeend = finaltime)
                            error[n] = norm(Q - exactsolution(finaltime))
                        end

                        rate = log2.(error[1:(end - 1)] ./ error[2:end])
                        order = mri_expected_order
                        @test isapprox(
                            rate[end],
                            mri_expected_order;
                            atol = 0.3,
                        )
                    end
                end
            end

            @testset "Implicit" begin
                for (mri_method, mri_expected_order) in mrigark_irk_methods
                    for (fast_method, fast_expected_order) in
                        fast_mrigark_methods
                        for (n, slow_dt) in enumerate(dts)
                            Q = exactsolution(0)
                            fast_dt = slow_dt / ω
                            fastsolver = fast_method(rhs_fast!, Q; dt = fast_dt)
                            solver = mri_method(
                                rhs_slow!,
                                ODETestConvNonLinBE(),
                                fastsolver,
                                Q,
                                dt = slow_dt,
                            )
                            solve!(Q, solver; timeend = finaltime)
                            error[n] = norm(Q - exactsolution(finaltime))
                        end

                        rate = log2.(error[1:(end - 1)] ./ error[2:end])
                        order = mri_expected_order
                        @test isapprox(
                            rate[end],
                            mri_expected_order;
                            atol = 0.3,
                        )
                    end
                end
            end

            @testset "IMEX" begin
                for (method, expected_order) in imex_methods
                    for (n, dt) in enumerate(dts)
                        Q = exactsolution(0)
                        solver = method(
                            rhs_fast!,
                            rhs_slow!,
                            ODETestConvNonLinBE(),
                            Q;
                            dt = dt,
                            t0 = 0.0,
                            split_explicit_implicit = true,
                            variant = NaiveVariant(),
                        )
                        solve!(Q, solver; timeend = finaltime)
                        error[n] = norm(Q - exactsolution(finaltime))
                    end
                    rate = log2.(error[1:(end - 1)] ./ error[2:end])
                    order = expected_order
                    @test isapprox(rate[end], expected_order; atol = 0.3)
                end
            end
        end

        # 3-rate modification of the above test problem
        @testset "3-rate problem" begin
            ω1, ω2, ω3 = 20, 5, 1
            λ1, λ2, λ3 = -20, -5, -1
            β1, β2, β3 = 2, 2, 2

            ξ12 = λ2 / (λ1 + λ2)
            ξ13 = λ3 / (λ1 + λ3)
            ξ23 = λ3 / (λ2 + λ3)

            α12, α13, α23 = 1, 1, 1

            η12 = ((1 - ξ12) / α12) * (λ1 - λ2)
            η13 = 0#((1 - ξ13) / α13) * (λ1 - λ3)
            η23 = ((1 - ξ23) / α23) * (λ2 - λ3)

            η21 = ξ12 * α12 * (λ2 - λ1)
            η31 = 0#ξ13 * α13 * (λ3 - λ1)
            η32 = ξ23 * α23 * (λ3 - λ2)

            Ω = @SMatrix [
                λ1 η12 η13
                η21 λ2 η23
                η31 η32 λ3
            ]

            function rhs1!(dQ, Q, param, t; increment)
                @kernel function knl!(dQ, Q, t, increment)
                    @inbounds begin
                        increment || (dQ .= 0)
                        y1, y2, y3 = Q[1], Q[2], Q[3]
                        g = @SVector [
                            (-β1 + y1^2 - cos(ω1 * t)) / 2y1,
                            (-β2 + y2^2 - cos(ω2 * t)) / 2y2,
                            (-β3 + y3^2 - cos(ω3 * t)) / 2y3,
                        ]
                        dQ[1] += Ω[1, :]' * g - ω1 * sin(ω1 * t) / 2y1
                    end
                end
                event = Event(device(Q))
                event = knl!(device(Q), 1)(
                    dQ,
                    Q,
                    t,
                    increment;
                    ndrange = 1,
                    dependencies = (event,),
                )
                wait(device(Q), event)
            end
            function rhs2!(dQ, Q, param, t; increment)
                @kernel function knl!(dQ, Q, t, increment)
                    @inbounds begin
                        increment || (dQ .= 0)
                        y1, y2, y3 = Q[1], Q[2], Q[3]
                        g = @SVector [
                            (-β1 + y1^2 - cos(ω1 * t)) / 2y1,
                            (-β2 + y2^2 - cos(ω2 * t)) / 2y2,
                            (-β3 + y3^2 - cos(ω3 * t)) / 2y3,
                        ]
                        dQ[2] += Ω[2, :]' * g - ω2 * sin(ω2 * t) / 2y2
                    end
                end
                event = Event(device(Q))
                event = knl!(device(Q), 1)(
                    dQ,
                    Q,
                    t,
                    increment;
                    ndrange = 1,
                    dependencies = (event,),
                )
                wait(device(Q), event)
            end
            function rhs3!(dQ, Q, param, t; increment)
                @kernel function knl!(dQ, Q, t, increment)
                    @inbounds begin
                        increment || (dQ .= 0)
                        y1, y2, y3 = Q[1], Q[2], Q[3]
                        g = @SVector [
                            (-β1 + y1^2 - cos(ω1 * t)) / 2y1,
                            (-β2 + y2^2 - cos(ω2 * t)) / 2y2,
                            (-β3 + y3^2 - cos(ω3 * t)) / 2y3,
                        ]
                        dQ[3] += Ω[3, :]' * g - ω3 * sin(ω3 * t) / 2y3
                    end
                end
                event = Event(device(Q))
                event = knl!(device(Q), 1)(
                    dQ,
                    Q,
                    t,
                    increment;
                    ndrange = 1,
                    dependencies = (event,),
                )
                wait(device(Q), event)
            end
            struct ODETestConvNonLinBE3Rate <: AbstractBackwardEulerSolver end
            ODESolvers.Δt_is_adjustable(::ODETestConvNonLinBE3Rate) = true
            function (::ODETestConvNonLinBE3Rate)(Q, Qhat, α, p, t)
                @kernel function knl!(Q, Qhat, α, p, t)
                    @inbounds begin
                        # Slower RHS has zero tendency of yf so just copy Qhat
                        Q[1] = y1 = Qhat[1]
                        Q[2] = y2 = Qhat[2]

                        g = @SVector [
                            (-β1 + y1^2 - cos(ω1 * t)) / 2y1,
                            (-β2 + y2^2 - cos(ω2 * t)) / 2y2,
                        ]

                        # solves: Q = Qhat + α * rhs_slow(Q, t)
                        # (simplifies to a quadratic equation)
                        a = 2 - α * Ω[3, 3]
                        b =
                            -2 *
                            (Qhat[3] + α * Ω[3, 1] * g[1] + α * Ω[3, 2] * g[2])
                        c = α * (Ω[3, 3] * (β3 + cos(t)) + sin(t))
                        Q[3] = (-b + sqrt(b^2 - 4 * a * c)) / (2a)
                    end
                end
                event = Event(device(Q))
                event = knl!(device(Q), 1)(
                    Q,
                    Qhat,
                    α,
                    p,
                    t;
                    ndrange = 1,
                    dependencies = (event,),
                )
                wait(device(Q), event)
            end

            exactsolution(t) = ArrayType([
                sqrt(β1 + cos(ω1 * t)),
                sqrt(β2 + cos(ω2 * t)),
                sqrt(β3 + cos(ω3 * t)),
            ])

            finaltime = 1
            dts = [2.0^(-k) for k in 1:7]
            error = similar(dts)
            @testset "Explicit" begin
                for (slow_method, slow_order) in mrigark_erk_methods
                    for (mid_method, mid_order) in mrigark_erk_methods
                        for (fast_method, fast_order) in fast_mrigark_methods
                            for (n, slow_dt) in enumerate(dts)
                                Q = exactsolution(0)
                                mid_dt = slow_dt / ω2
                                fast_dt = slow_dt / ω1
                                fastsolver = fast_method(rhs1!, Q; dt = fast_dt)
                                midsolver = mid_method(
                                    rhs2!,
                                    fastsolver,
                                    Q,
                                    dt = mid_dt,
                                )
                                slowsolver = slow_method(
                                    rhs3!,
                                    midsolver,
                                    Q,
                                    dt = slow_dt,
                                )
                                solve!(Q, slowsolver; timeend = finaltime)
                                error[n] = norm(Q - exactsolution(finaltime))
                            end

                            rate = log2.(error[1:(end - 1)] ./ error[2:end])
                            expected_order =
                                min(slow_order, mid_order, fast_order)
                            @test isapprox(
                                rate[end],
                                expected_order;
                                atol = 0.3,
                            )
                        end
                    end
                end
            end
            @testset "Implicit-Explicit" begin
                for (slow_method, slow_order) in mrigark_irk_methods
                    for (mid_method, mid_order) in mrigark_erk_methods
                        for (fast_method, fast_order) in fast_mrigark_methods
                            for (n, slow_dt) in enumerate(dts)
                                Q = exactsolution(0)
                                mid_dt = slow_dt / ω2
                                fast_dt = slow_dt / ω1
                                fastsolver = fast_method(rhs1!, Q; dt = fast_dt)
                                midsolver = mid_method(
                                    rhs2!,
                                    fastsolver,
                                    Q,
                                    dt = mid_dt,
                                )
                                slowsolver = slow_method(
                                    rhs3!,
                                    ODETestConvNonLinBE3Rate(),
                                    midsolver,
                                    Q,
                                    dt = slow_dt,
                                )
                                solve!(Q, slowsolver; timeend = finaltime)
                                error[n] = norm(Q - exactsolution(finaltime))
                            end

                            rate = log2.(error[1:(end - 1)] ./ error[2:end])
                            expected_order =
                                min(slow_order, mid_order, fast_order)
                            @test isapprox(
                                rate[end],
                                expected_order;
                                atol = 0.4,
                            )
                        end
                    end
                end
            end
        end
    end
end
