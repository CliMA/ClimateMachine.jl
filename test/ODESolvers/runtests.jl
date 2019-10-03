using Test
using CLIMA
using CLIMA.ODESolvers
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.StrongStabilityPreservingRungeKuttaMethod
using CLIMA.AdditiveRungeKuttaMethod
using CLIMA.MultirateRungeKuttaMethod
using CLIMA.LinearSolvers

const slow_mrrk_methods = [(LSRK54CarpenterKennedy, 4)
                           (LSRK144NiegemannDiehlBusch, 4)
                          ]
const fast_mrrk_methods = [(LSRK54CarpenterKennedy, 4)
                           (LSRK144NiegemannDiehlBusch, 4)
                           (SSPRK33ShuOsher, 3)
                           (SSPRK34SpiteriRuuth, 3)
                          ]
const explicit_methods = [(LSRK54CarpenterKennedy, 4)
                          (LSRK144NiegemannDiehlBusch, 4)
                          (SSPRK33ShuOsher, 3)
                          (SSPRK34SpiteriRuuth, 3)
                         ]

const imex_methods = [(ARK2GiraldoKellyConstantinescu, 2),
                      (ARK548L2SA2KennedyCarpenter, 5)
                     ]

let 
  function rhs!(dQ, Q, ::Nothing, time; increment)
    if increment
      dQ .+= Q * cos(time)
    else
      dQ .= Q * cos(time)
    end
  end
  exactsolution(q0, time) = q0 * exp(sin(time))

  @testset "ODE Solvers Convergence" begin
    q0 = 1.0
    finaltime = 20.0
    dts = [2.0 ^ (-k) for k = 0:7]

    for (method, expected_order) in explicit_methods
      errors = similar(dts)
      for (n, dt) in enumerate(dts)
        Q = [q0]
        solver = method(rhs!, Q; dt = dt, t0 = 0.0)
        solve!(Q, solver; timeend = finaltime)
        errors[n] = abs(Q[1] - exactsolution(q0, finaltime))
      end
      rates = log2.(errors[1:end-1] ./ errors[2:end])
      @test isapprox(rates[end], expected_order; atol = 0.15)
    end
  end

  @testset "ODE Solvers Composition of solve!" begin
    q0 = 1.0
    halftime = 10.0
    finaltime = 20.0
    dt = 0.75

    for (method, _) in explicit_methods
      Q1 = [q0]
      solver1 = method(rhs!, Q1; dt = dt, t0 = 0.0)
      solve!(Q1, solver1; timeend = finaltime)
      
      Q2 = [q0]
      solver2 = method(rhs!, Q2; dt = dt, t0 = 0.0)
      solve!(Q2, solver2; timeend = halftime, adjustfinalstep = false)
      solve!(Q2, solver2; timeend = finaltime)

      @test Q2 == Q1
    end
  end

  @static if haspkg("CuArrays")
    using CuArrays
    CuArrays.allowscalar(false)
    
    @testset "CUDA ODE Solvers Convergence" begin
      ninitial = 1337
      q0s = range(1.0, 2.0, length = ninitial)
      finaltime = 20.0
      dts = [2.0 ^ (-k) for k = 0:7]

      for (method, expected_order) in explicit_methods
        errors = similar(dts)
        for (n, dt) in enumerate(dts)
          Q = CuArray(q0s)
          solver = method(rhs!, Q; dt = dt, t0 = 0.0)
          solve!(Q, solver; timeend = finaltime)
          Q = Array(Q)
          errors[n] = maximum(abs.(Q .- exactsolution.(q0s, finaltime)))
        end
        rates = log2.(errors[1:end-1] ./ errors[2:end])
        @test isapprox(rates[end], expected_order; atol = 0.25)
      end
    end

    @testset "CUDA ODE Solvers Composition of solve!" begin
      ninitial = 1337
      q0s = range(1.0, 2.0, length = ninitial)
      halftime = 10.0
      finaltime = 20.0
      dt = 0.75

      for (method, _) in explicit_methods
        Q1 = CuArray(q0s)
        solver1 = method(rhs!, Q1; dt = dt, t0 = 0.0)
        solve!(Q1, solver1; timeend = finaltime)
        
        Q2 = CuArray(q0s)
        solver2 = method(rhs!, Q2; dt = dt, t0 = 0.0)
        solve!(Q2, solver2; timeend = halftime, adjustfinalstep = false)
        solve!(Q2, solver2; timeend = finaltime)

        Q1 = Array(Q1)
        Q2 = Array(Q2)
        @test Q2 == Q1
      end
    end
  end
end

let 
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
 
  function rhs_linear!(dQ, Q, ::Nothing, time; increment)
    if increment
      dQ .+= im * c * Q
    else
      dQ .= im * c * Q
    end
  end

  struct DivideLinearSolver <: AbstractLinearSolver end
  function LinearSolvers.linearsolve!(linearoperator!, Qtt, Qhat, ::DivideLinearSolver)
    @. Qhat = 1 / Qhat
    linearoperator!(Qtt, Qhat)
    @. Qtt = 1 / Qtt
  end

  function exactsolution(q0, time)
    q0 * exp(im * c * time) + (exp(im * time) - exp(im * c * time)) / (im * (1 - c))
  end

  @testset "Stiff Problem" begin
    q0 = ComplexF64(1)
    finaltime = pi / 2
    dts = [2.0 ^ (-k) for k = 2:13]

    for (method, expected_order) in imex_methods
      for split_nonlinear_linear in (false, true)
        errors = similar(dts)
        for (n, dt) in enumerate(dts)
          Q = [q0]
          rhs! = split_nonlinear_linear ? rhs_nonlinear! : rhs_full!
          solver = method(rhs!, rhs_linear!, DivideLinearSolver(),
                          Q; dt = dt, t0 = 0.0,
                          split_nonlinear_linear = split_nonlinear_linear)
          solve!(Q, solver; timeend = finaltime)
          errors[n] = abs(Q[1] - exactsolution(q0, finaltime))
        end

        rates = log2.(errors[1:end-1] ./ errors[2:end])
        @test errors[1] < 2.0
        @test isapprox(rates[end], expected_order; atol = 0.1)
      end
    end
  end

  @static if haspkg("CuArrays")
    using CuArrays
    CuArrays.allowscalar(false)

    @testset "Stiff Problem CUDA" begin
      ninitial = 1337
      q0s = range(-1, 1, length = ninitial)
      finaltime = pi / 2
      dts = [2.0 ^ (-k) for k = 2:13]

      for (method, expected_order) in imex_methods
        for split_nonlinear_linear in (false, true)
          errors = similar(dts)
          for (n, dt) in enumerate(dts)
            Q = CuArray{ComplexF64}(q0s)
            rhs! = split_nonlinear_linear ? rhs_nonlinear! : rhs_full!
            solver = method(rhs!, rhs_linear!, DivideLinearSolver(),
                            Q; dt = dt, t0 = 0.0,
                            split_nonlinear_linear = split_nonlinear_linear)
            solve!(Q, solver; timeend = finaltime)
            Q = Array(Q)
            errors[n] = maximum(abs.(Q - exactsolution.(q0s, finaltime)))
          end

          rates = log2.(errors[1:end-1] ./ errors[2:end])
          @test errors[1] < 2.0
          @test isapprox(rates[end], expected_order; atol = 0.1)
        end
      end
    end
  end
end

let 
  c = 100.0
  function rhs_fast!(dQ, Q, param, time; increment)
    if increment
      dQ .+= im * c * Q
    else
      dQ .= im * c * Q
    end
  end
  
  function rhs_slow!(dQ, Q, param, time; increment)
    if increment
      dQ .+= exp(im * time)
    else
      dQ .= exp(im * time)
    end
  end

  function exactsolution(q0, time)
    q0 * exp(im * c * time) + (exp(im * time) - exp(im * c * time)) / (im * (1 - c))
  end

  @testset "Multirate Problem (no substeps)" begin
    for (slow_method, slow_expected_order) in slow_mrrk_methods
      for (fast_method, fast_expected_order) in fast_mrrk_methods
        q0 = ComplexF64(1)
        finaltime = pi / 2
        dts = [2.0 ^ (-k) for k = 2:11]

        errors = similar(dts)
        for (n, dt) in enumerate(dts)
          Q = [q0]
          solver = MultirateRungeKutta(slow_method(rhs_slow!, Q),
                                       fast_method(rhs_fast!, Q);
                                       dt = dt, t0 = 0.0)
          param = (nothing, nothing)
          solve!(Q, solver, param; timeend = finaltime)
          errors[n] = abs(Q[1] - exactsolution(q0, finaltime))
        end

        rates = log2.(errors[1:end-1] ./ errors[2:end])
        min_order = min(slow_expected_order, fast_expected_order)
        max_order = max(slow_expected_order, fast_expected_order)
        @test (isapprox(rates[end], min_order; atol = 0.1) ||
               isapprox(rates[end], max_order; atol = 0.1) ||
               min_order <= rates[end] <= max_order)
      end
    end
  end

  @testset "Multirate Problem (with substeps)" begin
    for (slow_method, slow_expected_order) in slow_mrrk_methods
      for (fast_method, fast_expected_order) in fast_mrrk_methods
        q0 = ComplexF64(1)
        finaltime = pi / 2
        dts = [2.0 ^ (-k) for k = 2:15]

        errors = similar(dts)
        for (n, fast_dt) in enumerate(dts)
          slow_dt = c * fast_dt
          Q = [q0]
          solver = MultirateRungeKutta(slow_method(rhs_slow!, Q; dt=slow_dt),
                                       fast_method(rhs_fast!, Q; dt=fast_dt))
          param = (nothing, nothing)
          solve!(Q, solver, param; timeend = finaltime)
          errors[n] = abs(Q[1] - exactsolution(q0, finaltime))
        end

        rates = log2.(errors[1:end-1] ./ errors[2:end])
        min_order = min(slow_expected_order, fast_expected_order)
        max_order = max(slow_expected_order, fast_expected_order)
        @test (isapprox(rates[end], min_order; atol = 0.1) ||
               isapprox(rates[end], max_order; atol = 0.1) ||
               min_order <= rates[end] <= max_order)
      end
    end
  end

  @static if haspkg("CuArrays")
    using CuArrays
    CuArrays.allowscalar(false)

    @testset "Multirate Problem CUDA" begin
      ninitial = 1337
      q0s = range(-1, 1, length = ninitial)
      finaltime = pi / 2
      dts = [2.0 ^ (-k) for k = 2:11]

      for (slow_method, slow_expected_order) in slow_mrrk_methods
        for (fast_method, fast_expected_order) in fast_mrrk_methods
          errors = similar(dts)
          for (n, dt) in enumerate(dts)
            Q = CuArray{ComplexF64}(q0s)
            solver = MultirateRungeKutta(slow_method(rhs_slow!, Q),
                                         fast_method(rhs_fast!, Q);
                                         dt = dt, t0 = 0.0)
            param = (nothing, nothing)
            solve!(Q, solver, param; timeend = finaltime)
            Q = Array(Q)
            errors[n] = maximum(abs.(Q - exactsolution.(q0s, finaltime)))
          end

          rates = log2.(errors[1:end-1] ./ errors[2:end])
          min_order = min(slow_expected_order, fast_expected_order)
          max_order = max(slow_expected_order, fast_expected_order)
          @test (isapprox(rates[end], min_order; atol = 0.1) ||
                 isapprox(rates[end], max_order; atol = 0.1) ||
                 min_order <= rates[end] <= max_order)
        end
      end
    end

    @testset "Multirate Problem CUDA (with substeps)" begin
      ninitial = 1337
      q0s = range(-1, 1, length = ninitial)
      finaltime = pi / 2
      dts = [2.0 ^ (-k) for k = 2:15]

      for (slow_method, slow_expected_order) in slow_mrrk_methods
        for (fast_method, fast_expected_order) in fast_mrrk_methods
          errors = similar(dts)
          for (n, fast_dt) in enumerate(dts)
            slow_dt = c * fast_dt
            Q = CuArray{ComplexF64}(q0s)
            solver = MultirateRungeKutta(slow_method(rhs_slow!, Q; dt=slow_dt),
                                         fast_method(rhs_fast!, Q; dt=fast_dt))
            param = (nothing, nothing)
            solve!(Q, solver, param; timeend = finaltime)
            Q = Array(Q)
            errors[n] = maximum(abs.(Q - exactsolution.(q0s, finaltime)))
          end

          rates = log2.(errors[1:end-1] ./ errors[2:end])
          min_order = min(slow_expected_order, fast_expected_order)
          max_order = max(slow_expected_order, fast_expected_order)
          @test (isapprox(rates[end], min_order; atol = 0.1) ||
                 isapprox(rates[end], max_order; atol = 0.1) ||
                 min_order <= rates[end] <= max_order)
        end
      end
    end
  end
end
