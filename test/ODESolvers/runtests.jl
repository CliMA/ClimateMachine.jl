using Test
using CLIMA
using CLIMA.ODESolvers
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.StrongStabilityPreservingRungeKuttaMethod

let 
  function rhs!(dQ, Q, time)
    dQ .+= Q * cos(time)
  end
  exactsolution(q0, time) = q0 * exp(sin(time))

  method_order = [(LowStorageRungeKutta, 4),
                  (StrongStabilityPreservingRungeKutta33, 3),
                  (StrongStabilityPreservingRungeKutta34, 3)
                 ]

  @testset "ODE Solvers Convergence" begin
    q0 = 1.0
    finaltime = 20.0
    dts = [2.0 ^ (-k) for k = 0:7]

    for (method, expected_order) in method_order
      errors = similar(dts)
      for (n, dt) in enumerate(dts)
        Q = [q0]
        solver = method(rhs!, Q; dt = dt, t0 = 0.0)
        solve!(Q, solver; timeend = finaltime)
        errors[n] = abs(Q[1] - exactsolution(q0, finaltime))
      end
      rates = log2.(errors[1:end-1] ./ errors[2:end])
      @test isapprox(rates[end], expected_order; atol = 0.1)
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

      for (method, expected_order) in method_order
        errors = similar(dts)
        for (n, dt) in enumerate(dts)
          Q = CuArray(q0s)
          solver = method(rhs!, Q; dt = dt, t0 = 0.0)
          solve!(Q, solver; timeend = finaltime)
          Q = Array(Q)
          errors[n] = maximum(abs.(Q .- exactsolution.(q0s, finaltime)))
        end
        rates = log2.(errors[1:end-1] ./ errors[2:end])
        @test isapprox(rates[end], expected_order; atol = 0.1)
      end
    end
  end
end
