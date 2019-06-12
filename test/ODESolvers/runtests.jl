using Test
using CLIMA
using CLIMA.ODESolvers
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.StrongStabilityPreservingRungeKuttaMethod
using CLIMA.AdditiveRungeKuttaMethod

const explicit_methods = [LSRKCarpenterKennedy54,
                          StrongStabilityPreservingRungeKutta33,
                          StrongStabilityPreservingRungeKutta34,
                         ]
const imex_methods = [AdditiveRungeKutta2a, AdditiveRungeKutta5]

@testset "ODE Solvers `order` function" begin
  dummy!(dQ, Q, time; increment) = nothing
  another_dummy!(Qtt, Qhat, rhs_linear!, a) = nothing
  
  for method in explicit_methods
    solver = method(dummy!, [0.0]; dt = 1.0)
    @test order(method) == order(solver)
  end

  for method in imex_methods
    solver = method(dummy!, dummy!, another_dummy!, [0.0]; dt = 1.0)
    @test order(method) == order(solver)
  end
end

let 
  function rhs!(dQ, Q, time; increment)
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

    for method in explicit_methods
      errors = similar(dts)
      for (n, dt) in enumerate(dts)
        Q = [q0]
        solver = method(rhs!, Q; dt = dt, t0 = 0.0)
        solve!(Q, solver; timeend = finaltime)
        errors[n] = abs(Q[1] - exactsolution(q0, finaltime))
      end
      rates = log2.(errors[1:end-1] ./ errors[2:end])
      @test isapprox(rates[end], order(method); atol = 0.1)
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

      for method in explicit_methods
        errors = similar(dts)
        for (n, dt) in enumerate(dts)
          Q = CuArray(q0s)
          solver = method(rhs!, Q; dt = dt, t0 = 0.0)
          solve!(Q, solver; timeend = finaltime)
          Q = Array(Q)
          errors[n] = maximum(abs.(Q .- exactsolution.(q0s, finaltime)))
        end
        rates = log2.(errors[1:end-1] ./ errors[2:end])
        @test isapprox(rates[end], order(method); atol = 0.1)
      end
    end
  end
end

let 
  c = 100.0
  function rhs!(dQ, Q, time; increment)
    if increment
      dQ .+= im * c * Q .+ exp(im * time)
    else
      dQ .= im * c * Q .+ exp(im * time)
    end
  end
 
  function rhs_linear!(dQ, Q, time; increment)
    if increment
      dQ .+= im * c * Q
    else
      dQ .= im * c * Q
    end
  end
 
  function solve_linear_problem!(Qtt, Qhat, rhs_linear!, a)
    Qtt .= Qhat ./ (1 - a * im * c)
  end

  function exactsolution(q0, time)
    q0 * exp(im * c * time) + (exp(im * time) - exp(im * c * time)) / (im * (1 - c))
  end

  @testset "Stiff Problem" begin
    q0 = ComplexF64(1)
    finaltime = pi / 2
    dts = [2.0 ^ (-k) for k = 2:13]

    for method in imex_methods
      errors = similar(dts)
      for (n, dt) in enumerate(dts)
        Q = [q0]
        solver = method(rhs!, rhs_linear!, solve_linear_problem!, Q; dt = dt, t0 = 0.0)
        solve!(Q, solver; timeend = finaltime)
        errors[n] = abs(Q[1] - exactsolution(q0, finaltime))
      end

      rates = log2.(errors[1:end-1] ./ errors[2:end])
      #println("dts = $dts")
      #println("error = $errors")
      #println("rates = $rates")
      @test errors[1] < 2.0
      @test isapprox(rates[end], order(method); atol = 0.1)
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

      for method in imex_methods
        errors = similar(dts)
        for (n, dt) in enumerate(dts)
          Q = CuArray{ComplexF64}(q0s)
          solver = method(rhs!, rhs_linear!, solve_linear_problem!, Q; dt = dt, t0 = 0.0)
          solve!(Q, solver; timeend = finaltime)
          Q = Array(Q)
          errors[n] = maximum(abs.(Q - exactsolution.(q0s, finaltime)))
        end

        rates = log2.(errors[1:end-1] ./ errors[2:end])
        #println("dts = $dts")
        #println("error = $errors")
        #println("rates = $rates")
        @test errors[1] < 2.0
        @test isapprox(rates[end], order(method); atol = 0.1)
      end
    end
  end
end
