using Test
using CLIMA.ODESolvers
using CLIMA.AdditiveRungeKuttaMethod
using CuArrays
CuArrays.allowscalar(false)

function rhs!(dQ, Q, time; increment)
  if increment
    dQ .+= Q * cos(time)
  else
    dQ .= Q * cos(time)
  end
end

function rhs_linear!(dQ, Q, time; increment)
   if increment
     dQ .+= -Q
   else
     dQ .= -Q
   end
end

function solve_linear_problem!(Qtt, Qhat, rhs_linear!, a)
  Qtt .= Qhat ./ (1 + a)
end

exactsolution(q0, time) = q0 * exp(sin(time))

function benchmark()
      ninitial = 2 ^ 20
      q0s = range(-1, 1, length = ninitial)
      finaltime = 10
      dt = 2.0 ^ (-5)
      Q = CuArray{Float32}(q0s)
      solver = AdditiveRungeKutta(rhs!, rhs_linear!, solve_linear_problem!, Q; dt = dt, t0 = 0.0)
      solve!(Q, solver; timeend = finaltime)
      
      solver = AdditiveRungeKutta(rhs!, rhs_linear!, solve_linear_problem!, Q; dt = dt, t0 = finaltime)
      finaltime = 20
      @time solve!(Q, solver; timeend = finaltime)
      
      Q = Array(Q)
      error = maximum(abs.(Q - exactsolution.(q0s, finaltime)))
      @show error
end

benchmark()
