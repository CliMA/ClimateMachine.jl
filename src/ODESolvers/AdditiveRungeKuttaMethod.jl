module AdditiveRungeKuttaMethod
export AdditiveRungeKutta, updatedt!

using GPUifyLoops

using StaticArrays

using ..ODESolvers
ODEs = ODESolvers
using ..SpaceMethods

"""
    AdditiveRungeKutta(f, l, Q; dt, t0 = 0)

This is a time stepping object for implicit-explicit time stepping of the
decomposed differential equation given by the chosen linear operator 'l',
the full right-hand-side function `f` and the state `Q`, i.e.

Q̇ = [l(Q, t)] + [f(Q, t) - l(Q, t)],

with the required time step size `dt` and optional initial time `t0`. The
linear operator `l` is integrated implicitly whereas the remaining part
`f - l` is integrated explicitly. This time stepping object is intended
to be passed to the `solve!` command.

This uses the second-order-accurate additive Runge--Kutta scheme of
Giraldo, Kelly and Constantinescu (2013).

### References
F. X. Giraldo, J. Kelly, and E. M. Constantinescu, Implicit-explicit formulations of a three-dimensional nonhydrostatic unified model of the atmosphere (NUMA). SIAM J. Sci. Comput., 35(5), pp. B1162–B1194
"""
struct AdditiveRungeKutta{T, RT, AT, Nstages, Nstages_sq} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{RT,1}
  "time"
  t::Array{RT,1}
  "rhs function"
  rhs!::Function
  "rhs linear operator"
  rhs_linear!::Function
  "linear solver"
  solve_linear_problem!::Function
  "Storage for solution during the AdditiveRungeKutta update"
  Qstages::NTuple{Nstages, AT}
  "Storage for RHS during the AdditiveRungeKutta update"
  Rstages::NTuple{Nstages, AT}
  "Storage for the explicit solution"
  QE::AT
  "Storage for the linear solver rhs vector"
  Qhat::AT
  "Storage for the linear solver solution variable"
  Qtt::AT
  "RK coefficient matrix A for the explicit scheme"
  RKA_explicit::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
  "RK coefficient matrix A for the implicit scheme"
  RKA_implicit::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
  "RK coefficient vector B (rhs add in scaling)"
  RKB::SArray{Tuple{Nstages}, RT, 1, Nstages}
  "RK coefficient vector C (time scaling)"
  RKC::SArray{Tuple{Nstages}, RT, 1, Nstages}

  function AdditiveRungeKutta(rhs!::Function,
                              rhs_linear!::Function,
                              solve_linear_problem!::Function,
                              Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}

    @assert dt != nothing

    T = eltype(Q)
    RT = real(T)
    dt = [RT(dt)]
    t0 = [RT(t0)]
    a32 = RT((3 + 2sqrt(2)) / 6)
    RKA_explicit = [RT(0)           RT(0)   RT(0);
                    RT(2 - sqrt(2)) RT(0)   RT(0);
                    RT(1 - a32)     RT(a32) RT(0)]

    RKA_implicit = [RT(0)               RT(0)               RT(0);
                    RT(1 - 1 / sqrt(2)) RT(1 - 1 / sqrt(2)) RT(0);
                    RT(1 / (2sqrt(2)))  RT(1 / (2sqrt(2)))  RT(1 - 1 / sqrt(2))]

    RKB = [RT(1 / (2sqrt(2))), RT(1 / (2sqrt(2))), RT(1 - 1 / sqrt(2))]
    RKC = [RT(0), RT(2 - sqrt(2)), RT(1)]
    
    nstages = length(RKB)

    Qstages = ntuple(i -> similar(Q), nstages)
    Rstages = ntuple(i -> similar(Q), nstages)
    
    QE = similar(Q)
    Qhat = similar(Q)
    Qtt = similar(Q)

    new{T, RT, AT, nstages, nstages ^ 2}(dt, t0, rhs!, rhs_linear!, solve_linear_problem!, Qstages, Rstages, QE, Qhat, Qtt, RKA_explicit, RKA_implicit, RKB, RKC)
  end
end

function AdditiveRungeKutta(spacedisc::AbstractSpaceMethod, Q; dt=nothing,
                              t0=0)
  rhs! = (x...; increment) -> SpaceMethods.odefun!(spacedisc, x..., increment = increment)
  AdditiveRungeKutta(rhs!, Q; dt=dt, t0=t0)
end

ODEs.order(::Type{<:AdditiveRungeKutta}) = 2

"""
    updatedt!(ark::AdditiveRungeKutta, dt)

Change the time step size to `dt` for `ark`.
"""
updatedt!(ark::AdditiveRungeKutta, dt) = ark.dt[1] = dt

function ODEs.dostep!(Q, ark::AdditiveRungeKutta, timeend,
                      adjustfinalstep)

  time, dt = ark.t[1], ark.dt[1]
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
    @assert dt > 0
  end

  RKA_explicit, RKA_implicit = ark.RKA_explicit, ark.RKA_implicit
  RKB, RKC = ark.RKB, ark.RKC
  rhs!, rhs_linear! = ark.rhs!, ark.rhs_linear!
  Qstages, Rstages = ark.Qstages, ark.Rstages
  QE, Qhat, Qtt = ark.QE, ark.Qhat, ark.Qtt

  rv_Q = ODEs.realview(Q)
  rv_Qstages = ODEs.realview.(Qstages)
  rv_Rstages = ODEs.realview.(Rstages)
  rv_QE = ODEs.realview(QE)
  rv_Qhat = ODEs.realview(Qhat)
  rv_Qtt = ODEs.realview(Qtt)

  nstages = length(RKB)

  threads = 1024
  blocks = div(length(rv_Q) + threads - 1, threads)

  rv_Qstages[1] .= rv_Q
  for is = 2:nstages
    rhs!(Rstages[is-1], Qstages[is-1], time + RKC[is-1] * dt, increment = false)

    #construct QE
    rv_QE .= rv_Q
    for js = 1:is-1
      rv_QE .+= dt * RKA_explicit[is, js] * rv_Rstages[js]
    end
    
    #construct Qhat
    rv_Qhat .= rv_QE
    for js = 1:is-1
      rv_Qhat .+= (RKA_implicit[is, js] - RKA_explicit[is, js]) / RKA_implicit[is, is] * rv_Qstages[js]
    end

    #solves Q_tt = Qhat + dt * RKA_implicit[is, is] * rhs_linear!(Q_tt)
    ark.solve_linear_problem!(Qtt, Qhat, rhs_linear!, dt * RKA_implicit[is, is])
    
    #update Qstages
    rv_Qstages[is] .= rv_Qtt
    for js = 1:is-1
      rv_Qstages[is] .-= (RKA_implicit[is, js] - RKA_explicit[is, js]) / RKA_implicit[is, is] * rv_Qstages[js]
    end
  end
 
  # compute the rhs for the final stage
  rhs!(Rstages[nstages], Qstages[nstages], time + RKC[nstages] * dt; increment = false)

  # compose the final solution
  for is = 1:nstages
    rv_Q .+= RKB[is] * dt * rv_Rstages[is]
  end

  if dt == ark.dt[1]
    ark.t[1] += dt
  else
    ark.t[1] = timeend
  end

end

end
