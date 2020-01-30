module AdditiveRungeKuttaMethod
export AdditiveRungeKutta
export ARK2GiraldoKellyConstantinescu
export ARK548L2SA2KennedyCarpenter, ARK437L2SA1KennedyCarpenter

# Naive formulation that uses equation 3.8 from Giraldo, Kelly, and Constantinescu (2013) directly.
# Seems to cut the number of solver iterations by half but requires Nstages - 1 additional storage.
struct NaiveVariant end
additional_storage(::NaiveVariant, Q, Nstages) = (Lstages = ntuple(i -> similar(Q), Nstages),)

# Formulation that does things exactly as in Giraldo, Kelly, and Constantinescu (2013).
# Uses only one additional vector of storage regardless of the number of stages.
struct LowStorageVariant end
additional_storage(::LowStorageVariant, Q, Nstages) = (Qtt = similar(Q),)

using GPUifyLoops
include("AdditiveRungeKuttaMethod_kernels.jl")
include("AdditiveRungeKuttaMethod_tableaux.jl")

using StaticArrays

using ..ODESolvers
ODEs = ODESolvers
using ..SpaceMethods
using ..LinearSolvers
using ..MPIStateArrays: device, realview



"""
    op! = EulerOperator(f!, ϵ)

Construct a linear operator which performs an explicit Euler step ``Q + α f(Q)``,
where `f!` and `op!` both operate inplace, with extra arguments passed through, i.e.
```
op!(LQ, Q, args...)
```
is equivalent to
```
f!(dQ, Q, args...)
LQ .= Q .+ ϵ .* dQ
```
"""
mutable struct EulerOperator{F,FT}
  f!::F
  ϵ::FT
end

function (op::EulerOperator)(LQ, Q, args...)
  op.f!(LQ, Q, args..., increment=false)
  @. LQ = Q + op.ϵ * LQ
end

"""
    AdditiveRungeKutta(f, l, linearsolver, RKAe, RKAi, RKB, RKC, Q;
                       split_nonlinear_linear, variant, dt, t0 = 0)

This is a time stepping object for implicit-explicit time stepping of a
decomposed differential equation. When `split_nonlinear_linear == false`
the equation is assumed to be decomposed as

```math
  \\dot{Q} = [l(Q, t)] + [f(Q, t) - l(Q, t)]
```

where `Q` is the state, `f` is the full tendency and
`l` is the chosen linear operator. When `split_nonlinear_linear == true`
the assumed decomposition is

```math
  \\dot{Q} = [l(Q, t)] + [f(Q, t)]
```

where `f` is now only the nonlinear tendency. For both decompositions the
linear operator `l` is integrated implicitly whereas the remaining part
is integrated explicitly. Other arguments are the required time step size `dt`
and the optional initial time `t0`. The resulting linear systems are solved
using the provided `linearsolver` solver. This time stepping object is intended
to be passed to the `solve!` command.

The constructor builds an additive Runge--Kutta scheme 
based on the provided `RKAe`, `RKAi`, `RKB` and `RKC` coefficient arrays.
Additionally `variant` specifies which of the analytically equivalent but numerically
different formulations of the scheme is used.

The available concrete implementations are:

  - [`ARK2GiraldoKellyConstantinescu`](@ref)
  - [`ARK548L2SA2KennedyCarpenter`](@ref)
  - [`ARK437L2SA1KennedyCarpenter`](@ref)
"""
mutable struct AdditiveRungeKutta{T, RT, AT, LT, V, VS, GS, Nstages, Nstages_sq} <: ODEs.AbstractODESolver
  "time step"
  dt::RT
  "time"
  t::RT
  "rhs function"
  rhs!
  "rhs linear operator"
  rhs_linear!
  "rhs linear operator"
  rhs_offset!
  "implicit operator, pre-factorized"
  implicitoperator!
  "linear solver"
  linearsolver::LT
  "Storage for solution during the AdditiveRungeKutta update"
  Qstages::NTuple{Nstages, AT}
  "Storage for RHS during the AdditiveRungeKutta update"
  Rstages::NTuple{Nstages, AT}
  "Storage for RHS during the AdditiveRungeKutta update"
  Gstages::GS
  "Storage for the linear solver rhs vector"
  Qhat::AT
  "RK coefficient matrix A for the explicit scheme"
  RKA_explicit::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
  "RK coefficient matrix A for the implicit scheme"
  RKA_implicit::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
  "RK coefficient vector B (rhs add in scaling)"
  RKB::SArray{Tuple{Nstages}, RT, 1, Nstages}
  "RK coefficient vector C (time scaling)"
  RKC::SArray{Tuple{Nstages}, RT, 1, Nstages}
  split_nonlinear_linear::Bool
  "Variant of the ARK scheme"
  variant::V
  "Storage dependent on the variant of the ARK scheme"
  variant_storage::VS

  function AdditiveRungeKutta(rhs!,
                              rhs_affine!,
                              linearsolver::AbstractLinearSolver,
                              RKA_explicit, RKA_implicit, RKB, RKC,
                              split_nonlinear_linear, variant,
                              Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}

    @assert dt != nothing

    T = eltype(Q)
    LT = typeof(linearsolver)
    RT = real(T)
    
    Nstages = length(RKB)
    
    if rhs_affine! isa Tuple
      rhs_linear!, rhs_offset! = rhs_affine!
      Gstages = ntuple(i -> similar(Q), variant isa LowStorageVariant ? Nstages : 0)
    else
      rhs_linear!, rhs_offset! = rhs_affine!, nothing
      Gstages = ntuple(i -> similar(Q), 0)
    end
    GS = typeof(Gstages)

    Qstages = (Q, ntuple(i -> similar(Q), Nstages - 1)...)
    Rstages = ntuple(i -> similar(Q), Nstages)
    Qhat = similar(Q)
    
    V = typeof(variant)
    variant_storage = additional_storage(variant, Q, Nstages)
    VS = typeof(variant_storage)

    # The code throughout assumes SDIRK implicit tableau so we assert that
    # here.
    for is = 2:Nstages
      @assert RKA_implicit[is, is] ≈ RKA_implicit[2, 2]
    end

    α = dt * RKA_implicit[2, 2]
    # Here we are passing NaN for the time since prefactorization assumes the
    # operator is time independent.  If that is not the case the NaN will
    # surface.
    implicitoperator! = prefactorize(EulerOperator(rhs_linear!, -α),
                                     linearsolver, Q, nothing, T(NaN))

    new{T, RT, AT, LT, V, VS, GS, Nstages, Nstages ^ 2}(
      RT(dt), RT(t0),
      rhs!, rhs_linear!, rhs_offset!, implicitoperator!, linearsolver,
      Qstages, Rstages, Gstages, Qhat,
      RKA_explicit, RKA_implicit, RKB, RKC,
      split_nonlinear_linear,
      variant, variant_storage)
  end
end

function AdditiveRungeKutta(spacedisc::AbstractSpaceMethod,
                            spacedisc_linear::AbstractSpaceMethod,
                            linearsolver::AbstractLinearSolver,
                            RKA_explicit, RKA_implicit, RKB, RKC,
                            split_nonlinear_linear, variant,
                            Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}
  rhs! = (x...; increment) -> SpaceMethods.odefun!(spacedisc, x..., increment = increment)
  rhs_linear! = (x...; increment) -> SpaceMethods.odefun!(spacedisc_linear, x..., increment = increment)
  AdditiveRungeKutta(rhs!, rhs_linear!, linearsolver,
                     RKA_explicit, RKA_implicit, RKB, RKC,
                     split_nonlinear_linear, variant,
                     Q; dt=dt, t0=t0)
end

# this will only work for iterative solves
# direct solvers use prefactorization
ODEs.isadjustable(ark::AdditiveRungeKutta) = ark.implicitoperator! isa EulerOperator
function ODEs.updatedt!(ark::AdditiveRungeKutta, dt)
  @assert ODEs.isadjustable(ark)
  ark.dt = dt
  α = dt * ark.RKA_implicit[2, 2]
  ark.implicitoperator! = EulerOperator(ark.rhs_linear!, -α)
end
ODEs.updatetime!(ark::AdditiveRungeKutta, time) = (ark.t = time)

function ODEs.dostep!(Q, ark::AdditiveRungeKutta, p, timeend::Real,
                      adjustfinalstep::Bool)
  time, dt = ark.t, ark.dt
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
  end
  @assert dt > 0

  ODEs.dostep!(Q, ark, p, time, dt)

  if dt == ark.dt
    ark.t += dt
  else
    ark.t = timeend
  end

end

function ODEs.dostep!(Q, ark::AdditiveRungeKutta, p, time::Real, dt::Real,
                      slow_δ = nothing, slow_rv_dQ = nothing,
                      slow_scaling = nothing)
  ODEs.dostep!(Q, ark, ark.variant, p, time, dt, slow_δ, slow_rv_dQ, slow_scaling)
end

function ODEs.dostep!(Q, ark::AdditiveRungeKutta, variant::NaiveVariant,
                      p, time::Real, dt::Real,
                      slow_δ = nothing, slow_rv_dQ = nothing,
                      slow_scaling = nothing)
  implicitoperator!, linearsolver = ark.implicitoperator!, ark.linearsolver
  RKA_explicit, RKA_implicit = ark.RKA_explicit, ark.RKA_implicit
  RKB, RKC = ark.RKB, ark.RKC
  rhs!, rhs_linear!, rhs_offset! = ark.rhs!, ark.rhs_linear!, ark.rhs_offset!
  Qstages, Rstages = ark.Qstages, ark.Rstages
  Qhat = ark.Qhat
  split_nonlinear_linear = ark.split_nonlinear_linear
  Lstages = ark.variant_storage.Lstages
  affine = !isnothing(rhs_offset!)

  rv_Q = realview(Q)
  rv_Qstages = realview.(Qstages)
  rv_Lstages = realview.(Lstages)
  rv_Rstages = realview.(Rstages)
  rv_Qhat = realview(Qhat)

  Nstages = length(RKB)

  threads = 256
  blocks = div(length(rv_Q) + threads - 1, threads)

  # calculate the rhs at first stage to initialize the stage loop
  rhs!(Rstages[1], Qstages[1], p, time + RKC[1] * dt, increment = false)
  
  if dt != ark.dt
    α = dt * RKA_implicit[2, 2]
    implicitoperator! = EulerOperator(rhs_linear!, -α)
  end

  rhs_linear!(Lstages[1], Qstages[1], p, time + RKC[1] * dt, increment = false)
  if affine
    rhs_offset!(Lstages[1], time + RKC[1] * dt, increment = true)
  end

  # note that it is important that this loop does not modify Q!
  for istage = 2:Nstages
    stagetime = time + RKC[istage] * dt
    
    if affine
      rhs_offset!(Lstages[istage], stagetime, increment = false)
    end

    # this kernel also initializes Qstages[istage] with an initial guess
    # for the linear solver
    @launch(device(Q), threads = threads, blocks = blocks,
            stage_update!(variant, rv_Q, rv_Qstages, rv_Lstages, rv_Rstages, rv_Qhat,
                          RKA_explicit, RKA_implicit, dt, Val(istage),
                          Val(split_nonlinear_linear), Val(affine), slow_δ, slow_rv_dQ))

    #solves Q_tt = Qhat + dt * RKA_implicit[istage, istage] * rhs_linear!(Q_tt)
    α = dt * RKA_implicit[istage, istage]
    linearoperator! = function(LQ, Q)
      rhs_linear!(LQ, Q, p, stagetime; increment = false)
      @. LQ = Q - α * LQ
    end
    linearsolve!(implicitoperator!, linearsolver, Qstages[istage], Qhat, p, stagetime)
    
    rhs!(Rstages[istage], Qstages[istage], p, stagetime, increment = false)
    rhs_linear!(Lstages[istage], Qstages[istage], p, stagetime, increment = affine)
  end

  # compose the final solution
  @launch(device(Q), threads = threads, blocks = blocks,
          solution_update!(variant, rv_Q, rv_Lstages, rv_Rstages, RKB, dt,
                           Val(Nstages), Val(split_nonlinear_linear),
                           slow_δ, slow_rv_dQ, slow_scaling))
end

function ODEs.dostep!(Q, ark::AdditiveRungeKutta, variant::LowStorageVariant,
                      p, time::Real, dt::Real,
                      slow_δ = nothing, slow_rv_dQ = nothing,
                      slow_scaling = nothing)
  implicitoperator!, linearsolver = ark.implicitoperator!, ark.linearsolver
  RKA_explicit, RKA_implicit = ark.RKA_explicit, ark.RKA_implicit
  RKB, RKC = ark.RKB, ark.RKC
  rhs!, rhs_linear!, rhs_offset! = ark.rhs!, ark.rhs_linear!, ark.rhs_offset!
  Qstages, Rstages, Gstages = ark.Qstages, ark.Rstages, ark.Gstages
  Qhat = ark.Qhat
  split_nonlinear_linear = ark.split_nonlinear_linear
  Qtt = ark.variant_storage.Qtt
  affine = !isnothing(rhs_offset!)

  rv_Q = realview(Q)
  rv_Qstages = realview.(Qstages)
  rv_Rstages = realview.(Rstages)
  rv_Gstages = realview.(Gstages)
  rv_Qhat = realview(Qhat)
  rv_Qtt = realview(Qtt)

  Nstages = length(RKB)

  threads = 256
  blocks = div(length(rv_Q) + threads - 1, threads)

  # calculate the rhs at first stage to initialize the stage loop
  rhs!(Rstages[1], Qstages[1], p, time + RKC[1] * dt, increment = false)
  if affine
    rhs_offset!(Gstages[1], time + RKC[1] * dt, increment = false)
  end

  if dt != ark.dt
    α = dt * RKA_implicit[2, 2]
    implicitoperator! = EulerOperator(rhs_linear!, -α)
  end

  # note that it is important that this loop does not modify Q!
  for istage = 2:Nstages
    stagetime = time + RKC[istage] * dt
    
    if affine
      rhs_offset!(Gstages[istage], stagetime, increment = false)
    end

    # this kernel also initializes Qtt for the linear solver
    @launch(device(Q), threads = threads, blocks = blocks,
            stage_update!(variant, rv_Q, rv_Qstages, rv_Rstages, rv_Gstages,
                          rv_Qhat, rv_Qtt,
                          RKA_explicit, RKA_implicit, dt, Val(istage),
                          Val(split_nonlinear_linear), Val(affine),
                          slow_δ, slow_rv_dQ))

    #solves Q_tt = Qhat + dt * RKA_implicit[istage, istage] * rhs_linear!(Q_tt)
    linearsolve!(implicitoperator!, linearsolver, Qtt, Qhat, p, stagetime)

    #update Qstages
    Qstages[istage] .+= Qtt

    rhs!(Rstages[istage], Qstages[istage], p, stagetime, increment = false)
  end

  if split_nonlinear_linear
    for istage = 1:Nstages
      stagetime = time + RKC[istage] * dt
      rhs_linear!(Rstages[istage], Qstages[istage], p, stagetime, increment = true)
    end
  end

  # compose the final solution
  @launch(device(Q), threads = threads, blocks = blocks,
          solution_update!(variant, rv_Q, rv_Rstages, rv_Gstages, RKB, dt,
                           Val(Nstages), Val(affine), Val(split_nonlinear_linear),
                           slow_δ, slow_rv_dQ, slow_scaling))
end

"""
    ARK2GiraldoKellyConstantinescu(f, l, linearsolver, Q; dt, t0,
                                   split_nonlinear_linear, variant, paperversion)

This function returns an [`AdditiveRungeKutta`](@ref) time stepping object,
see the documentation of [`AdditiveRungeKutta`](@ref) for arguments definitions.
This time stepping object is intended to be passed to the `solve!` command.

`paperversion=true` uses the coefficients from the paper, `paperversion=false`
uses coefficients that make the scheme (much) more stable but less accurate

This uses the second-order-accurate 3-stage additive Runge--Kutta scheme of
Giraldo, Kelly and Constantinescu (2013).

### References
    @article{giraldo2013implicit,
      title={Implicit-explicit formulations of a three-dimensional nonhydrostatic unified model of the atmosphere ({NUMA})},
      author={Giraldo, Francis X and Kelly, James F and Constantinescu, Emil M},
      journal={SIAM Journal on Scientific Computing},
      volume={35},
      number={5},
      pages={B1162--B1194},
      year={2013},
      publisher={SIAM}
    }
"""
function ARK2GiraldoKellyConstantinescu(F, L,
                                        linearsolver::AbstractLinearSolver,
                                        Q::AT; dt=nothing, t0=0,
                                        split_nonlinear_linear=false,
                                        variant=LowStorageVariant(),
                                        paperversion=false) where {AT<:AbstractArray}

  @assert dt != nothing

  T = eltype(Q)
  RT = real(T)
  
  RKA_explicit, RKA_implicit, RKB, RKC = ARK2GiraldoKellyConstantinescu_tableau(RT, paperversion)

  AdditiveRungeKutta(F, L, linearsolver,
                     RKA_explicit, RKA_implicit, RKB, RKC,
                     split_nonlinear_linear,
                     variant,
                     Q; dt=dt, t0=t0)
end

"""
    ARK548L2SA2KennedyCarpenter(f, l, linearsolver, Q; dt, t0,
                                split_nonlinear_linear, variant)

This function returns an [`AdditiveRungeKutta`](@ref) time stepping object,
see the documentation of [`AdditiveRungeKutta`](@ref) for arguments definitions.
This time stepping object is intended to be passed to the `solve!` command.

This uses the fifth-order-accurate 8-stage additive Runge--Kutta scheme of
Kennedy and Carpenter (2013).

### References

    @article{kennedy2019higher,
      title={Higher-order additive Runge--Kutta schemes for ordinary differential equations},
      author={Kennedy, Christopher A and Carpenter, Mark H},
      journal={Applied Numerical Mathematics},
      volume={136},
      pages={183--205},
      year={2019},
      publisher={Elsevier}
    }
"""
function ARK548L2SA2KennedyCarpenter(F, L,
                                     linearsolver::AbstractLinearSolver,
                                     Q::AT; dt=nothing, t0=0,
                                     split_nonlinear_linear=false,
                                     variant=LowStorageVariant()) where {AT<:AbstractArray}

  @assert dt != nothing

  T = eltype(Q)
  RT = real(T)

  RKA_explicit, RKA_implicit, RKB, RKC = ARK548L2SA2KennedyCarpenter_tableau(RT)

  ark = AdditiveRungeKutta(F, L, linearsolver,
                           RKA_explicit, RKA_implicit, RKB, RKC,
                           split_nonlinear_linear,
                           variant,
                           Q; dt=dt, t0=t0)
end

"""
    ARK437L2SA1KennedyCarpenter(f, l, linearsolver, Q; dt, t0,
                                split_nonlinear_linear, variant)

This function returns an [`AdditiveRungeKutta`](@ref) time stepping object,
see the documentation of [`AdditiveRungeKutta`](@ref) for arguments definitions.
This time stepping object is intended to be passed to the `solve!` command.

This uses the fourth-order-accurate 7-stage additive Runge--Kutta scheme of
Kennedy and Carpenter (2013).

### References
    @article{kennedy2019higher,
      title={Higher-order additive Runge--Kutta schemes for ordinary differential equations},
      author={Kennedy, Christopher A and Carpenter, Mark H},
      journal={Applied Numerical Mathematics},
      volume={136},
      pages={183--205},
      year={2019},
      publisher={Elsevier}
    }
"""
function ARK437L2SA1KennedyCarpenter(F, L,
                                     linearsolver::AbstractLinearSolver,
                                     Q::AT; dt=nothing, t0=0,
                                     split_nonlinear_linear=false,
                                     variant=LowStorageVariant()) where {AT<:AbstractArray}

  @assert dt != nothing
  T = eltype(Q)
  RT = real(T)
  
  RKA_explicit, RKA_implicit, RKB, RKC = ARK437L2SA1KennedyCarpenter_tableau(RT)

  ark = AdditiveRungeKutta(F, L, linearsolver,
                           RKA_explicit, RKA_implicit, RKB, RKC,
                           split_nonlinear_linear,
                           variant,
                           Q; dt=dt, t0=t0)
end

end
