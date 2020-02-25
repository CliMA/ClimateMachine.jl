
export AbstractAdditiveRungeKutta
export LowStorageVariant, NaiveVariant
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

abstract type AbstractAdditiveRungeKutta <: AbstractODESolver end

include("AdditiveRungeKuttaMethod_kernels.jl")

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
mutable struct AdditiveRungeKutta{T, RT, AT, LT, V, VS, Nstages, Nstages_sq} <: AbstractAdditiveRungeKutta
  "time step"
  dt::RT
  "time"
  t::RT
  "rhs function"
  rhs!
  "rhs linear operator"
  rhs_linear!
  "implicit operator, pre-factorized"
  implicitoperator!
  "linear solver"
  linearsolver::LT
  "Storage for solution during the AdditiveRungeKutta update"
  Qstages::NTuple{Nstages, AT}
  "Storage for RHS during the AdditiveRungeKutta update"
  Rstages::NTuple{Nstages, AT}
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
                              rhs_linear!,
                              linearsolver::AbstractLinearSolver,
                              RKA_explicit, RKA_implicit, RKB, RKC,
                              split_nonlinear_linear, variant,
                              Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}

    @assert dt != nothing

    T = eltype(Q)
    LT = typeof(linearsolver)
    RT = real(T)

    Nstages = length(RKB)

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

    new{T, RT, AT, LT, V, VS, Nstages, Nstages ^ 2}(RT(dt), RT(t0),
                                                    rhs!, rhs_linear!, implicitoperator!, linearsolver,
                                                    Qstages, Rstages, Qhat,
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
isadjustable(ark::AdditiveRungeKutta) = ark.implicitoperator! isa EulerOperator
function updatedt!(ark::AdditiveRungeKutta, dt)
  @assert isadjustable(ark)
  ark.dt = dt
  α = dt * ark.RKA_implicit[2, 2]
  ark.implicitoperator! = EulerOperator(ark.rhs_linear!, -α)
end
updatetime!(ark::AdditiveRungeKutta, time) = (ark.t = time)

function dostep!(Q, ark::AdditiveRungeKutta, p, timeend::Real,
                      adjustfinalstep::Bool)
  time, dt = ark.t, ark.dt
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
  end
  @assert dt > 0

  dostep!(Q, ark, p, time, dt)

  if dt == ark.dt
    ark.t += dt
  else
    ark.t = timeend
  end

end

function dostep!(Q, ark::AdditiveRungeKutta, p, time::Real, dt::Real,
                      slow_δ = nothing, slow_rv_dQ = nothing,
                      slow_scaling = nothing)
  dostep!(Q, ark, ark.variant, p, time, dt, slow_δ, slow_rv_dQ, slow_scaling)
end

function dostep!(Q, ark::AdditiveRungeKutta, variant::NaiveVariant,
                      p, time::Real, dt::Real,
                      slow_δ = nothing, slow_rv_dQ = nothing,
                      slow_scaling = nothing)
  implicitoperator!, linearsolver = ark.implicitoperator!, ark.linearsolver
  RKA_explicit, RKA_implicit = ark.RKA_explicit, ark.RKA_implicit
  RKB, RKC = ark.RKB, ark.RKC
  rhs!, rhs_linear! = ark.rhs!, ark.rhs_linear!
  Qstages, Rstages = ark.Qstages, ark.Rstages
  Qhat = ark.Qhat
  split_nonlinear_linear = ark.split_nonlinear_linear
  Lstages = ark.variant_storage.Lstages

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

  # note that it is important that this loop does not modify Q!
  for istage = 2:Nstages
    stagetime = time + RKC[istage] * dt

    # this kernel also initializes Qstages[istage] with an initial guess
    # for the linear solver
    @launch(device(Q), threads = threads, blocks = blocks,
            stage_update!(variant, rv_Q, rv_Qstages, rv_Lstages, rv_Rstages, rv_Qhat,
                          RKA_explicit, RKA_implicit, dt, Val(istage),
                          Val(split_nonlinear_linear), slow_δ, slow_rv_dQ))

    #solves Q_tt = Qhat + dt * RKA_implicit[istage, istage] * rhs_linear!(Q_tt)
    α = dt * RKA_implicit[istage, istage]
    linearoperator! = function(LQ, Q)
      rhs_linear!(LQ, Q, p, stagetime; increment = false)
      @. LQ = Q - α * LQ
    end
    linearsolve!(implicitoperator!, linearsolver, Qstages[istage], Qhat, p, stagetime)

    rhs!(Rstages[istage], Qstages[istage], p, stagetime, increment = false)
    rhs_linear!(Lstages[istage], Qstages[istage], p, stagetime, increment = false)
  end

  # compose the final solution
  @launch(device(Q), threads = threads, blocks = blocks,
          solution_update!(variant, rv_Q, rv_Lstages, rv_Rstages, RKB, dt,
                           Val(Nstages), Val(split_nonlinear_linear),
                           slow_δ, slow_rv_dQ, slow_scaling))
end

function dostep!(Q, ark::AdditiveRungeKutta, variant::LowStorageVariant,
                      p, time::Real, dt::Real,
                      slow_δ = nothing, slow_rv_dQ = nothing,
                      slow_scaling = nothing)
  implicitoperator!, linearsolver = ark.implicitoperator!, ark.linearsolver
  RKA_explicit, RKA_implicit = ark.RKA_explicit, ark.RKA_implicit
  RKB, RKC = ark.RKB, ark.RKC
  rhs!, rhs_linear! = ark.rhs!, ark.rhs_linear!
  Qstages, Rstages = ark.Qstages, ark.Rstages
  Qhat = ark.Qhat
  split_nonlinear_linear = ark.split_nonlinear_linear
  Qtt = ark.variant_storage.Qtt

  rv_Q = realview(Q)
  rv_Qstages = realview.(Qstages)
  rv_Rstages = realview.(Rstages)
  rv_Qhat = realview(Qhat)
  rv_Qtt = realview(Qtt)

  Nstages = length(RKB)

  threads = 256
  blocks = div(length(rv_Q) + threads - 1, threads)

  # calculate the rhs at first stage to initialize the stage loop
  rhs!(Rstages[1], Qstages[1], p, time + RKC[1] * dt, increment = false)

  if dt != ark.dt
    α = dt * RKA_implicit[2, 2]
    implicitoperator! = EulerOperator(rhs_linear!, -α)
  end

  # note that it is important that this loop does not modify Q!
  for istage = 2:Nstages
    stagetime = time + RKC[istage] * dt

    # this kernel also initializes Qtt for the linear solver
    @launch(device(Q), threads = threads, blocks = blocks,
            stage_update!(variant, rv_Q, rv_Qstages, rv_Rstages, rv_Qhat, rv_Qtt,
                          RKA_explicit, RKA_implicit, dt, Val(istage),
                          Val(split_nonlinear_linear), slow_δ, slow_rv_dQ))

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
          solution_update!(variant, rv_Q, rv_Rstages, RKB, dt, Val(Nstages),
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

  a32 = RT(paperversion ? (3 + 2sqrt(2)) / 6 : 1 // 2)
  RKA_explicit = [RT(0)           RT(0)   RT(0);
                  RT(2 - sqrt(2)) RT(0)   RT(0);
                  RT(1 - a32)     RT(a32) RT(0)]

  RKA_implicit = [RT(0)               RT(0)               RT(0);
                  RT(1 - 1 / sqrt(2)) RT(1 - 1 / sqrt(2)) RT(0);
                  RT(1 / (2sqrt(2)))  RT(1 / (2sqrt(2)))  RT(1 - 1 / sqrt(2))]

  RKB = [RT(1 / (2sqrt(2))), RT(1 / (2sqrt(2))), RT(1 - 1 / sqrt(2))]
  RKC = [RT(0), RT(2 - sqrt(2)), RT(1)]

  Nstages = length(RKB)

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

  Nstages = 8
  gamma = RT(2 // 9)

  # declared as Arrays for mutability, later these will be converted to static arrays
  RKA_explicit = zeros(RT, Nstages, Nstages)
  RKA_implicit = zeros(RT, Nstages, Nstages)
  RKB = zeros(RT, Nstages)
  RKC = zeros(RT, Nstages)

  # the main diagonal
  for is = 2:Nstages
    RKA_implicit[is, is] = gamma
  end

  RKA_implicit[3, 2] = RT(2366667076620 //  8822750406821)
  RKA_implicit[4, 2] = RT(-257962897183 // 4451812247028)
  RKA_implicit[4, 3] = RT(128530224461 // 14379561246022)
  RKA_implicit[5, 2] = RT(-486229321650 // 11227943450093)
  RKA_implicit[5, 3] = RT(-225633144460 // 6633558740617)
  RKA_implicit[5, 4] = RT(1741320951451 // 6824444397158)
  RKA_implicit[6, 2] = RT(621307788657 // 4714163060173)
  RKA_implicit[6, 3] = RT(-125196015625 // 3866852212004)
  RKA_implicit[6, 4] = RT(940440206406 // 7593089888465)
  RKA_implicit[6, 5] = RT(961109811699 // 6734810228204)
  RKA_implicit[7, 2] = RT(2036305566805 // 6583108094622)
  RKA_implicit[7, 3] = RT(-3039402635899 // 4450598839912)
  RKA_implicit[7, 4] = RT(-1829510709469 // 31102090912115)
  RKA_implicit[7, 5] = RT(-286320471013 // 6931253422520)
  RKA_implicit[7, 6] = RT(8651533662697 // 9642993110008)

  RKA_explicit[3, 1] = RT(1 // 9)
  RKA_explicit[3, 2] = RT(1183333538310 // 1827251437969)
  RKA_explicit[4, 1] = RT(895379019517 // 9750411845327)
  RKA_explicit[4, 2] = RT(477606656805 // 13473228687314)
  RKA_explicit[4, 3] = RT(-112564739183 // 9373365219272)
  RKA_explicit[5, 1] = RT(-4458043123994 // 13015289567637)
  RKA_explicit[5, 2] = RT(-2500665203865 // 9342069639922)
  RKA_explicit[5, 3] = RT(983347055801 // 8893519644487)
  RKA_explicit[5, 4] = RT(2185051477207 // 2551468980502)
  RKA_explicit[6, 1] = RT(-167316361917 // 17121522574472)
  RKA_explicit[6, 2] = RT(1605541814917 // 7619724128744)
  RKA_explicit[6, 3] = RT(991021770328 // 13052792161721)
  RKA_explicit[6, 4] = RT(2342280609577 // 11279663441611)
  RKA_explicit[6, 5] = RT(3012424348531 // 12792462456678)
  RKA_explicit[7, 1] = RT(6680998715867 // 14310383562358)
  RKA_explicit[7, 2] = RT(5029118570809 // 3897454228471)
  RKA_explicit[7, 3] = RT(2415062538259 // 6382199904604)
  RKA_explicit[7, 4] = RT(-3924368632305 // 6964820224454)
  RKA_explicit[7, 5] = RT(-4331110370267 // 15021686902756)
  RKA_explicit[7, 6] = RT(-3944303808049 // 11994238218192)
  RKA_explicit[8, 1] = RT(2193717860234 // 3570523412979)
  RKA_explicit[8, 2] = RKA_explicit[8, 1]
  RKA_explicit[8, 3] = RT(5952760925747 // 18750164281544)
  RKA_explicit[8, 4] = RT(-4412967128996 // 6196664114337)
  RKA_explicit[8, 5] = RT(4151782504231 // 36106512998704)
  RKA_explicit[8, 6] = RT(572599549169 // 6265429158920)
  RKA_explicit[8, 7] = RT(-457874356192 // 11306498036315)

  RKB[2] = 0
  RKB[3] = RT(3517720773327 // 20256071687669)
  RKB[4] = RT(4569610470461 // 17934693873752)
  RKB[5] = RT(2819471173109 // 11655438449929)
  RKB[6] = RT(3296210113763 // 10722700128969)
  RKB[7] = RT(-1142099968913 // 5710983926999)

  RKC[2] = RT(4 // 9)
  RKC[3] = RT(6456083330201 // 8509243623797)
  RKC[4] = RT(1632083962415 // 14158861528103)
  RKC[5] = RT(6365430648612 // 17842476412687)
  RKC[6] = RT(18 // 25)
  RKC[7] = RT(191 // 200)

  for is = 2:Nstages
    RKA_implicit[is, 1] = RKA_implicit[is, 2]
  end

  for is = 1:Nstages-1
    RKA_implicit[Nstages, is] = RKB[is]
  end

  RKB[1] = RKB[2]
  RKB[8] = gamma

  RKA_explicit[2, 1] = RKC[2]
  RKA_explicit[Nstages, 1] = RKA_explicit[Nstages, 2]

  RKC[1] = 0
  RKC[Nstages] = 1

  # conversion to static arrays
  RKA_explicit = SMatrix{Nstages, Nstages}(RKA_explicit)
  RKA_implicit = SMatrix{Nstages, Nstages}(RKA_implicit)
  RKB = SVector{Nstages}(RKB)
  RKC = SVector{Nstages}(RKC)

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

  Nstages = 7
  gamma = RT(1235 // 10000)

  # declared as Arrays for mutability, later these will be converted to static arrays
  RKA_explicit = zeros(RT, Nstages, Nstages)
  RKA_implicit = zeros(RT, Nstages, Nstages)
  RKB = zeros(RT, Nstages)
  RKC = zeros(RT, Nstages)

  # the main diagonal
  for is = 2:Nstages
    RKA_implicit[is, is] = gamma
  end

  RKA_implicit[3, 2] = RT(624185399699 // 4186980696204)
  RKA_implicit[4, 2] = RT(1258591069120 // 10082082980243)
  RKA_implicit[4, 3] = RT(-322722984531 // 8455138723562)
  RKA_implicit[5, 2] = RT(-436103496990 // 5971407786587)
  RKA_implicit[5, 3] = RT(-2689175662187 // 11046760208243)
  RKA_implicit[5, 4] = RT(4431412449334 // 12995360898505)
  RKA_implicit[6, 2] = RT(-2207373168298 // 14430576638973)
  RKA_implicit[6, 3] = RT(242511121179 // 3358618340039)
  RKA_implicit[6, 4] = RT(3145666661981 // 7780404714551)
  RKA_implicit[6, 5] = RT(5882073923981 // 14490790706663)
  RKA_implicit[7, 2] = 0
  RKA_implicit[7, 3] = RT(9164257142617 // 17756377923965)
  RKA_implicit[7, 4] = RT(-10812980402763 // 74029279521829)
  RKA_implicit[7, 5] = RT(1335994250573 // 5691609445217)
  RKA_implicit[7, 6] = RT(2273837961795 // 8368240463276)

  RKA_explicit[3, 1] = RT(247 // 4000)
  RKA_explicit[3, 2] = RT(2694949928731 // 7487940209513)
  RKA_explicit[4, 1] = RT(464650059369 // 8764239774964)
  RKA_explicit[4, 2] = RT(878889893998 // 2444806327765)
  RKA_explicit[4, 3] = RT(-952945855348 // 12294611323341)
  RKA_explicit[5, 1] = RT(476636172619 // 8159180917465)
  RKA_explicit[5, 2] = RT(-1271469283451 // 7793814740893)
  RKA_explicit[5, 3] = RT(-859560642026 // 4356155882851)
  RKA_explicit[5, 4] = RT(1723805262919 // 4571918432560)
  RKA_explicit[6, 1] = RT(6338158500785 // 11769362343261)
  RKA_explicit[6, 2] = RT(-4970555480458 // 10924838743837)
  RKA_explicit[6, 3] = RT(3326578051521 // 2647936831840)
  RKA_explicit[6, 4] = RT(-880713585975 // 1841400956686)
  RKA_explicit[6, 5] = RT(-1428733748635 // 8843423958496)
  RKA_explicit[7, 2] = RT(760814592956 // 3276306540349)
  RKA_explicit[7, 3] = RT(-47223648122716 // 6934462133451)
  RKA_explicit[7, 4] = RT(71187472546993 // 9669769126921)
  RKA_explicit[7, 5] = RT(-13330509492149 // 9695768672337)
  RKA_explicit[7, 6] = RT(11565764226357 // 8513123442827)

  RKB[2] = 0
  RKB[3] = RT(9164257142617 // 17756377923965)
  RKB[4] = RT(-10812980402763 // 74029279521829)
  RKB[5] = RT(1335994250573 // 5691609445217)
  RKB[6] = RT(2273837961795 // 8368240463276)
  RKB[7] = RT(247 // 2000)

  RKC[2] = RT(247 // 2000)
  RKC[3] = RT(4276536705230 // 10142255878289)
  RKC[4] = RT(67 // 200)
  RKC[5] = RT(3 // 40)
  RKC[6] = RT(7 // 10)

  for is = 2:Nstages
    RKA_implicit[is, 1] = RKA_implicit[is, 2]
  end

  for is = 1:Nstages-1
    RKA_implicit[Nstages, is] = RKB[is]
  end

  RKB[1] = RKB[2]

  RKA_explicit[2, 1] = RKC[2]
  RKA_explicit[Nstages, 1] = RKA_explicit[Nstages, 2]

  RKC[1] = 0
  RKC[Nstages] = 1

  # conversion to static arrays
  RKA_explicit = SMatrix{Nstages, Nstages}(RKA_explicit)
  RKA_implicit = SMatrix{Nstages, Nstages}(RKA_implicit)
  RKB = SVector{Nstages}(RKB)
  RKC = SVector{Nstages}(RKC)

  ark = AdditiveRungeKutta(F, L, linearsolver,
                           RKA_explicit, RKA_implicit, RKB, RKC,
                           split_nonlinear_linear,
                           variant,
                           Q; dt=dt, t0=t0)
end
