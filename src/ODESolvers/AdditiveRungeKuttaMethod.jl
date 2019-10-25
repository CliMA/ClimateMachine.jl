module AdditiveRungeKuttaMethod
export AdditiveRungeKutta
export ARK2GiraldoKellyConstantinescu, ARK548L2SA2KennedyCarpenter

using GPUifyLoops
include("AdditiveRungeKuttaMethod_kernels.jl")
using StaticArrays
include("SchurModel.jl")


using ..ODESolvers
ODEs = ODESolvers
using ..SpaceMethods
using ..LinearSolvers
using ..MPIStateArrays: device, realview
using Printf
using LinearAlgebra

using CLIMA.GeneralizedMinimalResidualSolver: GeneralizedMinimalResidual

"""
    AdditiveRungeKutta(f, l, linsol, RKAe, RKAi, RKB, RKC, Q; dt, t0 = 0)

This is a time stepping object for implicit-explicit time stepping of the
decomposed differential equation given by the chosen linear operator `l`,
the full right-hand-side function `f` and the state `Q`, i.e.,

```math
  \\dot{Q} = [l(Q, t)] + [f(Q, t) - l(Q, t)]
```

with the required time step size `dt` and optional initial time `t0`. The
linear operator `l` is integrated implicitly whereas the remaining part
`f - l` is integrated explicitly. This time stepping object is intended
to be passed to the `solve!` command.

The constructor builds an additive Runge--Kutta scheme 
based on the provided `RKAe`, `RKAi`, `RKB` and `RKC` coefficient arrays.
The resulting linear systems are solved using the provided `linsol` function.

The available concrete implementations are:

  - [`ARK2GiraldoKellyConstantinescu`](@ref)
  - [`ARK548L2SA2KennedyCarpenter`](@ref)
"""
mutable struct AdditiveRungeKutta{T, RT, AT, AT1, LT, Nstages, Nstages_sq} <: ODEs.AbstractODESolver
  "time step"
  dt::RT
  "time"
  t::RT
  "rhs function"
  rhs!
  "rhs linear operator"
  rhs_linear!
  "linear solver"
  linearsolver::LT
  "Storage for solution during the AdditiveRungeKutta update"
  Qstages::NTuple{Nstages, AT}
  "Storage for RHS during the AdditiveRungeKutta update"
  Rstages::NTuple{Nstages, AT}
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
  split_nonlinear_linear::Bool
  schur_lhs::Union{Nothing, DGModel}
  schur_rhs::Union{Nothing, DGModel}
  schur_upd::Union{Nothing, DGModel}
  schurP::AT1
  schurR::AT1
  schur::Bool
  Lstages::NTuple{Nstages, AT}
  noqtt::Bool

  function AdditiveRungeKutta(rhs!,
                              rhs_linear!,
                              linearsolver::AbstractLinearSolver,
                              RKA_explicit, RKA_implicit, RKB, RKC,
                              split_nonlinear_linear,
                              schur,
                              noqtt,
                              Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}

    @assert dt != nothing

    T = eltype(Q)
    RT = real(T)
    
    nstages = length(RKB)

    Qstages = (Q, ntuple(i -> similar(Q), nstages - 1)...)
    Rstages = ntuple(i -> similar(Q), nstages)
   
    if noqtt
      Lstages = ntuple(i -> similar(Q), nstages)
    else
      Lstages = Rstages
    end
    
    Qhat = similar(Q)
    Qtt = similar(Q)

    grid = rhs!.grid
    schur_lhs = nothing
    schur_rhs = nothing
    schur_upd = nothing
    schurP = nothing
    schurR = nothing
    if schur
      schur_lhs = DGModel(SchurLHSModel(),
                          grid,
                          ZeroNumFluxNonDiffusive(),
                          CentralNumericalFluxDiffusive(),
                          CentralGradPenalty())
      schur_rhs = DGModel(SchurRHSModel(),
                          grid,
                          CentralNumericalFluxNonDiffusive(),
                          CentralNumericalFluxDiffusive(),
                          CentralGradPenalty(),
                          auxstate=schur_lhs.auxstate)
      
      schur_upd = DGModel(SchurUpdateModel(),
                          grid,
                          ZeroNumFluxNonDiffusive(),
                          CentralNumericalFluxDiffusive(),
                          CentralGradPenalty())
                          #auxstate=schur_lhs.auxstate)

      schurP = init_ode_state(schur_lhs, zero(eltype(Q)))
      #schurR = init_ode_state(schur_rhs, zero(eltype(Q)))
      schurR = similar(schurP)
     
      # init auxstate
      # h0 = (ρe + p) / ρ
      # isentropic vortex
      #schur_lhs.auxstate[:, 1, :] .= (rhs!.auxstate[:, 5, :] + rhs!.auxstate[:, 6, :]) ./ rhs!.auxstate[:, 4, :]
      # rtb
      @views schur_lhs.auxstate[:, 1, :] .= (rhs!.auxstate[:, 9, :] .+
                                             rhs!.auxstate[:, 11, :]) ./ rhs!.auxstate[:, 8, :] .- rhs!.auxstate[:, 4, :]
      # ∇h0
      grad_auxiliary_state!(schur_lhs, 1, (2, 3, 4))
      # Φ
      # isentropic vortex
      #schur_lhs.auxstate[:, 5, :] .= 0
      # rtb
      @views schur_lhs.auxstate[:, 5, :] .= rhs!.auxstate[:, 4, :] 
      @views schur_upd.auxstate[:, 1:5, :] .= schur_lhs.auxstate[:, 1:5, :]
      
      #schur_rhs.auxstate[:, 6:10, :] .= schur_lhs.auxstate[:, 1:5, :]

      #for i = 1:12
      #  @show maximum(rhs!.auxstate[:, i, :])
      #end

      #FIXME
      linearsolver = GeneralizedMinimalResidual(10, schurP, 1e-5)
    end

    LT = typeof(linearsolver)
    AT1 = typeof(schurP)

    #@show typeof(schurP)
    #@show typeof(schurR)
    #@show typeof(schurU)
    new{T, RT, AT, AT1, LT, nstages, nstages ^ 2}(RT(dt), RT(t0),
                                             rhs!, rhs_linear!, linearsolver,
                                             Qstages, Rstages, Qhat, Qtt,
                                             RKA_explicit, RKA_implicit, RKB, RKC,
                                             split_nonlinear_linear,
                                             schur_lhs, schur_rhs, schur_upd,
                                             schurP, schurR,
                                             schur,
                                             Lstages, noqtt)
  end
end

#function AdditiveRungeKutta(spacedisc::AbstractSpaceMethod,
#                            spacedisc_linear::AbstractSpaceMethod,
#                            linearsolver::AbstractLinearSolver,
#                            RKA_explicit, RKA_implicit, RKB, RKC,
#                            split_nonlinear_linear,
#                            Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}
#  rhs! = (x...; increment) -> SpaceMethods.odefun!(spacedisc, x..., increment = increment)
#  rhs_linear! = (x...; increment) -> SpaceMethods.odefun!(spacedisc_linear, x..., increment = increment)
#  AdditiveRungeKutta(rhs!, rhs_linear!, linearsolver,
#                     RKA_explicit, RKA_implicit, RKB, RKC,
#                     split_nonlinear_linear,
#                     Q; dt=dt, t0=t0)
#end


"""
    ARK2GiraldoKellyConstantinescu(f, l, linsol, Q; dt, t0 = 0)

This function returns an [`AdditiveRungeKutta`](@ref) 
time stepping object for implicit-explicit time stepping of the
decomposed differential equation given by the chosen linear operator `l`,
the full right-hand-side function `f` and the state `Q`, i.e.,

```math
  \\dot{Q} = [l(Q, t)] + [f(Q, t) - l(Q, t)]
```

with the required time step size `dt` and optional initial time `t0`. The
linear operator `l` is integrated implicitly whereas the remaining part
`f - l` is integrated explicitly. This time stepping object is intended
to be passed to the `solve!` command.

The resulting linear systems are solved using the provided `linsol` function.

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
                                        schur=false, noqtt=false) where {AT<:AbstractArray}

  @assert dt != nothing

  T = eltype(Q)
  RT = real(T)
  
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

  AdditiveRungeKutta(F, L, linearsolver,
                     RKA_explicit, RKA_implicit, RKB, RKC,
                     split_nonlinear_linear,
                     schur,
                     noqtt,
                     Q; dt=dt, t0=t0)
end

"""
    ARK548L2SA2KennedyCarpenter(f, l, linsol, Q; dt, t0 = 0)

This function returns an [`AdditiveRungeKutta`](@ref) 
time stepping object for implicit-explicit time stepping of the
decomposed differential equation given by the chosen linear operator `l`,
the full right-hand-side function `f` and the state `Q`, i.e.,

```math
  \\dot{Q} = [l(Q, t)] + [f(Q, t) - l(Q, t)]
```

with the required time step size `dt` and optional initial time `t0`. The
linear operator `l` is integrated implicitly whereas the remaining part
`f - l` is integrated explicitly. This time stepping object is intended
to be passed to the `solve!` command.

The resulting linear systems are solved using the provided `linsol` function.

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
                                     schur=false, noqtt=false) where {AT<:AbstractArray}

  @assert dt != nothing

  T = eltype(Q)
  RT = real(T)

  nstages = 8
  gamma = RT(2 // 9)

  # declared as Arrays for mutability, later these will be converted to static arrays
  RKA_explicit = zeros(RT, nstages, nstages)
  RKA_implicit = zeros(RT, nstages, nstages)
  RKB = zeros(RT, nstages)
  RKC = zeros(RT, nstages)

  # the main diagonal
  for is = 2:nstages
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
  
  for is = 2:nstages
    RKA_implicit[is, 1] = RKA_implicit[is, 2]
  end
 
  for is = 1:nstages-1
    RKA_implicit[nstages, is] = RKB[is]
  end

  RKB[1] = RKB[2]
  RKB[8] = gamma
  
  RKA_explicit[2, 1] = RKC[2]
  RKA_explicit[nstages, 1] = RKA_explicit[nstages, 2]

  RKC[1] = 0
  RKC[nstages] = 1

  # conversion to static arrays
  RKA_explicit = SMatrix{nstages, nstages}(RKA_explicit)
  RKA_implicit = SMatrix{nstages, nstages}(RKA_implicit)
  RKB = SVector{nstages}(RKB)
  RKC = SVector{nstages}(RKC)

  ark = AdditiveRungeKutta(F, L, linearsolver,
                           RKA_explicit, RKA_implicit, RKB, RKC,
                           split_nonlinear_linear,
                           schur,
                           noqtt,
                           Q; dt=dt, t0=t0)
end

ODEs.updatedt!(ark::AdditiveRungeKutta, dt) = (ark.dt = dt)
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
  linearsolver = ark.linearsolver
  RKA_explicit, RKA_implicit = ark.RKA_explicit, ark.RKA_implicit
  RKB, RKC = ark.RKB, ark.RKC
  rhs!, rhs_linear! = ark.rhs!, ark.rhs_linear!
  Qstages, Rstages = ark.Qstages, ark.Rstages
  Qhat, Qtt = ark.Qhat, ark.Qtt
  split_nonlinear_linear = ark.split_nonlinear_linear
  
  schur = ark.schur
  schur_lhs = ark.schur_lhs
  schur_rhs = ark.schur_rhs
  schur_upd = ark.schur_upd
  schurP = ark.schurP
  schurR = ark.schurR
  
  noqtt = ark.noqtt
  Lstages = ark.Lstages

  rv_Q = realview(Q)
  rv_Qstages = realview.(Qstages)
  rv_Rstages = realview.(Rstages)
  rv_Lstages = realview.(Lstages)
  rv_Qhat = realview(Qhat)
  rv_Qtt = realview(Qtt)

  nstages = length(RKB)

  threads = 256
  blocks = div(length(rv_Q) + threads - 1, threads)

  # calculate the rhs at first stage to initialize the stage loop
  rhs!(Rstages[1], Qstages[1], p, time + RKC[1] * dt, increment = false)
  
  if noqtt
    rhs_linear!(Lstages[1], Qstages[1], p, time + RKC[1] * dt, increment = false)
    for istage = 2:nstages
      stagetime = time + RKC[istage] * dt
      @launch(device(Q), threads = threads, blocks = blocks,
              stage_update_noqtt!(rv_Q, rv_Lstages, rv_Rstages, rv_Qhat, rv_Qstages,
                                  RKA_explicit, RKA_implicit, dt, Val(istage),
                                  Val(split_nonlinear_linear)))
      
      α = dt * RKA_implicit[istage, istage]
      linearoperator! = function(LQ, Q)
        rhs_linear!(LQ, Q, p, stagetime; increment = false)
        @. LQ = Q - α * LQ
      end
      linearsolve!(linearoperator!, Qstages[istage], Qhat, linearsolver)

      rhs!(Rstages[istage], Qstages[istage], p, stagetime, increment = false)
      rhs_linear!(Lstages[istage], Qstages[istage], p, stagetime, increment = false)
    end

    @launch(device(Q), threads = threads, blocks = blocks,
            solution_update_noqtt!(rv_Q, rv_Lstages, rv_Rstages, RKB, dt,
                                   Val(split_nonlinear_linear), Val(nstages)))
  else
    # note that it is important that this loop does not modify Q!
    for istage = 2:nstages
      stagetime = time + RKC[istage] * dt

      # this kernel also initializes Qtt for the linear solver
      @launch(device(Q), threads = threads, blocks = blocks,
              stage_update!(rv_Q, rv_Qstages, rv_Rstages, rv_Qhat, rv_Qtt,
                            RKA_explicit, RKA_implicit, dt, Val(istage),
                            Val(split_nonlinear_linear), slow_δ, slow_rv_dQ))

      #solves Q_tt = Qhat + dt * RKA_implicit[istage, istage] * rhs_linear!(Q_tt)
      α = dt * RKA_implicit[istage, istage]
      #let 
      #  ρ = extrema(Qtt[:, 1, :])
      #  u = extrema(Qtt[:, 2, :])
      #  v = extrema(Qtt[:, 3, :])
      #  w = extrema(Qtt[:, 4, :])
      #  ρe = extrema(Qtt[:, 5, :])
      #  @info @sprintf """Before stage %d
      #  ρ  = (%.16e, %.16e)
      #  u  = (%.16e, %.16e)
      #  v  = (%.16e, %.16e)
      #  w  = (%.16e, %.16e)
      #  ρe = (%.16e, %.16e)
      #  """ istage ρ... u... v... w... ρe...
      #end
      γ = 1 / (1 - kappa_d)
      if !schur
        linearoperator! = function(LQ, Q)
          rhs_linear!(LQ, Q, p, stagetime; increment = false)
          @. LQ = Q - α * LQ
        end
        linearsolve!(linearoperator!, Qtt, Qhat, linearsolver)

        #aux = rhs!.auxstate
        #lastaux = size(aux, 2)
        ##isentropic vortex
        ##aux[:, lastaux-3, :] .= (@. (γ - 1) * (Qtt[:, 5, :] + Qtt[:, 1, :] *  R_d * T_0 / (γ - 1)))
        ## rtb
        #aux[:, lastaux-3, :] .= (@. (γ - 1) * (Qtt[:, 5, :] - Qtt[:, 1, :] * (aux[:, 4, :] - R_d * T_0 / (γ - 1))))
        #P = extrema(aux[:, lastaux-3, :])
        #grad_auxiliary_state!(rhs!, lastaux-3, (lastaux-2, lastaux-1, lastaux))
        #∇P = (extrema(aux[:, lastaux-2, :])... , extrema(aux[:, lastaux-1, :])..., extrema(aux[:, lastaux, :])...) 
      else
        aux = rhs!.auxstate
        #schurP .= 0
        #isentropic vortex
        #schurP[:, 1, :] .= (@. (γ - 1) * (Qtt[:, 5, :] + Qtt[:, 1, :] *  R_d * T_0 / (γ - 1)))
        #rtb
        @views schurP[:, 1, :] .= (@. (γ - 1) * (Qtt[:, 5, :] - Qtt[:, 1, :] * (aux[:, 4, :] - R_d * T_0 / (γ - 1))))
        #schur_rhs.auxstate[:, 1:5, :] .= Qhat[:, 1:5, :]
        schur_rhs(schurR, Qhat, p, α; increment = false)
        linearoperator! = function(LQ, Q)
          schur_lhs(LQ, Q, p, α; increment = false)
        end
        linearsolve!(linearoperator!, schurP, schurR, linearsolver)
        #schur_lhs(schurR, schurP, p, α; increment = false)
        #∇P = (extrema(schur_lhs.diffstate[:, 1, :])...,
        #      extrema(schur_lhs.diffstate[:, 2, :])...,
        #      extrema(schur_lhs.diffstate[:, 3, :])...)

        #P = extrema(schurP[:, 1, :])
        
        #lastaux = size(aux, 2)
        #aux[:, lastaux-3, :] .= schurP[:, 1, :]
        #grad_auxiliary_state!(rhs!, lastaux-3, (lastaux-2, lastaux-1, lastaux))
        #∇P = (extrema(aux[:, lastaux-2, :])... , extrema(aux[:, lastaux-1, :])..., extrema(aux[:, lastaux, :])...) 

        #schur_lhs.diffstate[:, 1, :] .= aux[:, lastaux-2, :]
        #schur_lhs.diffstate[:, 2, :] .= aux[:, lastaux-1, :]
        #schur_lhs.diffstate[:, 3, :] .= aux[:, lastaux  , :]
        #
        #nodal_update_schur!(schur_update!, rhs!,
        #                       rhs!.balancelaw, schur_lhs.balancelaw,
        #                       Qtt, Qhat,
        #                       schurP, schur_lhs.auxstate, schur_lhs.diffstate,
        #                       α)
       
        #@views schurU[:, 1, :] .= schurP[:, 1, :]
        #@views schurU[:, 2:6, :] .= Qtt[:, 1:5, :]
        #schur_upd(schurD, schurU, p, α; increment = false)
        ##∇P = (extrema(schurD[:, 3, :] ./ -α)...,
        ##      extrema(schurD[:, 4, :] ./ -α)..., 
        ##      extrema(schurD[:, 5, :] ./ -α)...)
        #@views Qtt[:, 1:5, :] .+= schurD[:, 2:6, :]

        @views schur_upd.auxstate[:, 6:8, :] .= Qtt[:, 2:4, :]
        schur_upd(Qtt, schurP, p, α; increment = true)
      end
      #let 
      #  ρ = extrema(Qtt[:, 1, :])
      #  u = extrema(Qtt[:, 2, :])
      #  v = extrema(Qtt[:, 3, :])
      #  w = extrema(Qtt[:, 4, :])
      #  ρe = extrema(Qtt[:, 5, :])
      #  @info @sprintf """After stage %d
      #  P    = (%.16e, %.16e)
      #  ∇P_x = (%.16e, %.16e)
      #  ∇P_y = (%.16e, %.16e)
      #  ∇P_z = (%.16e, %.16e)
      #  ρ    = (%.16e, %.16e)
      #  u    = (%.16e, %.16e)
      #  v    = (%.16e, %.16e)
      #  w    = (%.16e, %.16e)
      #  ρe   = (%.16e, %.16e)
      #  """ istage P... ∇P... ρ... u... v... w... ρe...
      #  flush(stderr)
      #end
      
      #update Qstages
      Qstages[istage] .+= Qtt
      
      rhs!(Rstages[istage], Qstages[istage], p, stagetime, increment = false)
    end
   
    if split_nonlinear_linear
      for istage = 1:nstages
        stagetime = time + RKC[istage] * dt
        rhs_linear!(Rstages[istage], Qstages[istage], p, stagetime, increment = true)
      end
    end

    # compose the final solution
    @launch(device(Q), threads = threads, blocks = blocks,
            solution_update!(rv_Q, rv_Rstages, RKB, dt, Val(nstages), slow_δ,
                             slow_rv_dQ, slow_scaling))
  end # qtt

end

end
