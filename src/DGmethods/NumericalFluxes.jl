module NumericalFluxes

export Rusanov, CentralGradPenalty, CentralNumericalFluxDiffusive

using StaticArrays
import ..DGmethods: BalanceLaw, Grad, Vars, vars_state, vars_diffusive,
                    vars_aux, vars_gradient, boundary_state!, wavespeed,
                    flux_nondiffusive!, flux_diffusive!, diffusive!, num_state,
                    num_gradient, gradvariables!

"""
    GradNumericalPenalty

Any `P <: GradNumericalPenalty` should define methods for:

   diffusive_penalty!(gnf::P, bl::BalanceLaw, diffF, nM, QM, QdiffM, QauxM, QP,
                      QdiffP, QauxP, t)
   diffusive_boundary_penalty!(gnf::P, bl::BalanceLaw, l_Qvisc, nM, l_GM, l_QM,
                               l_auxM, l_GP, l_QP, l_auxP, bctype, t)

"""
abstract type GradNumericalPenalty end

function diffusive_penalty! end
function diffusive_boundary_penalty! end

"""
    CentralGradPenalty <: GradNumericalPenalty

"""
struct CentralGradPenalty <: GradNumericalPenalty end

function diffusive_penalty!(::CentralGradPenalty, bl::BalanceLaw,
                            VF, nM, diffM, QM, aM, diffP, QP, aP, t)
  DFloat = eltype(QM)

  @inbounds begin
    ndim = 3
    ngradstate = num_gradient(bl,DFloat)
    n_Δdiff = similar(VF, Size(ndim, ngradstate))
    for j = 1:ngradstate, i = 1:ndim
      n_Δdiff[i, j] = nM[i] * (diffP[j] - diffM[j]) / 2
    end
    diffusive!(bl, Vars{vars_diffusive(bl,DFloat)}(VF),
               Grad{vars_gradient(bl,DFloat)}(n_Δdiff),
               Vars{vars_state(bl,DFloat)}(QM), Vars{vars_aux(bl,DFloat)}(aM),
               t)
  end
end

function diffusive_boundary_penalty!(nf::CentralGradPenalty, bl::BalanceLaw,
                                     VF, nM, diffM, QM, aM, diffP, QP, aP,
                                     bctype, t, Q1, aux1)
  DFloat = eltype(diffP)
  boundary_state!(nf, bl, Vars{vars_state(bl,DFloat)}(QP),
                  Vars{vars_aux(bl,DFloat)}(aP), nM,
                  Vars{vars_state(bl,DFloat)}(QM),
                  Vars{vars_aux(bl,DFloat)}(aM), bctype, t,
                  Vars{vars_state(bl,DFloat)}(Q1),
                  Vars{vars_aux(bl,DFloat)}(aux1))

  gradvariables!(bl, Vars{vars_gradient(bl,DFloat)}(diffP),
                 Vars{vars_state(bl,DFloat)}(QP),
                 Vars{vars_aux(bl,DFloat)}(aP), t)

  diffusive_penalty!(nf, bl, VF, nM, diffM, QM, aM, diffP, QP, aP, t)
end


"""
    NumericalFluxNonDiffusive

Any `N <: NumericalFluxNonDiffusive` should define the a method for

    numerical_flux_nondiffusive!(nf::N, bl::BalanceLaw, F, nM, QM, QauxM, QP,
                                 QauxP, t)

where
- `F` is the numerical flux array
- `nM` is the unit normal
- `QM`/`QP` are the minus/positive state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_nondiffusive!(nf::N, bl::BalanceLaw, F, nM, QM,
                                          QauxM, QP, QauxP, bctype, t)

"""
abstract type NumericalFluxNonDiffusive end

function numerical_flux_nondiffusive! end

function numerical_boundary_flux_nondiffusive!(nf::NumericalFluxNonDiffusive,
                                               bl::BalanceLaw,
                                               F::MArray{Tuple{nstate}}, nM, QM,
                                               auxM, QP, auxP, bctype, t,
                                               Q1, aux1) where {nstate}
  DFloat = eltype(F)
  boundary_state!(nf, bl, Vars{vars_state(bl,DFloat)}(QP),
                  Vars{vars_aux(bl,DFloat)}(auxP), nM,
                  Vars{vars_state(bl,DFloat)}(QM),
                  Vars{vars_aux(bl,DFloat)}(auxM), bctype, t,
                  Vars{vars_state(bl,DFloat)}(Q1),
                  Vars{vars_aux(bl,DFloat)}(aux1))
  numerical_flux_nondiffusive!(nf, bl, F, nM, QM, auxM, QP, auxP, t)
end



"""
    Rusanov <: NumericalFluxNonDiffusive

The Rusanov (aka local Lax-Friedrichs) numerical flux.

# Usage

    Rusanov()

Requires a `flux_nondiffusive!` and `wavespeed` method for the balance law.
"""
struct Rusanov <: NumericalFluxNonDiffusive end


function numerical_flux_nondiffusive!(::Rusanov, bl::BalanceLaw, F::MArray, nM,
                                      QM, auxM, QP, auxP, t)
  DFloat = eltype(F)
  nstate = num_state(bl,DFloat)

  λM = wavespeed(bl, nM, Vars{vars_state(bl,DFloat)}(QM),
                 Vars{vars_aux(bl,DFloat)}(auxM), t)
  FM = similar(F, Size(3, nstate))
  fill!(FM, -zero(eltype(FM)))
  flux_nondiffusive!(bl, Grad{vars_state(bl,DFloat)}(FM),
                     Vars{vars_state(bl,DFloat)}(QM),
                     Vars{vars_aux(bl,DFloat)}(auxM), t)

  λP = wavespeed(bl, nM, Vars{vars_state(bl,DFloat)}(QP),
                 Vars{vars_aux(bl,DFloat)}(auxP), t)
  FP = similar(F, Size(3, nstate))
  fill!(FP, -zero(eltype(FP)))
  flux_nondiffusive!(bl, Grad{vars_state(bl,DFloat)}(FP),
                     Vars{vars_state(bl,DFloat)}(QP),
                     Vars{vars_aux(bl,DFloat)}(auxP), t)

  λ  =  max(λM, λP)

  @inbounds for s = 1:nstate
    F[s] += (nM[1] * (FM[1, s] + FP[1, s]) + nM[2] * (FM[2, s] + FP[2, s]) +
             nM[3] * (FM[3, s] + FP[3, s]) + λ * (QM[s] - QP[s])) / 2
  end
end

"""
    NumericalFluxDiffusive

Any `N <: NumericalFluxDiffusive` should define the a method for

    numerical_flux_diffusive!(nf::N, bl::BalanceLaw, F, nM, QM, QdiffM, QauxM, QP,
                              QdiffP, QauxP, t)

where
- `F` is the numerical flux array
- `nM` is the unit normal
- `QM`/`QP` are the minus/positive state arrays
- `QdiffM`/`QdiffP` are the minus/positive diffusive state arrays
- `QdiffM`/`QdiffP` are the minus/positive auxiliary state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_diffusive!(nf::N, bl::BalanceLaw, F, nM, QM, QdiffM,
                                       QauxM, QP, QdiffP, QauxP, bctype, t)

"""
abstract type NumericalFluxDiffusive end

function numerical_flux_diffusive! end

function numerical_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                                            bl::BalanceLaw,
                                            F::MArray{Tuple{nstate}}, nM,
                                            QM, QVM, auxM, QP, QVP, auxP,
                                            bctype, t, Q1, QV1,
                                            aux1) where {nstate}
  DFloat = eltype(F)
  boundary_state!(nf, bl, Vars{vars_state(bl,DFloat)}(QP),
                  Vars{vars_diffusive(bl,DFloat)}(QVP),
                  Vars{vars_aux(bl,DFloat)}(auxP), nM,
                  Vars{vars_state(bl,DFloat)}(QM),
                  Vars{vars_diffusive(bl,DFloat)}(QVM),
                  Vars{vars_aux(bl,DFloat)}(auxM), bctype, t,
                  Vars{vars_state(bl,DFloat)}(Q1),
                  Vars{vars_diffusive(bl,DFloat)}(QV1),
                  Vars{vars_aux(bl,DFloat)}(aux1))
  numerical_flux_diffusive!(nf, bl, F, nM, QM, QVM, auxM, QP, QVP, auxP, t)
end

"""
    CentralNumericalFluxDiffusive <: NumericalFluxDiffusive

The central numerical flux for diffusive terms

# Usage

    CentralNumericalFluxDiffusive()

Requires a `flux_diffusive!` for the balance law.
"""
struct CentralNumericalFluxDiffusive <: NumericalFluxDiffusive end


function numerical_flux_diffusive!(::CentralNumericalFluxDiffusive,
                                   bl::BalanceLaw, F::MArray, nM,
                                   QM, QVM, auxM,
                                   QP, QVP, auxP,
                                   t)
  DFloat = eltype(F)
  nstate = num_state(bl,DFloat)

  FM = similar(F, Size(3, nstate))
  fill!(FM, -zero(eltype(FM)))
  flux_diffusive!(bl, Grad{vars_state(bl,DFloat)}(FM),
                  Vars{vars_state(bl,DFloat)}(QM),
                  Vars{vars_diffusive(bl,DFloat)}(QVM),
                  Vars{vars_aux(bl,DFloat)}(auxM), t)

  FP = similar(F, Size(3, nstate))
  fill!(FP, -zero(eltype(FP)))
  flux_diffusive!(bl, Grad{vars_state(bl,DFloat)}(FP),
                  Vars{vars_state(bl,DFloat)}(QP),
                  Vars{vars_diffusive(bl,DFloat)}(QVP),
                  Vars{vars_aux(bl,DFloat)}(auxP), t)

  @inbounds for s = 1:nstate
    F[s] += (nM[1] * (FM[1, s] + FP[1, s]) + nM[2] * (FM[2, s] + FP[2, s]) +
             nM[3] * (FM[3, s] + FP[3, s])) / 2
  end
end


end
