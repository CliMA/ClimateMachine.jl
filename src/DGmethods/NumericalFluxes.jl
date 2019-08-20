module NumericalFluxes

export Rusanov, DefaultGradNumericalFlux

using StaticArrays
import ..DGmethods:  BalanceLaw, Grad, Vars,
   vars_state, vars_diffusive, vars_aux, vars_gradient, boundarycondition!, wavespeed, flux!, diffusive!,
   num_state, num_gradient


"""
    GradNumericalFlux

Any `P <: GradNumericalFlux` should define methods for:

   diffusive_penalty!(gnf::P, bl::BalanceLaw, diffF, nM, QM, QdiffM, QauxM, QP, QdiffP, QauxP, t)
l_Qvisc, nM, l_GM, l_QM, l_auxM, l_GP, l_QP, l_auxP, t)
   diffusive_boundary_penalty!(gnf::P, bl::BalanceLaw, l_Qvisc, nM, l_GM, l_QM, l_auxM, l_GP, l_QP, l_auxP, bctype, t)

"""
abstract type GradNumericalFlux end

function diffusive_penalty! end
function diffusive_boundary_penalty! end

"""
    DefaultGradNumericalFlux <: GradNumericalFlux

"""
struct DefaultGradNumericalFlux <: GradNumericalFlux
end

function diffusive_penalty!(::DefaultGradNumericalFlux, bl::BalanceLaw, 
  VF, nM, 
  velM, QM, aM, 
  velP, QP, aP, t)
  DFloat = eltype(QM)

  @inbounds begin
    ndim = 3
    ngradstate = num_gradient(bl,DFloat)
    n_Δvel = similar(VF, Size(ndim, ngradstate))
    for j = 1:ngradstate, i = 1:ndim
      n_Δvel[i, j] = nM[i] * (velP[j] - velM[j]) / 2
    end
    diffusive!(bl, Vars{vars_diffusive(bl,DFloat)}(VF), Grad{vars_gradient(bl,DFloat)}(n_Δvel),
      Vars{vars_state(bl,DFloat)}(QM), Vars{vars_aux(bl,DFloat)}(aM), t)
  end
end

@inline diffusive_boundary_penalty!(::DefaultGradNumericalFlux, bl::BalanceLaw, VF, _...) = VF.=0


"""
    DivNumericalFlux

Any `N <: DivNumericalFlux` should define the a method for

    numerical_flux!(dnf::N, bl::BalanceLaw, F, nM, QM, QdiffM, QauxM, QP, QdiffP, QauxP, t)

where
- `F` is the numerical flux array
- `nM` is the unit normal
- `QM`/`QP` are the minus/positive state arrays
- `QdiffM`/`QdiffP` are the minus/positive diffusive state arrays
- `QdiffM`/`QdiffP` are the minus/positive auxiliary state arrays
- `t` is the time

An optional method can also be defined for

    boundary_numerical_flux!(dnf::N, bl::BalanceLaw, F, nM, QM, QdiffM, QauxM, QP, QdiffP, QauxP, bctype, t)

"""
abstract type DivNumericalFlux end

function numerical_flux! end

function numerical_boundary_flux!(dnf::DivNumericalFlux, bl::BalanceLaw,
                                  F::MArray{Tuple{nstate}}, nM,
                                  QM, QVM, auxM,
                                  QP, QVP, auxP,
                                  bctype, t,
                                  Q1, QV1, aux1) where {nstate}
  DFloat = eltype(F)
  boundarycondition!(bl, Vars{vars_state(bl,DFloat)}(QP), Vars{vars_diffusive(bl,DFloat)}(QVP), Vars{vars_aux(bl,DFloat)}(auxP),
                     nM, Vars{vars_state(bl,DFloat)}(QM), Vars{vars_diffusive(bl,DFloat)}(QVM), Vars{vars_aux(bl,DFloat)}(auxM),
                     bctype, t, Vars{vars_state(bl,DFloat)}(Q1), Vars{vars_state(bl,DFloat)}(QV1), Vars{vars_state(bl,DFloat)}(aux1))
  numerical_flux!(dnf, bl, F, nM, QM, QVM, auxM, QP, QVP, auxP, t)
end



"""
    Rusanov <: DivNumericalFlux

The Rusanov (aka local Lax-Friedrichs) numerical flux.

# Usage

    Rusanov()

Requires a `flux! and `wavespeed` method for the balance law.
"""
struct Rusanov <: DivNumericalFlux
end


function numerical_flux!(::Rusanov, bl::BalanceLaw,
                         F::MArray, nM,
                         QM, QVM, auxM,
                         QP, QVP, auxP,
                         t)
  DFloat = eltype(F)
  nstate = num_state(bl,DFloat)

  λM = wavespeed(bl, nM, Vars{vars_state(bl,DFloat)}(QM), Vars{vars_aux(bl,DFloat)}(auxM), t)
  FM = similar(F, Size(3, nstate))
  flux!(bl, Grad{vars_state(bl,DFloat)}(FM), Vars{vars_state(bl,DFloat)}(QM), Vars{vars_diffusive(bl,DFloat)}(QVM), Vars{vars_aux(bl,DFloat)}(auxM), t)
  
  λP = wavespeed(bl, nM, Vars{vars_state(bl,DFloat)}(QP), Vars{vars_aux(bl,DFloat)}(auxP), t)
  FP = similar(F, Size(3, nstate))
  flux!(bl, Grad{vars_state(bl,DFloat)}(FP), Vars{vars_state(bl,DFloat)}(QP), Vars{vars_diffusive(bl,DFloat)}(QVP), Vars{vars_aux(bl,DFloat)}(auxP), t)

  λ  =  max(λM, λP)

  #TODO: support a "computeQjump!" function
  # if computeQjump! === nothing
    @inbounds for s = 1:nstate
      F[s] = (nM[1] * (FM[1, s] + FP[1, s]) + nM[2] * (FM[2, s] + FP[2, s]) +
              nM[3] * (FM[3, s] + FP[3, s]) + λ * (QM[s] - QP[s])) / 2
    end
  # else
  #   ΔQ = copy(QM)
  #   computeQjump!(ΔQ, QM, auxM, QP, auxP)
  #   @inbounds for s = 1:nstate
  #     F[s] = (nM[1] * (FM[1, s] + FP[1, s]) + nM[2] * (FM[2, s] + FP[2, s]) +
  #             nM[3] * (FM[3, s] + FP[3, s]) + λ * ΔQ[s]) / 2
  #   end
  # end
end



end
