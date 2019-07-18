module NumericalFluxes

export Rusanov, DefaultGradNumericalFlux

using StaticArrays
import ..DGmethods:  BalanceLaw, Grad, Vars,
   vars_state, vars_diffusive, vars_aux, vars_gradient, boundarycondition!, wavespeed, flux!, diffusive!,
   num_state, num_gradient



"""
    GradNumericalFlux

Any `P <: GradNumericalFlux` should define the following:

- `diffusive_penalty!(gnf::P, bl::BalanceLaw, l_Qvisc, nM, l_GM, l_QM, l_auxM, l_GP, l_QP, l_auxP, t)`
- `diffusive_boundary_penalty!(gnf::P, bl::BalanceLaw, l_Qvisc, nM, l_GM, l_QM, l_auxM, l_GP, l_QP, l_auxP, bctype, t)`
"""
abstract type GradNumericalFlux end

"""
    DefaultGradNumericalFlux

"""
struct DefaultGradNumericalFlux <: GradNumericalFlux
end

function diffusive_penalty!(::DefaultGradNumericalFlux, bl::BalanceLaw, 
  VF, nM, 
  velM, QM, aM, 
  velP, QP, aP, t)
  DFloat = eltype(QM)

  @inbounds begin
    ndim = 3 # should this be dimension(bl)?
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




function diffusive_penalty! end
function diffusive_boundary_penalty! end

"""
    DivNumericalFlux

Any `N <: DivNumericalFlux` should define the following:

- `numerical_flux!(dnf::N, bl::BalanceLaw, l_F, nM, l_QM, l_QviscM, l_auxM, l_QP, l_QviscP, l_auxP, t)`
- `numerical_boundary_flux!(dnf::N, bl::BalanceLaw, l_F, nM, l_QM, l_QviscM, l_auxM, l_QP, l_QviscP, l_auxP, bctype, t)`
"""
abstract type DivNumericalFlux end


function numerical_boundary_flux!(dnf::DivNumericalFlux, bl::BalanceLaw,
                                  F::MArray{Tuple{nstate}}, nM,
                                  QM, QVM, auxM,
                                  QP, QVP, auxP,
                                  bctype, t) where {nstate}
  DFloat = eltype(F)
  boundarycondition!(bl, Vars{vars_state(bl,DFloat)}(QP), Vars{vars_diffusive(bl,DFloat)}(QVP), Vars{vars_aux(bl,DFloat)}(auxP),
                     nM, Vars{vars_state(bl,DFloat)}(QM), Vars{vars_diffusive(bl,DFloat)}(QVM), Vars{vars_aux(bl,DFloat)}(auxM),
                     bctype, t)
  numerical_flux!(dnf, bl, F, nM, QM, QVM, auxM, QP, QVP, auxP, t)
end



"""
    rusanov!(F::MArray, nM, QM, QVM, auxM, QP, QVP, auxP, t, flux!, wavespeed,
             [preflux = (_...) -> (), computeQjump!])

Calculate the Rusanov (aka local Lax-Friedrichs) numerical flux given the plus
and minus side states/viscous states `QP`/`QVP` and `QM`/`QVM` using the physical
flux function `flux!` and `wavespeed` calculation.

The `flux!` has almost the same calling convention as `flux!` from
[`DGBalanceLaw`](@ref) except that `preflux(Q, aux, t)` is splatted at the end
of the call.

The function `wavespeed` should return the maximum wavespeed for a state and is
called as `wavespeed(nM, QM, auxM, t, preflux(QM, auxM, t)...)` and
`wavespeed(nM, QP, auxP, t, preflux(QP, auxP, t)...)` where `nM` is the outward
unit normal for the minus side.

When present `computeQjump!(ΔQ, QM, auxM, QP, auxP)` will be called after so
that the user specify the value to use for `QM - QP`; this is useful for
correcting `Q` to include discontinuous reference states.

!!! note

    The undocumented arguments `PM` and `PP` for the function should not be used
    by external callers and are used only internally by the function
    `rusanov_boundary_flux!`

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


# legacy methods
include("NumericalFluxes_old.jl")


end
