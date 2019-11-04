module NumericalFluxes

export Rusanov, CentralGradPenalty, CentralNumericalFluxDiffusive,
       CentralNumericalFluxNonDiffusive

using StaticArrays
using GPUifyLoops: @unroll
import ..DGmethods: BalanceLaw, Grad, Vars, vars_state, vars_diffusive,
                    vars_aux, vars_gradient, boundary_state!, wavespeed,
                    flux_nondiffusive!, flux_diffusive!, diffusive!, num_state,
                    num_gradient, gradvariables!

"""
    GradNumericalPenalty

Any `P <: GradNumericalPenalty` should define methods for:

   diffusive_penalty!(gnf::P, bl::BalanceLaw, σ, n⁻, Y⁻, G⁻, A⁻, Y⁺,
                      G⁺, A⁺, t)
   diffusive_boundary_penalty!(gnf::P, bl::BalanceLaw, l_∇G, n⁻, l_G⁻, l_Y⁻,
                               l_A⁻, l_G⁺, l_Y⁺, l_A⁺, bctype, t)

"""
abstract type GradNumericalPenalty end

function diffusive_penalty! end
function diffusive_boundary_penalty! end

"""
    CentralGradPenalty <: GradNumericalPenalty

"""
struct CentralGradPenalty <: GradNumericalPenalty end

function diffusive_penalty!(::CentralGradPenalty, bl::BalanceLaw,
                            σ, n⁻, G⁻, Y⁻, A⁻, G⁺, Y⁺, A⁺, t)
  FT = eltype(Y⁻)

  @inbounds begin
    Nd = 3
    ngradstate = num_gradient(bl,FT)
    ∇G = similar(σ, Size(Nd, ngradstate))
    for j = 1:ngradstate, i = 1:Nd
      ∇G[i, j] = n⁻[i] * (G⁺[j] - G⁻[j]) / 2
    end
    diffusive!(bl,
               Vars{vars_diffusive(bl,FT)}(σ),
               Grad{vars_gradient(bl,FT)}(∇G),
               Vars{vars_state(bl,FT)}(Y⁻),
               Vars{vars_aux(bl,FT)}(A⁻),
               t)
  end
end

function diffusive_boundary_penalty!(nf::CentralGradPenalty, bl::BalanceLaw,
                                     σ, n⁻, G⁻, Y⁻, A⁻, G⁺, Y⁺, A⁺,
                                     bctype, t, Y1, A1)
  FT = eltype(G⁺)

  boundary_state!(nf, bl,
                  Vars{vars_state(bl,FT)}(Y⁺),
                  Vars{vars_aux(bl,FT)}(A⁺),
                  n⁻,
                  Vars{vars_state(bl,FT)}(Y⁻),
                  Vars{vars_aux(bl,FT)}(A⁻),
                  bctype, t,
                  Vars{vars_state(bl,FT)}(Y1),
                  Vars{vars_aux(bl,FT)}(A1))

  gradvariables!(bl,
                 Vars{vars_gradient(bl,FT)}(G⁺),
                 Vars{vars_state(bl,FT)}(Y⁺),
                 Vars{vars_aux(bl,FT)}(A⁺),
                 t)

  diffusive_penalty!(nf, bl, σ, n⁻, G⁻, Y⁻, A⁻, G⁺, Y⁺, A⁺, t)
end


"""
    NumericalFluxNonDiffusive

Any `N <: NumericalFluxNonDiffusive` should define the a method for

    numerical_flux_nondiffusive!(nf::N, bl::BalanceLaw, F, n⁻, Y⁻, A⁻, Y⁺,
                                 A⁺, t)

where
- `F` is the numerical flux array
- `n⁻` is the unit normal
- `Y⁻`/`Y⁺` are the minus/positive state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_nondiffusive!(nf::N, bl::BalanceLaw, F, n⁻, Y⁻,
                                          A⁻, Y⁺, A⁺, bctype, t)

"""
abstract type NumericalFluxNonDiffusive end

function numerical_flux_nondiffusive! end

function numerical_boundary_flux_nondiffusive!(nf::NumericalFluxNonDiffusive,
                                               bl::BalanceLaw,
                                               F::MArray{Tuple{nstate}},
                                               n⁻, Y⁻, A⁻, Y⁺, A⁺, bctype, t,
                                               Y1, A1) where {nstate}
  FT = eltype(F)

  boundary_state!(nf, bl,
                  Vars{vars_state(bl,FT)}(Y⁺),
                  Vars{vars_aux(bl,FT)}(A⁺),
                  n⁻,
                  Vars{vars_state(bl,FT)}(Y⁻),
                  Vars{vars_aux(bl,FT)}(A⁻),
                  bctype, t,
                  Vars{vars_state(bl,FT)}(Y1),
                  Vars{vars_aux(bl,FT)}(A1))

  numerical_flux_nondiffusive!(nf, bl, F, n⁻, Y⁻, A⁻, Y⁺, A⁺, t)
end



"""
    Rusanov <: NumericalFluxNonDiffusive

The Rusanov (aka local Lax-Friedrichs) numerical flux.

# Usage

    Rusanov()

Requires a `flux_nondiffusive!` and `wavespeed` method for the balance law.
"""
struct Rusanov <: NumericalFluxNonDiffusive end


function numerical_flux_nondiffusive!(::Rusanov, bl::BalanceLaw, F::MArray,
                                      n⁻, Y⁻, A⁻, Y⁺, A⁺, t)
  FT = eltype(F)
  nstate = num_state(bl,FT)

  λ⁻ = wavespeed(bl, n⁻,
                 Vars{vars_state(bl,FT)}(Y⁻),
                 Vars{vars_aux(bl,FT)}(A⁻),
                 t)

  F⁻ = similar(F, Size(3, nstate))
  fill!(F⁻, -zero(eltype(F⁻)))

  flux_nondiffusive!(bl, Grad{vars_state(bl,FT)}(F⁻),
                     Vars{vars_state(bl,FT)}(Y⁻),
                     Vars{vars_aux(bl,FT)}(A⁻), t)

  λ⁺ = wavespeed(bl, n⁻,
                 Vars{vars_state(bl,FT)}(Y⁺),
                 Vars{vars_aux(bl,FT)}(A⁺),
                 t)

  F⁺ = similar(F, Size(3, nstate))
  fill!(F⁺, -zero(eltype(F⁺)))

  flux_nondiffusive!(bl,
                     Grad{vars_state(bl,FT)}(F⁺),
                     Vars{vars_state(bl,FT)}(Y⁺),
                     Vars{vars_aux(bl,FT)}(A⁺),
                     t)

  λ  =  max(λ⁻, λ⁺)

  @unroll for s = 1:nstate
    @inbounds F[s] += 0.5 * (n⁻[1] * (F⁻[1, s] + F⁺[1, s]) +
                   n⁻[2] * (F⁻[2, s] + F⁺[2, s]) +
                   n⁻[3] * (F⁻[3, s] + F⁺[3, s]) +
                   λ * (Y⁻[s] - Y⁺[s]))
  end
end

"""
    CentralNumericalFluxNonDiffusive() <: NumericalFluxNonDiffusive

The central numerical flux for nondiffusive terms

# Usage

    CentralNumericalFluxNonDiffusive()

Requires a `flux_nondiffusive!` method for the balance law.
"""
struct CentralNumericalFluxNonDiffusive <: NumericalFluxNonDiffusive end

function numerical_flux_nondiffusive!(::CentralNumericalFluxNonDiffusive,
                                      bl::BalanceLaw, F::MArray, n⁻,
                                      Q⁻, A⁻, Q⁺, A⁺, t)
  FT = eltype(F)
  nstate = num_state(bl,FT)

  F⁻ = similar(F, Size(3, nstate))
  fill!(F⁻, -zero(eltype(F⁻)))
  flux_nondiffusive!(bl,
                     Grad{vars_state(bl,FT)}(F⁻),
                     Vars{vars_state(bl,FT)}(Q⁻),
                     Vars{vars_aux(bl,FT)}(A⁻),
                     t)

  F⁺ = similar(F, Size(3, nstate))
  fill!(F⁺, -zero(eltype(F⁺)))
  flux_nondiffusive!(bl,
                     Grad{vars_state(bl,FT)}(F⁺),
                     Vars{vars_state(bl,FT)}(Q⁺),
                     Vars{vars_aux(bl,FT)}(A⁺),
                     t)

  @unroll for s = 1:nstate
    @inbounds F[s] += 0.5 * (n⁻[1] * (F⁻[1, s] + F⁺[1, s]) +
                   n⁻[2] * (F⁻[2, s] + F⁺[2, s]) +
                   n⁻[3] * (F⁻[3, s] + F⁺[3, s]))
  end
end

"""
    NumericalFluxDiffusive

Any `N <: NumericalFluxDiffusive` should define the a method for

    numerical_flux_diffusive!(nf::N, bl::BalanceLaw, F, n⁻, Y⁻, σ⁻, A⁻, Y⁺,
                              σ⁺, A⁺, t)

where
- `F` is the numerical flux array
- `n⁻` is the unit normal
- `Y⁻`/`Y⁺` are the minus/positive state arrays
- `σ⁻`/`σ⁺` are the minus/positive diffusive state arrays
- `A⁻`/`A⁺` are the minus/positive auxiliary state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_diffusive!(nf::N, bl::BalanceLaw, F, n⁻, Y⁻, σ⁻,
                                       A⁻, Y⁺, σ⁺, A⁺, bctype, t)

"""
abstract type NumericalFluxDiffusive end

function numerical_flux_diffusive! end

function numerical_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                                            bl::BalanceLaw,
                                            F::MArray{Tuple{nstate}},
                                            n⁻, Y⁻, σ⁻, A⁻, Y⁺, σ⁺, A⁺,
                                            bctype, t, Y1, σ1,
                                            A1) where {nstate}
  FT = eltype(F)

  boundary_state!(nf, bl, Vars{vars_state(bl,FT)}(Y⁺),
                  Vars{vars_diffusive(bl,FT)}(σ⁺),
                  Vars{vars_aux(bl,FT)}(A⁺),
                  n⁻,
                  Vars{vars_state(bl,FT)}(Y⁻),
                  Vars{vars_diffusive(bl,FT)}(σ⁻),
                  Vars{vars_aux(bl,FT)}(A⁻),
                  bctype, t,
                  Vars{vars_state(bl,FT)}(Y1),
                  Vars{vars_diffusive(bl,FT)}(σ1),
                  Vars{vars_aux(bl,FT)}(A1))

  numerical_flux_diffusive!(nf, bl, F, n⁻, Y⁻, σ⁻, A⁻, Y⁺, σ⁺, A⁺, t)
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
                                   bl::BalanceLaw, F::MArray, n⁻,
                                   Y⁻, σ⁻, A⁻, Y⁺, σ⁺, A⁺, t)
  FT = eltype(F)
  nstate = num_state(bl,FT)

  F⁻ = similar(F, Size(3, nstate))
  fill!(F⁻, -zero(eltype(F⁻)))

  flux_diffusive!(bl,
                  Grad{vars_state(bl,FT)}(F⁻),
                  Vars{vars_state(bl,FT)}(Y⁻),
                  Vars{vars_diffusive(bl,FT)}(σ⁻),
                  Vars{vars_aux(bl,FT)}(A⁻),
                  t)

  F⁺ = similar(F, Size(3, nstate))
  fill!(F⁺, -zero(eltype(F⁺)))

  flux_diffusive!(bl,
                  Grad{vars_state(bl,FT)}(F⁺),
                  Vars{vars_state(bl,FT)}(Y⁺),
                  Vars{vars_diffusive(bl,FT)}(σ⁺),
                  Vars{vars_aux(bl,FT)}(A⁺),
                  t)

  @inbounds for s = 1:nstate
    F[s] += 0.5 * (n⁻[1] * (F⁻[1, s] + F⁺[1, s]) +
                   n⁻[2] * (F⁻[2, s] + F⁺[2, s]) +
                   n⁻[3] * (F⁻[3, s] + F⁺[3, s]))
  end
end


end
