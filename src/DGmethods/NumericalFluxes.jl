module NumericalFluxes

export NumericalFluxNonDiffusive, NumericalFluxDiffusive, GradNumericalPenalty,
       Rusanov, CentralGradPenalty, CentralNumericalFluxDiffusive,
       CentralNumericalFluxNonDiffusive

using StaticArrays, LinearAlgebra
using GPUifyLoops: @unroll
using CLIMA.VariableTemplates
import ..DGmethods: BalanceLaw, Grad, Vars, vars_state, vars_diffusive,
                    vars_aux, vars_gradient, boundary_state!, wavespeed,
                    flux_nondiffusive!, flux_diffusive!, diffusive!, num_state,
                    num_gradient, gradvariables!,
                    num_gradient_laplacian, vars_gradient_laplacian,
                    vars_hyperdiffusive, hyperdiffusive!

"""
    GradNumericalPenalty

Any `P <: GradNumericalPenalty` should define methods for:

   gradient_penalty!(gnf::P, bl::BalanceLaw, diffF, nM, QM, QdiffM, QauxM, QP,
                     QdiffP, QauxP, t)
   gradient_boundary_penalty!(gnf::P, bl::BalanceLaw, l_Qvisc, nM, l_GM, l_QM,
                              l_auxM, l_GP, l_QP, l_auxP, bctype, t)

"""
abstract type GradNumericalPenalty end

"""
    CentralGradPenalty <: GradNumericalPenalty

"""
struct CentralGradPenalty <: GradNumericalPenalty end

function gradient_penalty!(::CentralGradPenalty, bl::BalanceLaw,
    grad_penalty::Grad{D}, n::SVector,
    transform⁻::Vars{T}, state⁻::Vars{S}, aux⁻::Vars{A},
    transform⁺::Vars{T}, state⁺::Vars{S}, aux⁺::Vars{A},
    t) where {D,T,S,A}

  parent(grad_penalty) .= n .* (parent(transform⁺) .- parent(transform⁻))' ./ 2
end

function gradient_boundary_penalty!(nf::CentralGradPenalty, bl::BalanceLaw,
    grad_penalty::Grad{D}, n::SVector,
    transform⁻::Vars{T}, state⁻::Vars{S}, aux⁻::Vars{A},
    transform⁺::Vars{T}, state⁺::Vars{S}, aux⁺::Vars{A},
    bctype, t, state1⁻::Vars{S}, aux1⁻::Vars{A}) where {D,T,S,A}

  boundary_state!(nf, bl, state⁺, aux⁺, n, state⁻, aux⁻,
    bctype, t, state1⁻, aux1⁻)

  gradvariables!(bl, transform⁺, state⁺, aux⁺, t)

  gradient_penalty!(nf, bl, grad_penalty, n,
    transform⁻, state⁻, aux⁻,
    transform⁺, state⁺, aux⁺,
    t)
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
    bl::BalanceLaw, fluxᵀn::Vars{S}, n::SVector,
    state⁻::Vars{S}, aux⁻::Vars{A}, state⁺::Vars{S}, aux⁺::Vars{A},
    bctype, t, state1⁻::Vars{S}, aux1⁻::Vars{A}) where {S,A}

  boundary_state!(nf, bl, state⁺, aux⁺, n,
    state⁻, aux⁻, bctype, t, state1⁻, aux1⁻)

  numerical_flux_nondiffusive!(nf, bl, fluxᵀn, n, state⁻, aux⁻, state⁺, aux⁺, t)
end


"""
    Rusanov <: NumericalFluxNonDiffusive

The Rusanov (aka local Lax-Friedrichs) numerical flux.

# Usage

    Rusanov()

Requires a `flux_nondiffusive!` and `wavespeed` method for the balance law.
"""
struct Rusanov <: NumericalFluxNonDiffusive end

update_penalty!(::Rusanov, ::BalanceLaw, _...) = nothing

function numerical_flux_nondiffusive!(nf::Rusanov,
    bl::BalanceLaw, fluxᵀn::Vars{S}, n::SVector,
    state⁻::Vars{S}, aux⁻::Vars{A}, state⁺::Vars{S}, aux⁺::Vars{A}, t) where {S,A}

  numerical_flux_nondiffusive!(CentralNumericalFluxNonDiffusive(),
    bl, fluxᵀn, n, state⁻, aux⁻, state⁺, aux⁺, t)

  Fᵀn = parent(fluxᵀn)
  λ⁻ = wavespeed(bl, n, state⁻, aux⁻, t)
  λ⁺ = wavespeed(bl, n, state⁺, aux⁺, t)
  λ = max(λ⁻, λ⁺)
  λΔQ = λ * (parent(state⁻) - parent(state⁺))

  # TODO: should this operate on ΔQ or λΔQ?
  update_penalty!(nf, bl, n, λ,
    Vars{S}(λΔQ), state⁻, aux⁻, state⁺, aux⁺, t)

  Fᵀn .+= λΔQ/2
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
    bl::BalanceLaw, fluxᵀn::Vars{S}, n::SVector,
    state⁻::Vars{S}, aux⁻::Vars{A}, state⁺::Vars{S}, aux⁺::Vars{A}, t) where {S,A}

  FT = eltype(fluxᵀn)
  nstate = num_state(bl,FT)
  Fᵀn = parent(fluxᵀn)

  F⁻ = similar(Fᵀn, Size(3, nstate))
  fill!(F⁻, -zero(FT))
  flux_nondiffusive!(bl, Grad{S}(F⁻), state⁻, aux⁻, t)

  F⁺ = similar(Fᵀn, Size(3, nstate))
  fill!(F⁺, -zero(FT))
  flux_nondiffusive!(bl, Grad{S}(F⁺), state⁺, aux⁺, t)

  Fᵀn .+= (F⁻ + F⁺)' * (n/2)
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
    bl::BalanceLaw, fluxᵀn::Vars{S}, n::SVector,
    state⁻::Vars{S}, diff⁻::Vars{D}, hyperdiff⁻::Vars{HD}, aux⁻::Vars{A},
    state⁺::Vars{S}, diff⁺::Vars{D}, hyperdiff⁺::Vars{HD}, aux⁺::Vars{A},
    bctype, t,
    state1⁻::Vars{S}, diff1⁻::Vars{D}, aux1⁻::Vars{A}) where {S,D,HD,A}

  boundary_state!(nf, bl, state⁺, diff⁺, aux⁺,
    n, state⁻, diff⁻, aux⁻, bctype, t,
    state1⁻, diff1⁻, aux1⁻)

  numerical_flux_diffusive!(nf, bl, fluxᵀn, n,
    state⁻, diff⁻, hyperdiff⁻, aux⁻, state⁺, diff⁺, hyperdiff⁺, aux⁺, t)
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
  bl::BalanceLaw, fluxᵀn::Vars{S}, n::SVector,
  state⁻::Vars{S}, diff⁻::Vars{D}, hyperdiff⁻::Vars{HD}, aux⁻::Vars{A},
  state⁺::Vars{S}, diff⁺::Vars{D}, hyperdiff⁺::Vars{HD}, aux⁺::Vars{A}, t) where {S,D,HD,A}

  FT = eltype(fluxᵀn)
  nstate = num_state(bl,FT)
  Fᵀn = parent(fluxᵀn)

  F⁻ = similar(Fᵀn, Size(3, nstate))
  fill!(F⁻, -zero(FT))
  flux_diffusive!(bl, Grad{S}(F⁻), state⁻, diff⁻, hyperdiff⁻, aux⁻, t)

  F⁺ = similar(Fᵀn, Size(3, nstate))
  fill!(F⁺, -zero(FT))
  flux_diffusive!(bl, Grad{S}(F⁺), state⁺, diff⁺, hyperdiff⁺, aux⁺, t)

  Fᵀn .+= (F⁻ + F⁺)' * (n/2)
end

abstract type DivNumericalPenalty end
struct CentralDivPenalty <: DivNumericalPenalty end

function divergence_penalty!(::CentralDivPenalty, bl::BalanceLaw,
                             div_penalty, nM, gradM, gradP)
  FT = eltype(gradM)
  @inbounds begin
    ndim = 3
    ngradlapstate = num_gradient_laplacian(bl,FT)
    @unroll for j = 1:ngradlapstate
      div_penalty[j] = zero(FT)
      @unroll for i = 1:ndim
        div_penalty[j] += nM[i] * (gradP[j, i] - gradM[j, i]) / 2
      end
    end
  end
end

function divergence_boundary_penalty!(nf::CentralDivPenalty, bl::BalanceLaw,
                                      div_penalty, nM, gradM, gradP, bctype)
  FT = eltype(gradM)
  boundary_state!(nf, bl,
                  Grad{vars_gradient_laplacian(bl,FT)}(gradP),
                  nM,
                  Grad{vars_gradient_laplacian(bl,FT)}(gradM),
                  bctype)
  divergence_penalty!(nf, bl, div_penalty, nM, gradM, gradP)
end

abstract type GradNumericalFlux end
struct CentralHyperDiffusiveFlux <: GradNumericalFlux end

function numerical_flux_hyperdiffusive!(::CentralHyperDiffusiveFlux, bl::BalanceLaw,
                                        HVF, nM, lapM, QM, aM, lapP, QP, aP, t)
  FT = eltype(lapM)
  @inbounds begin
    ndim = 3
    ngradlapstate = num_gradient_laplacian(bl,FT)
    n_Δdiff = similar(HVF, Size(ndim, ngradlapstate))
    @unroll for j = 1:ngradlapstate
      @unroll for i = 1:ndim
        n_Δdiff[i, j] = nM[i] * (lapM[j] + lapP[j]) / 2
      end
    end
    hyperdiffusive!(bl, Vars{vars_hyperdiffusive(bl,FT)}(HVF),
                    Grad{vars_gradient_laplacian(bl,FT)}(n_Δdiff),
                    Vars{vars_state(bl,FT)}(QM),
                    Vars{vars_aux(bl,FT)}(aM),
                    t)
  end
end

function numerical_boundary_flux_hyperdiffusive!(nf::CentralHyperDiffusiveFlux, bl::BalanceLaw,
                                                 HVF, nM, lapM, QM, aM, lapP, QP, aP,
                                                 bctype, t)
  FT = eltype(lapM)
  boundary_state!(nf, bl,
                  Vars{vars_state(bl,FT)}(QP),
                  Vars{vars_aux(bl,FT)}(aP),
                  Vars{vars_gradient_laplacian(bl,FT)}(lapP),
                  nM,
                  Vars{vars_state(bl,FT)}(QM),
                  Vars{vars_aux(bl,FT)}(aM),
                  Vars{vars_gradient_laplacian(bl,FT)}(lapM),
                  bctype, t)
  numerical_flux_hyperdiffusive!(nf, bl, HVF,
                                 nM, lapM, QM, aM, lapP, QP, aP, t)
end

end
