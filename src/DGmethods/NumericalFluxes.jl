module NumericalFluxes

export NumericalFluxNonDiffusive, NumericalFluxDiffusive, NumericalFluxGradient,
       Rusanov, CentralNumericalFluxGradient, CentralNumericalFluxDiffusive,
       CentralNumericalFluxNonDiffusive

using StaticArrays, LinearAlgebra
using GPUifyLoops: @unroll
using CLIMA.VariableTemplates
import ..DGmethods: BalanceLaw, Grad, Vars, vars_state, vars_diffusive,
                    vars_aux, vars_gradient, boundary_state!, wavespeed,
                    flux_nondiffusive!, flux_diffusive!, diffusive!, num_state,
                    num_gradient, gradvariables!


"""
    NumericalFluxGradient

Any `P <: NumericalFluxGradient` should define methods for:

   numerical_flux_gradient!(gnf::P, bl::BalanceLaw, diffF, nM, QM, QdiffM, QauxM, QP,
                            QdiffP, QauxP, t)
   numerical_boundary_flux_gradient!(gnf::P, bl::BalanceLaw, l_Qvisc, nM, l_GM, l_QM,
                                     l_auxM, l_GP, l_QP, l_auxP, bctype, t)

"""
abstract type NumericalFluxGradient end

function numerical_flux_gradient! end
function numerical_boundary_flux_gradient! end

"""
    CentralNumericalFluxGradient <: NumericalFluxGradient

"""
struct CentralNumericalFluxGradient <: NumericalFluxGradient end

function numerical_flux_gradient!(::CentralNumericalFluxGradient, bl::BalanceLaw,
                                  diff::Vars{D}, n::SVector,
                                  transform⁻::Vars{T}, state⁻::Vars{S},
                                  aux⁻::Vars{A}, transform⁺::Vars{T},
                                  state⁺::Vars{S}, aux⁺::Vars{A},
                                  t) where {D,T,S,A}

  G = n .* (parent(transform⁺) .+ parent(transform⁻))' ./ 2
  diffusive!(bl, diff, Grad{T}(G), state⁻, aux⁻, t)
end

function numerical_boundary_flux_gradient!(nf::CentralNumericalFluxGradient,
                                           bl::BalanceLaw,
                                           diff_penalty::Vars{D}, n::SVector,
                                           transform⁻::Vars{T}, state⁻::Vars{S},
                                           aux⁻::Vars{A}, transform⁺::Vars{T},
                                           state⁺::Vars{S}, aux⁺::Vars{A},
                                           bctype, t, state1⁻::Vars{S},
                                           aux1⁻::Vars{A}) where {D,T,S,A}

  boundary_state!(nf, bl, state⁺, aux⁺, n, state⁻, aux⁻,
                  bctype, t, state1⁻, aux1⁻)

  gradvariables!(bl, transform⁺, state⁺, aux⁺, t)

  numerical_flux_gradient!(nf, bl, diff_penalty, n, transform⁻, state⁻, aux⁻,
                           transform⁺, state⁺, aux⁺, t)
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

function numerical_boundary_flux_diffusive! end
function boundary_flux_diffusive!(nf::NumericalFluxDiffusive, bl,
                                  F⁺, state⁺, diff⁺, aux⁺, n⁻,
                                  F⁻, state⁻, diff⁻, aux⁻,
                                  bctype, t,
                                  state1⁻, diff1⁻, aux1⁻)
  FT = eltype(F⁺)
  boundary_state!(nf, bl, state⁺, diff⁺, aux⁺, n⁻,
                  state⁻, diff⁻, aux⁻, bctype, t,
                  state1⁻, diff1⁻, aux1⁻)
  fill!(parent(F⁺), -zero(FT))
  flux_diffusive!(bl, F⁺, state⁺, diff⁺, aux⁺, t)
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
  bl::BalanceLaw, fluxᵀn::Vars{S}, n⁻::SVector,
  state⁻::Vars{S}, diff⁻::Vars{D}, aux⁻::Vars{A},
  state⁺::Vars{S}, diff⁺::Vars{D}, aux⁺::Vars{A}, t) where {S,D,A}

  FT = eltype(fluxᵀn)
  nstate = num_state(bl,FT)
  Fᵀn = parent(fluxᵀn)

  F⁻ = similar(Fᵀn, Size(3, nstate))
  fill!(F⁻, -zero(FT))
  flux_diffusive!(bl, Grad{S}(F⁻), state⁻, diff⁻, aux⁻, t)

  F⁺ = similar(Fᵀn, Size(3, nstate))
  fill!(F⁺, -zero(FT))
  flux_diffusive!(bl, Grad{S}(F⁺), state⁺, diff⁺, aux⁺, t)

  Fᵀn .+= (F⁻ + F⁺)' * (n⁻/2)
end

function numerical_boundary_flux_diffusive!(nf::CentralNumericalFluxDiffusive,
    bl::BalanceLaw, fluxᵀn::Vars{S}, n⁻::SVector,
    state⁻::Vars{S}, diff⁻::Vars{D}, aux⁻::Vars{A},
    state⁺::Vars{S}, diff⁺::Vars{D}, aux⁺::Vars{A},
    bctype, t,
    state1⁻::Vars{S}, diff1⁻::Vars{D}, aux1⁻::Vars{A}) where {S,D,A}

  FT = eltype(fluxᵀn)
  nstate = num_state(bl,FT)
  Fᵀn = parent(fluxᵀn)

  F⁻ = similar(Fᵀn, Size(3, nstate))
  fill!(F⁻, -zero(FT))
  flux_diffusive!(bl, Grad{S}(F⁻), state⁻, diff⁻, aux⁻, t)

  F⁺ = similar(Fᵀn, Size(3, nstate))
  fill!(F⁺, -zero(FT))
  boundary_flux_diffusive!(nf, bl,
                           Grad{S}(F⁺), state⁺, diff⁺, aux⁺, n⁻,
                           Grad{S}(F⁻), state⁻, diff⁻, aux⁻,
                           bctype, t,
                           state1⁻, diff1⁻, aux1⁻)

  Fᵀn .+= (F⁻ + F⁺)' * (n⁻/2)
end

end
