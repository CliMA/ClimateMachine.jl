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
                    num_gradient, gradvariables!,
                    num_gradient_laplacian, vars_gradient_laplacian,
                    vars_hyperdiffusive, hyperdiffusive!

"""
    NumericalFluxGradient

Any `P <: NumericalFluxGradient` should define methods for:

   numerical_flux_gradient!(gnf::P, bl::BalanceLaw, diffF, n⁻, Q⁻, Qdiff⁻, Qaux⁻, Q⁺,
                            Qdiff⁺, Qaux⁺, t)
   numerical_boundary_flux_gradient!(gnf::P, bl::BalanceLaw, l_Qvisc, n⁻, l_G⁻, l_Q⁻,
                                     l_aux⁻, l_G⁺, l_Q⁺, l_aux⁺, bctype, t)

"""
abstract type NumericalFluxGradient end

"""
    CentralNumericalFluxGradient <: NumericalFluxGradient

"""
struct CentralNumericalFluxGradient <: NumericalFluxGradient end

function numerical_flux_gradient!(::CentralNumericalFluxGradient, bl::BalanceLaw,
                                  G::MMatrix, n::SVector,
                                  transform⁻::Vars{T}, state⁻::Vars{S},
                                  aux⁻::Vars{A}, transform⁺::Vars{T},
                                  state⁺::Vars{S}, aux⁺::Vars{A},
                                  t) where {T,S,A}

  G .= n .* (parent(transform⁺) .+ parent(transform⁻))' ./ 2
end

function numerical_boundary_flux_gradient!(nf::CentralNumericalFluxGradient,
                                           bl::BalanceLaw,
                                           G::MMatrix, n::SVector,
                                           transform⁻::Vars{T}, state⁻::Vars{S},
                                           aux⁻::Vars{A}, transform⁺::Vars{T},
                                           state⁺::Vars{S}, aux⁺::Vars{A},
                                           bctype, t, state1⁻::Vars{S},
                                           aux1⁻::Vars{A}) where {D,T,S,A}
  boundary_state!(nf, bl, state⁺, aux⁺, n, state⁻, aux⁻,
                  bctype, t, state1⁻, aux1⁻)

  gradvariables!(bl, transform⁺, state⁺, aux⁺, t)
  G .= n .* parent(transform⁺)'
end

"""
    NumericalFluxNonDiffusive

Any `N <: NumericalFluxNonDiffusive` should define the a method for

    numerical_flux_nondiffusive!(nf::N, bl::BalanceLaw, F, n⁻, Q⁻, Qaux⁻, Q⁺,
                                 Qaux⁺, t)

where
- `F` is the numerical flux array
- `n⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_nondiffusive!(nf::N, bl::BalanceLaw, F, n⁻, Q⁻,
                                          Qaux⁻, Q⁺, Qaux⁺, bctype, t)

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

    numerical_flux_diffusive!(nf::N, bl::BalanceLaw, F, n⁻, Q⁻, Qdiff⁻, Qaux⁻, Q⁺,
                              Qdiff⁺, Qaux⁺, t)

where
- `F` is the numerical flux array
- `n⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `Qdiff⁻`/`Qdiff⁺` are the minus/positive diffusive state arrays
- `Qdiff⁻`/`Qdiff⁺` are the minus/positive auxiliary state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_diffusive!(nf::N, bl::BalanceLaw, F, n⁻, Q⁻, Qdiff⁻,
                                       Qaux⁻, Q⁺, Qdiff⁺, Qaux⁺, bctype, t)

"""
abstract type NumericalFluxDiffusive end

function numerical_flux_diffusive! end

function numerical_boundary_flux_diffusive! end

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

  Fᵀn .+= (F⁻ + F⁺)' * (n⁻/2)
end

abstract type DivNumericalPenalty end
struct CentralDivPenalty <: DivNumericalPenalty end

function divergence_penalty!(::CentralDivPenalty, bl::BalanceLaw,
                             div_penalty::Vars{GL}, n::SVector,
                             grad⁻::Grad{GL}, grad⁺::Grad{GL}) where {GL}
  parent(div_penalty) .= (parent(grad⁺) .- parent(grad⁻))' * (n/2)
end

function divergence_boundary_penalty!(nf::CentralDivPenalty, bl::BalanceLaw,
                                      div_penalty::Vars{GL}, n::SVector,
                                      grad⁻::Grad{GL}, grad⁺::Grad{GL}, bctype) where {GL}
  boundary_state!(nf, bl, grad⁺, n, grad⁻, bctype)
  divergence_penalty!(nf, bl, div_penalty, n, grad⁻, grad⁺)
end

abstract type GradNumericalFlux end
struct CentralHyperDiffusiveFlux <: GradNumericalFlux end

function numerical_flux_hyperdiffusive!(::CentralHyperDiffusiveFlux, bl::BalanceLaw,
                                        hyperdiff::Vars{HD}, n::SVector,
                                        lap⁻::Vars{GL}, state⁻::Vars{S}, aux⁻::Vars{A},
                                        lap⁺::Vars{GL}, state⁺::Vars{S}, aux⁺::Vars{A},
                                        t) where {HD, GL, S, A}
  G = n .* (parent(lap⁻) .+ parent(lap⁺))' ./ 2
  hyperdiffusive!(bl, hyperdiff, Grad{GL}(G), state⁻, aux⁻, t)
end

function numerical_boundary_flux_hyperdiffusive!(nf::CentralHyperDiffusiveFlux, bl::BalanceLaw,
                                                 hyperdiff::Vars{HD}, n::SVector,
                                                 lap⁻::Vars{GL}, state⁻::Vars{S}, aux⁻::Vars{A},
                                                 lap⁺::Vars{GL}, state⁺::Vars{S}, aux⁺::Vars{A},
                                                 bctype, t) where {HD, GL, S, A}
  boundary_state!(nf, bl, state⁺, aux⁺, lap⁺, n, state⁻, aux⁻, lap⁻, bctype, t)
  numerical_flux_hyperdiffusive!(nf, bl, hyperdiff, n,
                                 lap⁻, state⁻, aux⁻, lap⁺, state⁺, aux⁺, t)
end

numerical_boundary_flux_diffusive!(nf::CentralNumericalFluxDiffusive,
    bl::BalanceLaw, fluxᵀn::Vars{S}, n⁻::SVector,
    state⁻::Vars{S}, diff⁻::Vars{D}, hyperdiff⁻::Vars{HD}, aux⁻::Vars{A},
    state⁺::Vars{S}, diff⁺::Vars{D}, hyperdiff⁺::Vars{HD}, aux⁺::Vars{A},
    bctype, t,
    state1⁻::Vars{S}, diff1⁻::Vars{D}, aux1⁻::Vars{A}) where {S,D,HD,A} =
  normal_boundary_flux_diffusive!(nf, bl, fluxᵀn, n⁻,
                                  state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                  state⁺, diff⁺, hyperdiff⁺, aux⁺,
                                  bctype, t,
                                  state1⁻, diff1⁻, aux1⁻)

function normal_boundary_flux_diffusive!(nf,
                                         bl::BalanceLaw, fluxᵀn::Vars{S}, n⁻,
                                         state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                         state⁺, diff⁺, hyperdiff⁺, aux⁺,
                                         bctype, t,
                                         state1⁻, diff1⁻, aux1⁻) where {S}
  FT = eltype(fluxᵀn)
  nstate = num_state(bl,FT)
  Fᵀn = parent(fluxᵀn)

  F = similar(Fᵀn, Size(3, nstate))
  fill!(F, -zero(FT))
  boundary_flux_diffusive!(nf, bl,
                           Grad{S}(F),
                           state⁺, diff⁺, hyperdiff⁺, aux⁺,
                           n⁻,
                           state⁻, diff⁻, hyperdiff⁻, aux⁻,
                           bctype, t,
                           state1⁻, diff1⁻, aux1⁻)

  Fᵀn .+= F' * n⁻
end

# This is the function that my be overloaded for flux-based BCs
function boundary_flux_diffusive!(nf::NumericalFluxDiffusive, bl,
                                  F,
                                  state⁺, diff⁺, hyperdiff⁺, aux⁺, n⁻,
                                  state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                  bctype, t,
                                  state1⁻, diff1⁻, aux1⁻)
  boundary_state!(nf, bl,
                  state⁺, diff⁺, aux⁺,
                  n⁻,
                  state⁻, diff⁻, aux⁻,
                  bctype, t,
                  state1⁻, diff1⁻, aux1⁻)
  flux_diffusive!(bl, F, state⁺, diff⁺, hyperdiff⁺, aux⁺, t)
end

end
