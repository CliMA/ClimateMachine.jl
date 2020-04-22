using StaticArrays
using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw,
                        vars_aux, vars_state, vars_gradient, vars_diffusive,
                        flux_nondiffusive!, flux_diffusive!, source!,
                        gradvariables!, diffusive!,
                        init_aux!, init_state!,
                        boundary_state!, wavespeed, LocalGeometry,
                        vars_gradient_laplacian, vars_hyperdiffusive,
                        hyperdiffusive!
using CLIMA.DGmethods.NumericalFluxes: NumericalFluxNonDiffusive,
                                       NumericalFluxDiffusive

abstract type HyperDiffusionProblem end
struct HyperDiffusion{dim, P} <: BalanceLaw
  problem::P
  function HyperDiffusion{dim}(problem::P) where {dim, P <: HyperDiffusionProblem}
    new{dim, P}(problem)
  end
end

vars_aux(::HyperDiffusion, FT) = @vars(D::SMatrix{3, 3, FT, 9})
#
# Density is only state
vars_state(::HyperDiffusion, FT) = @vars(ρ::FT)

# Take the gradient of density
vars_gradient(::HyperDiffusion, FT) = @vars(ρ::FT)
# Take the gradient of laplacian of density
vars_gradient_laplacian(::HyperDiffusion, FT) = @vars(ρ::FT)

vars_diffusive(::HyperDiffusion, FT) = @vars()
# The hyperdiffusion DG auxiliary variable: D ∇ Δρ
vars_hyperdiffusive(::HyperDiffusion, FT) = @vars(σ::SVector{3,FT})

function flux_nondiffusive!(m::HyperDiffusion, flux::Grad, state::Vars,
                            aux::Vars, t::Real)
end

"""
    flux_diffusive!(m::HyperDiffusion, flux::Grad, state::Vars,
                     auxDG::Vars, auxHDG::Vars, aux::Vars, t::Real)

Computes diffusive flux `F` in:

```
∂ρ
-- = - ∇ • (σ) = - ∇ • F
∂t
```
Where

 - `σ` is hyperdiffusion DG auxiliary variable (`σ = D ∇ Δρ` with D being the hyperdiffusion tensor)
"""
function flux_diffusive!(m::HyperDiffusion, flux::Grad, state::Vars,
                         auxDG::Vars, auxHDG::Vars, aux::Vars, t::Real)
  σ = auxHDG.σ
  flux.ρ += σ
end

"""
    gradvariables!(m::HyperDiffusion, transform::Vars, state::Vars,
                   aux::Vars, t::Real)

Set the variable to take the gradient of (`ρ` in this case)
"""
function gradvariables!(m::HyperDiffusion, transform::Vars, state::Vars,
                        aux::Vars, t::Real)
  transform.ρ = state.ρ
end

diffusive!(m::HyperDiffusion, _...) = nothing
function hyperdiffusive!(m::HyperDiffusion, auxHDG::Vars, gradvars::Grad,
                         state::Vars, aux::Vars, t::Real)
  ∇Δρ = gradvars.ρ
  D = aux.D
  auxHDG.σ = D * ∇Δρ
end

"""
    source!(m::HyperDiffusion, _...)

There is no source in the hyperdiffusion model
"""
source!(m::HyperDiffusion, _...) = nothing

"""
    init_aux!(m::HyperDiffusion, aux::Vars, geom::LocalGeometry)

initialize the auxiliary state
"""
function init_aux!(m::HyperDiffusion, aux::Vars, geom::LocalGeometry)
  init_hyperdiffusion_tensor!(m.problem, aux, geom)
end

function init_state!(m::HyperDiffusion, state::Vars, aux::Vars,
                     coords, t::Real)
  initial_condition!(m.problem, state, aux, coords, t)
end

boundary_state!(nf, ::HyperDiffusion, _...) = nothing
