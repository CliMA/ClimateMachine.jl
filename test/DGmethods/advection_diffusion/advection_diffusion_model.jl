using StaticArrays
using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw,
                        vars_aux, vars_state, vars_gradient, vars_diffusive,
                        flux_nondiffusive!, flux_diffusive!, source!,
                        gradvariables!, diffusive!,
                        init_aux!, init_state!,
                        boundary_state!, wavespeed, LocalGeometry,
                        num_state, num_gradient
using CLIMA.DGmethods.NumericalFluxes: NumericalFluxNonDiffusive,
                                       NumericalFluxDiffusive,
                                       NumericalFluxGradient
import CLIMA.DGmethods.NumericalFluxes: boundary_flux_diffusive!

abstract type AdvectionDiffusionProblem end
struct AdvectionDiffusion{dim, P, fluxBC} <: BalanceLaw
  problem::P
  function AdvectionDiffusion{dim}(problem::P
                                  ) where {dim, P <: AdvectionDiffusionProblem}
    new{dim, P, false}(problem)
  end
  function AdvectionDiffusion{dim, fluxBC}(problem::P
                                  ) where {dim, P <: AdvectionDiffusionProblem,
                                           fluxBC}
    new{dim, P, fluxBC}(problem)
  end
end

# Stored in the aux state are:
#   `coord` coordinate points (needed for BCs)
#   `u` advection velocity
#   `D` Diffusion tensor
vars_aux(::AdvectionDiffusion, FT) = @vars(coord::SVector{3, FT},
                                           u::SVector{3, FT},
                                           D::SMatrix{3, 3, FT, 9})
#
# Density is only state
vars_state(::AdvectionDiffusion, FT) = @vars(ρ::FT)

# Take the gradient of density
vars_gradient(::AdvectionDiffusion, FT) = @vars(ρ::FT)

# The DG auxiliary variable: D ∇ρ
vars_diffusive(::AdvectionDiffusion, FT) = @vars(σ::SVector{3,FT})

"""
    flux_nondiffusive!(m::AdvectionDiffusion, flux::Grad, state::Vars,
                       aux::Vars, t::Real)

Computes non-diffusive flux `F` in:

```
∂ρ
-- = - ∇ • (u ρ - σ) = - ∇ • F
∂t
```
Where

 - `u` is the advection velocity
 - `ρ` is the advected quantity
 - `σ` is DG auxiliary variable (`σ = D ∇ ρ` with D being the diffusion tensor)
"""
function flux_nondiffusive!(m::AdvectionDiffusion, flux::Grad, state::Vars,
                            aux::Vars, t::Real)
  ρ = state.ρ
  u = aux.u
  flux.ρ += u * ρ
end

"""
flux_diffusive!(m::AdvectionDiffusion, flux::Grad, auxDG::Vars)

Computes diffusive flux `F` in:

```
∂ρ
-- = - ∇ • (u ρ - σ) = - ∇ • F
∂t
```
Where

 - `u` is the advection velocity
 - `ρ` is the advected quantity
 - `σ` is DG auxiliary variable (`σ = D ∇ ρ` with D being the diffusion tensor)
"""
function flux_diffusive!(m::AdvectionDiffusion, flux::Grad, auxDG::Vars)
  σ = auxDG.σ
  flux.ρ += -σ
end
flux_diffusive!(m::AdvectionDiffusion, flux::Grad, state::Vars, auxDG::Vars,
                auxHDG::Vars, aux::Vars, t::Real) = flux_diffusive!(m, flux, auxDG)

"""
    gradvariables!(m::AdvectionDiffusion, transform::Vars, state::Vars,
                   aux::Vars, t::Real)

Set the variable to take the gradient of (`ρ` in this case)
"""
function gradvariables!(m::AdvectionDiffusion, transform::Vars, state::Vars,
                        aux::Vars, t::Real)
  transform.ρ = state.ρ
end

"""
    diffusive!(m::AdvectionDiffusion, transform::Vars, gradvars::Vars,
               aux::Vars)

Set the variable to take the gradient of (`ρ` in this case)
"""
function diffusive!(m::AdvectionDiffusion, auxDG::Vars, gradvars::Grad,
                    aux::Vars)
  ∇ρ = gradvars.ρ
  D = aux.D
  auxDG.σ = D * ∇ρ
end
diffusive!(m::AdvectionDiffusion, auxDG::Vars, gradvars::Grad,
           state::Vars, aux::Vars, t::Real) = diffusive!(m, auxDG, gradvars,
                                                         aux)

"""
    source!(m::AdvectionDiffusion, _...)

There is no source in the advection-diffusion model
"""
source!(m::AdvectionDiffusion, _...) = nothing

"""
    wavespeed(m::AdvectionDiffusion, nM, state::Vars, aux::Vars, t::Real)

Wavespeed with respect to vector `nM`
"""
function wavespeed(m::AdvectionDiffusion, nM, state::Vars, aux::Vars, t::Real)
  u = aux.u
  abs(dot(nM, u))
end

"""
    init_aux!(m::AdvectionDiffusion, aux::Vars, geom::LocalGeometry)

initialize the auxiliary state
"""
function init_aux!(m::AdvectionDiffusion, aux::Vars, geom::LocalGeometry)
  aux.coord = geom.coord
  init_velocity_diffusion!(m.problem, aux, geom)
end

function init_state!(m::AdvectionDiffusion, state::Vars, aux::Vars,
                     coords, t::Real)
  initial_condition!(m.problem, state, aux, coords, t)
end

Neumann_data!(problem, ∇state, aux, x, t) = nothing
Dirichlet_data!(problem, state, aux, x, t) = nothing

function boundary_state!(nf, m::AdvectionDiffusion, stateP::Vars, auxP::Vars,
                         nM, stateM::Vars, auxM::Vars, bctype, t, _...)
  if bctype == 1 # Dirichlet
    Dirichlet_data!(m.problem, stateP, auxP, auxP.coord, t)
  elseif bctype ∈ (2, 4) # Neumann
    stateP.ρ = stateM.ρ
  elseif bctype == 3 # zero Dirichlet
    stateP.ρ = 0
  end
end

function boundary_state!(nf::CentralNumericalFluxDiffusive,
    m::AdvectionDiffusion, 
    state⁺::Vars, diff⁺::Vars, aux⁺::Vars,
    n⁻::SVector,
    state⁻::Vars, diff⁻::Vars, aux⁻::Vars,
    bctype, t,
    _...)

  if bctype ∈ (1,3) # Dirchlet
    # Just use the minus side values since Dirchlet
    diff⁺.σ = diff⁻.σ
  elseif bctype == 2 # Neumann with data
    FT = eltype(diff⁺)
    ngrad = num_gradient(m, FT)
    ∇state = Grad{vars_gradient(m, FT)}(similar(parent(diff⁺), Size(3, ngrad)))
    # Get analytic gradient
    Neumann_data!(m.problem, ∇state, aux⁻, aux⁻.coord, t)
    diffusive!(m, diff⁺, ∇state, aux⁻)
    # compute the diffusive flux using the boundary state
  elseif bctype == 4 # zero Neumann
    FT = eltype(diff⁺)
    ngrad = num_gradient(m, FT)
    ∇state = Grad{vars_gradient(m, FT)}(similar(parent(diff⁺), Size(3, ngrad)))
    # Get analytic gradient
    ∇state.ρ = SVector{3, FT}(0, 0, 0)
    # convert to auxDG variables
    diffusive!(m, diff⁺, ∇state, aux⁻)
  end
  nothing
end

function boundary_flux_diffusive!(nf::CentralNumericalFluxDiffusive,
                                  m::AdvectionDiffusion{dim, P, true},
                                  F,
                                  state⁺, diff⁺, hyperdiff⁺, aux⁺, n⁻,
                                  state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                  bctype, t,
                                  _...) where {dim, P}

  # Default initialize flux to minus side
  if bctype ∈ (1,3) # Dirchlet
    # Just use the minus side values since Dirchlet
    flux_diffusive!(m, F, state⁻, diff⁻, hyperdiff⁻, aux⁻, t)
  elseif bctype == 2 # Neumann data
    FT = eltype(diff⁺)
    ngrad = num_gradient(m, FT)
    ∇state = Grad{vars_gradient(m, FT)}(similar(parent(diff⁺), Size(3, ngrad)))
    # Get analytic gradient
    Neumann_data!(m.problem, ∇state, aux⁻, aux⁻.coord, t)
    # get the diffusion coefficient
    D = aux⁻.D
    # exact the exact data
    ∇ρ = ∇state.ρ
    # set the flux
    F.ρ = - D * ∇ρ
  elseif bctype == 4 # Zero Neumann
    FT = eltype(diff⁺)
    F.ρ = SVector{3, FT}(0, 0, 0)
  end
  nothing
end
