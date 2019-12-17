using StaticArrays
using Unitful
using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw,
                        vars_aux, vars_state, vars_gradient, vars_diffusive,
                        space_unit, mass_unit, time_unit,
                        flux_nondiffusive!, flux_diffusive!, source!,
                        gradvariables!, diffusive!,
                        init_aux!, init_state!,
                        boundary_state!, wavespeed, LocalGeometry
using CLIMA.DGmethods.NumericalFluxes: NumericalFluxNonDiffusive,
                                       NumericalFluxDiffusive,
                                       GradNumericalPenalty

abstract type AdvectionDiffusionProblem end
struct AdvectionDiffusion{dim, P} <: BalanceLaw
  problem::P
  function AdvectionDiffusion{dim}(problem::P) where {dim, P <: AdvectionDiffusionProblem}
    new{dim, P}(problem)
  end
end

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
    flux_diffusive!(m::AdvectionDiffusion, flux::Grad, state::Vars,
                     auxDG::Vars, aux::Vars, t::Real)

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
function flux_diffusive!(m::AdvectionDiffusion, flux::Grad, state::Vars,
                         auxDG::Vars, aux::Vars, t::Real)
  σ = auxDG.σ
  flux.ρ += -σ
end

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
    diffusive!(m::AdvectionDiffusion, transform::Vars, state::Vars, aux::Vars,
               t::Real)

Set the variable to take the gradient of (`ρ` in this case)
"""
function diffusive!(m::AdvectionDiffusion, auxDG::Vars, gradvars::Grad,
                    state::Vars, aux::Vars, t::Real)
  ∇ρ = gradvars.ρ
  D = aux.D
  auxDG.σ = D * ∇ρ
end

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
  FT = get_T(eltype(coords))
  coords = units(FT, space_unit(m)).(coords)
  t = units(FT, time_unit(m))(t)
  initial_condition!(m.problem, state, aux, coords, t)
end

function boundary_state!(nf, m::AdvectionDiffusion, stateP::Vars, auxP::Vars,
                         nM, stateM::Vars, auxM::Vars, bctype, t, _...)
  if bctype == 1
    boundary_state_Dirichlet!(nf, m, stateP, auxP, nM, stateM, auxM, t)
  elseif bctype == 2
    # TODO: boundary_state_Neumann(nf, m, stateP, auxP, nM, stateM, auxM, t)
  elseif bctype == 3 # zero Dirichlet
    stateP.ρ = 0
  end
end

function boundary_state!(nf, m::AdvectionDiffusion, stateP::Vars, diffP::Vars,
                         auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars,
                         bctype, t, _...)
  if bctype == 1
    boundary_state_Dirichlet!(nf, m, stateP, diffP, auxP, nM, stateM, diffM,
                              auxM, t)
  elseif bctype == 2
    # boundary_state_Neumann(nf, m, stateP, auxP, nM, stateM, auxM, t)
  elseif bctype == 3 # zero Dirichlet
    stateP.ρ = - stateM.ρ
  end
end

###
### Dirchlet Boundary Condition
###
function boundary_state_Dirichlet!(::NumericalFluxNonDiffusive,
                                   m::AdvectionDiffusion,
                                   stateP, auxP, nM, stateM, auxM, t)
  # Set the plus side to the exact boundary data
  init_state!(m, stateP, auxP, auxP.coord, t)
end
function boundary_state_Dirichlet!(::GradNumericalPenalty,
                                   m::AdvectionDiffusion,
                                   stateP, auxP, nM, stateM, auxM, t)
  # Set the plus side sot that after average the numerical flux is the boundary
  # data
  init_state!(m, stateP, auxP, auxP.coord, t)
  stateP.ρ = 2stateP.ρ - stateM.ρ
end

# Do nothing in this case since the plus-side is the minus side
boundary_state_Dirichlet!(::NumericalFluxDiffusive, ::AdvectionDiffusion,
                          _...) = nothing
