using StaticArrays
using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw,
                        vars_aux, vars_state, vars_gradient, vars_diffusive,
                        flux_nondiffusive!, flux_diffusive!, source!,
                        gradvariables!, diffusive!,
                        init_aux!, init_state!,
                        boundary_state!, wavespeed, LocalGeometry
using CLIMA.DGmethods.NumericalFluxes: NumericalFluxNonDiffusive,
                                       NumericalFluxDiffusive,
                                       GradNumericalPenalty

"""
    SoilModel

Computes diffusive flux `F` in:

```math
∂(ρcT)   ∂      ∂T
------ = --(λ * --)
  ∂t     ∂z     ∂z
```
where

 - `ρ` is the density of the soil (kg/m³)
 - `c` is the soil heat capacity (J/(kg K))
 - `λ` is the thermal conductivity (W/(m K))
"""
struct SoilModel <: BalanceLaw
  surfaceT::Float64
  initialT::Float64
end

# Stored in the aux state are:
#   `coord` coordinate points (needed for BCs)
#   `u` advection velocity
#   `D` Diffusion tensor
vars_aux(::SoilModel, FT) = @vars(z::FT, ρ::FT, c::FT, λ::FT) # stored dg.auxstate
vars_state(::SoilModel, FT) = @vars(ρcT::FT) # stored in Q
vars_gradient(::SoilModel, FT) = @vars(T::FT) # not stored
vars_diffusive(::SoilModel, FT) = @vars(∂T∂z::FT) # stored in dg.diffstate

function gradvariables!(m::SoilModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.T = state.ρcT / (aux.ρ * aux.c)
  #@show aux.z transform.T 
end
function diffusive!(m::SoilModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
 # @show aux.z ∇transform.T[2]
  diffusive.∂T∂z = ∇transform.T[2]
end
function flux_nondiffusive!(m::SoilModel, flux::Grad, state::Vars, aux::Vars, t::Real)
end
function flux_diffusive!(m::SoilModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  flux.ρcT += SVector(0, aux.λ * diffusive.∂T∂z, 0)
end

source!(m::SoilModel, _...) = nothing

function wavespeed(m::SoilModel, nM, state::Vars, aux::Vars, t::Real)
  zero(eltype(state))
end

function init_aux!(m::SoilModel, aux::Vars, geom::LocalGeometry)
  aux.z = geom.coord[2]
  aux.ρ = 1
  aux.c = 1
  aux.λ = 1
end

function init_state!(m::SoilModel, state::Vars, aux::Vars, coords, t::Real)
  w = exp(-aux.z)
  state.ρcT = w * m.surfaceT + (1-w) * m.initialT
end

function boundary_state!(nf, m::SoilModel, stateP::Vars, auxP::Vars,
                         nM, stateM::Vars, auxM::Vars, bctype, t, _...)
  if bctype == 1
    #boundary_state_Dirichlet!(nf, m, stateP, auxP, nM, stateM, auxM, t)
  # elseif bctype == 2
   # TODO: boundary_state_Neumann!(nf, m, stateP, auxP, nM, stateM, auxM, t)
  end
end

function boundary_state!(nf, m::SoilModel, stateP::Vars, diffP::Vars,
                         auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars,
                         bctype, t, _...)
  if bctype == 1
    #boundary_state_Dirichlet!(nf, m, stateP, diffP, auxP, nM, stateM, diffM, auxM, t)
  # elseif bctype == 2
  #   boundary_state_Neumann!(nf, m, stateP, auxP, nM, stateM, auxM, t)
  end
end

###
### Dirchlet Boundary Condition
###
function boundary_state_Dirichlet!(::NumericalFluxNonDiffusive,
                                   m::SoilModel,
                                   stateP, auxP, nM, stateM, auxM, t)
  # Set the plus side to the exact boundary data
  stateP.ρcT = auxP.z == 0 ? m.surfaceT : m.initialT
end
function boundary_state_Dirichlet!(::GradNumericalPenalty,
                                   m::SoilModel,
                                   stateP, auxP, nM, stateM, auxM, t)
  # Set the plus side so that after average the numerical flux is the boundary
  # data
  stateP.ρcT = m.surfaceT
  stateP.ρcT = 2stateP.ρcT - stateM.ρcT
end

# Do nothing in this case since the plus-side is the minus side
function boundary_state_Dirichlet!(::NumericalFluxDiffusive, ::SoilModel, _...)
  nothing
end