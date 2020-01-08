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
  ρ::Float64
  c::Float64
  λ::Float64
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
  transform.T = state.ρcT / (m.ρ * m.c)
end
function diffusive!(m::SoilModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  diffusive.∂T∂z = ∇transform.T[2]
end
function flux_nondiffusive!(m::SoilModel, flux::Grad, state::Vars, aux::Vars, t::Real)
end
function flux_diffusive!(m::SoilModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  flux.ρcT += SVector(0, m.λ * diffusive.∂T∂z, 0)
end

source!(m::SoilModel, _...) = nothing

#=
function wavespeed(m::SoilModel, nM, state::Vars, aux::Vars, t::Real)
  zero(eltype(state))
end
=#

function init_aux!(m::SoilModel, aux::Vars, geom::LocalGeometry)
  aux.z = geom.coord[2]
end

function init_state!(m::SoilModel, state::Vars, aux::Vars, coords, t::Real)
  state.ρcT = 10.0
end

# Neumann boundary conditions
function boundary_state!(nf, m::SoilModel, stateP::Vars, auxP::Vars,
                         nM, stateM::Vars, auxM::Vars, bctype, t, _...)
  nothing
end

function boundary_state!(nf, m::SoilModel, stateP::Vars, diffP::Vars,
                         auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars,
                         bctype, t, _...)
  if bctype == 1
    # top
    diffP.∂T∂z = -diffM.∂T∂z + 1.0
  elseif bctype == 2
    diffP.∂T∂z = -diffM.∂T∂z
  end
end
