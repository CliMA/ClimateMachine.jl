module Atmos

export AtmosModel,
  NoViscosity, ConstantViscosityWithDivergence, SmagorinskyLilly,
  DryModel, EquilMoist,
  NoRadiation,
  Gravity,
  NoFluxBC, InitStateBC, RayleighBenardBC,
  FlatOrientation, SphericalOrientation

using LinearAlgebra, StaticArrays
using ..VariableTemplates
using ..MoistThermodynamics
using ..PlanetParameters
import ..MoistThermodynamics: internal_energy
using ..SubgridScaleParameters

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient, vars_diffusive, vars_integrals,
  flux!, source!, wavespeed, boundarycondition!, gradvariables!, diffusive!,
  init_aux!, init_state!, update_aux!, integrate_aux!, LocalGeometry, lengthscale, resolutionmetric

"""
    AtmosModel <: BalanceLaw

A `BalanceLaw` for atmosphere modeling.

# Usage

    AtmosModel(orientation, ref_state, turbulence, moisture, radiation, source, boundarycondition, init_state)

"""
struct AtmosModel{O,RS,T,M,R,S,BC,IS} <: BalanceLaw
  orientation::O
  ref_state::RS
  turbulence::T
  moisture::M
  radiation::R
  source::S
  boundarycondition::BC
  init_state::IS
end

function vars_state(m::AtmosModel, T)
  @vars begin
    ρ::T
    ρu::SVector{3,T}
    ρe::T
    turbulence::vars_state(m.turbulence,T)
    moisture::vars_state(m.moisture,T)
    radiation::vars_state(m.radiation,T)
  end
end

function vars_gradient(m::AtmosModel, T)
  @vars begin
    turbulence::vars_gradient(m.turbulence,T)
    moisture::vars_gradient(m.moisture,T)
    radiation::vars_gradient(m.radiation,T)
  end
end

function vars_diffusive(m::AtmosModel, T)
  @vars begin
    turbulence::vars_diffusive(m.turbulence,T)
    moisture::vars_diffusive(m.moisture,T)
    radiation::vars_diffusive(m.radiation,T)
  end
end

function vars_integrals(m::AtmosModel, T)
  @vars begin
    radiation::vars_integrals(m.radiation, T)
  end
end

function vars_aux(m::AtmosModel, T)
  @vars begin
    ∫dz::vars_integrals(m,T)
    ∫dnz::vars_integrals(m,T)
    coord::SVector{3,T}
    orientation::vars_aux(m.orientation, T)
    ref_state::vars_aux(m.ref_state,T)
    turbulence::vars_aux(m.turbulence,T)
    moisture::vars_aux(m.moisture,T)
    radiation::vars_aux(m.radiation,T)
  end
end

"""
    flux!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)

Computes flux `F` in:

```
∂Y
-- = - ∇ • (F_{adv} + F_{press} + F_{nondiff} + F_{diff}) + S(Y)
∂t
```
Where

 - `F_{adv}`      Advective flux                                  , see [`flux_advective!`]@ref()    for this term
 - `F_{press}`    Pressure flux                                   , see [`flux_pressure!`]@ref()     for this term
 - `F_{nondiff}`  Fluxes that do *not* contain gradients          , see [`flux_nondiffusive!`]@ref() for this term
 - `F_{diff}`     Fluxes that contain gradients of state variables, see [`flux_diffusive!`]@ref()    for this term
"""
function flux!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  flux_advective!(m, flux, state, diffusive, aux, t)
  flux_pressure!(m, flux, state, diffusive, aux, t)
  # flux_nondiffusive!(m, flux, state, diffusive, aux, t)
  flux_diffusive!(m, flux, state, diffusive, aux, t)
end

function flux_advective!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  # preflux
  ρinv = 1/state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  # advective terms
  flux.ρ   = ρu
  flux.ρu  = ρu .* u'
  flux.ρe  = u * state.ρe
end

function flux_pressure!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  # preflux
  ρinv = 1/state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  p = pressure(m.moisture, state, aux)
  # pressure terms
  flux.ρu += p*I
  flux.ρe += u*p
end

# function flux_nondiffusive!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
# end

function flux_diffusive!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  flux_diffusive!(m.turbulence, flux, state, diffusive, aux, t)
  flux_diffusive!(m.moisture, flux, state, diffusive, aux, t)
end

function wavespeed(m::AtmosModel, nM, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  return abs(dot(nM, u)) + soundspeed(m.moisture, state, aux)
end

function gradvariables!(m::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  gradvariables!(m.turbulence, transform, state, aux, t)
  gradvariables!(m.moisture, transform, state, aux, t)
end

function diffusive!(m::AtmosModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  # diffusion terms required for SGS turbulence computations
  ρν = diffusive!(m.turbulence, diffusive, ∇transform, state, aux, t)
  # diffusivity of moisture components
  diffusive!(m.moisture, diffusive, ∇transform, state, aux, t, ρν)
end

function update_aux!(m::AtmosModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  update_aux!(m.moisture, state, diffusive, aux, t)
end

function integrate_aux!(m::AtmosModel, integ::Vars, state::Vars, aux::Vars)
  integrate_aux!(m.radiation, integ, state, aux)
end

include("ref_state.jl")
include("turbulence.jl")
include("moisture.jl")
include("radiation.jl")
include("orientation.jl")
include("source.jl")
include("boundaryconditions.jl")

# TODO: figure out a nice way to handle this
function init_aux!(m::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.coord = geom.coord
  init_aux!(m.orientation, aux, geom)
  init_aux!(m.ref_state, aux)
  init_aux!(m.turbulence, aux, geom)
end

"""
    source!(m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)

Computes flux `S(Y)` in:

```
∂Y
-- = - ∇ • F + S(Y)
∂t
```
"""
function source!(m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  atmos_source!(m.source, m, source, state, aux, t)
end

function boundarycondition!(m::AtmosModel, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t)
  atmos_boundarycondition!(m.boundarycondition, m, stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
end

function init_state!(m::AtmosModel, state::Vars, aux::Vars, coords, t)
  m.init_state(state, aux, coords, t)
end

end # module
