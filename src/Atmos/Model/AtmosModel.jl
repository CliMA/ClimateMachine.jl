module Atmos

export AtmosModel,
  NoViscosity, ConstantViscosityWithDivergence, SmagorinskyLilly,
  DryModel, EquilMoist,
  NoRadiation, StevensRadiation,
  Gravity, RayleighSponge, Subsidence, GeostrophicForcing,
  PeriodicBC, NoFluxBC, InitStateBC, DYCOMS_BC,
  FlatOrientation, SphericalOrientation

using LinearAlgebra, StaticArrays
using ..VariableTemplates
using ..MoistThermodynamics
using ..PlanetParameters
import ..MoistThermodynamics: internal_energy
using ..SubgridScaleParameters
using GPUifyLoops
using ..MPIStateArrays: MPIStateArray

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, vars_integrals, flux_nondiffusive!,
                        flux_diffusive!, source!, wavespeed, boundary_state!,
                        gradvariables!, diffusive!, init_aux!, init_state!,
                        update_aux!, integrate_aux!, LocalGeometry, lengthscale,
                        resolutionmetric, DGModel, num_integrals,
                        nodal_update_aux!, indefinite_stack_integral!,
                        reverse_indefinite_stack_integral!
using ..DGmethods.NumericalFluxes

"""
    AtmosModel <: BalanceLaw

A `BalanceLaw` for atmosphere modeling.

# Usage

    AtmosModel(orientation, ref_state, turbulence, moisture, radiation, source,
               boundarycondition, init_state)

"""
struct AtmosModel{O,RS,T,M,R,S,BC,IS} <: BalanceLaw
  orientation::O
  ref_state::RS
  turbulence::T
  moisture::M
  radiation::R
  source::S
  # TODO: Probably want to have different bc for state and diffusion...
  boundarycondition::BC
  init_state::IS
end

# defined here so that the main variables and flux definitions
# can be found in this file since some of these are specialized for NoViscosity
abstract type TurbulenceClosure end
struct NoViscosity <: TurbulenceClosure end

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

vars_gradient(m::AtmosModel, T) = vars_gradient(m, T, m.turbulence)
function vars_gradient(m::AtmosModel, T, ::TurbulenceClosure)
  @vars begin
    u::SVector{3,T}
    turbulence::vars_gradient(m.turbulence,T)
    moisture::vars_gradient(m.moisture,T)
    radiation::vars_gradient(m.radiation,T)
  end
end
vars_gradient(m::AtmosModel, T, ::NoViscosity) = @vars()

vars_diffusive(m::AtmosModel, T) = vars_diffusive(m, T, m.turbulence)
function vars_diffusive(m::AtmosModel, T, ::TurbulenceClosure)
  @vars begin
    ρτ::SHermitianCompact{3,T,6}
    turbulence::vars_diffusive(m.turbulence,T)
    moisture::vars_diffusive(m.moisture,T)
    radiation::vars_diffusive(m.radiation,T)
  end
end
vars_diffusive(m::AtmosModel, T, ::NoViscosity) = @vars()

function vars_aux(m::AtmosModel, T)
  @vars begin
    ∫dz::vars_integrals(m, T)
    ∫dnz::vars_integrals(m, T)
    coord::SVector{3,T}
    orientation::vars_aux(m.orientation, T)
    ref_state::vars_aux(m.ref_state,T)
    turbulence::vars_aux(m.turbulence,T)
    moisture::vars_aux(m.moisture,T)
    radiation::vars_aux(m.radiation,T)
  end
end
function vars_integrals(m::AtmosModel,T)
  @vars begin
    radiation::vars_integrals(m.radiation,T)
  end
end

"""
    flux_nondiffusive!(m::AtmosModel, flux::Grad, state::Vars, aux::Vars,
                       t::Real)

Computes flux non-diffusive flux portion of `F` in:

```
∂Y
-- = - ∇ • (F_{adv} + F_{press} + F_{nondiff} + F_{diff}) + S(Y)
∂t
```
Where

 - `F_{adv}`      Advective flux             ; see [`flux_advective!`]@ref()
 - `F_{press}`    Pressure flux              ; see [`flux_pressure!`]@ref()
 - `F_{diff}`     Fluxes that state gradients; see [`flux_diffusive!`]@ref()
"""
@inline function flux_nondiffusive!(m::AtmosModel, flux::Grad, state::Vars,
                                    aux::Vars, t::Real)
  flux_advective!(m, flux, state, aux, t)
  flux_pressure!(m, flux, state, aux, t)
  flux_radiation!(m, flux, state, aux, t)
end

@inline function flux_advective!(m::AtmosModel, flux::Grad, state::Vars,
                                 aux::Vars, t::Real)
  # preflux
  ρinv = 1/state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  # advective terms
  flux.ρ   = ρu
  flux.ρu  = ρu .* u'
  flux.ρe  = u * state.ρe
end

@inline function flux_pressure!(m::AtmosModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  # preflux
  ρinv = 1/state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  p = pressure(m.moisture, m.orientation, state, aux)
  # pressure terms
  flux.ρu += p*I
  flux.ρe += u*p
end

@inline function flux_radiation!(m::AtmosModel, flux::Grad, state::Vars,
                                 aux::Vars, t::Real)
  flux_radiation!(m.radiation, flux, state, aux,t)
end

flux_diffusive!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) =
  flux_diffusive!(m, flux, state, diffusive, aux, t, m.turbulence)
@inline function flux_diffusive!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real,
                                 ::TurbulenceClosure)
  ρinv = 1/state.ρ
  u = ρinv * state.ρu

  # diffusive
  ρτ = diffusive.ρτ
  flux.ρu += ρτ
  flux.ρe += ρτ*u
  flux_diffusive!(m.moisture, flux, state, diffusive, aux, t)
end
flux_diffusive!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real,
                ::NoViscosity) = nothing

@inline function wavespeed(m::AtmosModel, nM, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  return abs(dot(nM, u)) + soundspeed(m.moisture, m.orientation, state, aux)
end

gradvariables!(m::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real) =
  gradvariables!(m::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real, m.turbulence)
function gradvariables!(m::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real, ::TurbulenceClosure)
  ρinv = 1 / state.ρ
  transform.u = ρinv * state.ρu

  gradvariables!(m.moisture, m, transform, state, aux, t)
  gradvariables!(m.turbulence, transform, state, aux, t)
end
gradvariables!(m::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real, ::NoViscosity) = nothing

function symmetrize(X::StaticArray{Tuple{3,3}})
  SHermitianCompact(SVector(X[1,1], (X[2,1] + X[1,2])/2, (X[3,1] + X[1,3])/2, X[2,2], (X[3,2] + X[2,3])/2, X[3,3]))
end

diffusive!(m::AtmosModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real) =
  diffusive!(m::AtmosModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, m.turbulence)
function diffusive!(m::AtmosModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real,
                    ::TurbulenceClosure)
  ∇u = ∇transform.u
  # strain rate tensor
  S = symmetrize(∇u)
  # kinematic viscosity tensor
  ρν = dynamic_viscosity_tensor(m.turbulence, S, state, diffusive, aux, t)
  # momentum flux tensor
  diffusive.ρτ = scaled_momentum_flux_tensor(m.turbulence, ρν, S)
  # diffusivity of moisture components
  diffusive!(m.moisture, diffusive, ∇transform, state, aux, t, ρν, inv_Pr_turb)
  # diffusion terms required for SGS turbulence computations
  diffusive!(m.turbulence, diffusive, ∇transform, state, aux, t, ρν)
end
diffusive!(m::AtmosModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real,
           ::NoViscosity) = nothing

function update_aux!(dg::DGModel, m::AtmosModel, Q::MPIStateArray,
                     auxstate::MPIStateArray, t::Real)
  DFloat = eltype(Q)
  if num_integrals(m, DFloat) > 0
    indefinite_stack_integral!(dg, m, Q, auxstate, t)
    reverse_indefinite_stack_integral!(dg, m, Q, auxstate, t)
  end

  nodal_update_aux!(atmos_nodal_update_aux!, dg, m, Q, auxstate, t)
end

function atmos_nodal_update_aux!(m::AtmosModel, state::Vars, aux::Vars, t::Real)
  atmos_nodal_update_aux!(m.moisture, m, state, aux, t)
  atmos_nodal_update_aux!(m.radiation, m, state, aux, t)
  atmos_nodal_update_aux!(m.turbulence, m, state, aux, t)
end

function integrate_aux!(m::AtmosModel, integ::Vars, state::Vars, aux::Vars)
  integrate_aux!(m.radiation, integ, state, aux)
end

include("orientation.jl")
include("ref_state.jl")
include("turbulence.jl")
include("moisture.jl")
include("radiation.jl")
include("source.jl")
include("boundaryconditions.jl")

# TODO: figure out a nice way to handle this
function init_aux!(m::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.coord = geom.coord
  atmos_init_aux!(m.orientation, m, aux, geom)
  atmos_init_aux!(m.ref_state, m, aux, geom)
  atmos_init_aux!(m.turbulence, m, aux, geom)
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

boundary_state!(nf, m::AtmosModel, x...) =
  atmos_boundary_state!(nf, m.boundarycondition, m, x...)

# FIXME: This is probably not right....
boundary_state!(::CentralGradPenalty, bl::AtmosModel, _...) = nothing

function init_state!(m::AtmosModel, state::Vars, aux::Vars, coords, t)
  m.init_state(state, aux, coords, t)
end

end # module
