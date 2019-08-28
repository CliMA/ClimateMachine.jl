module Atmos

export AtmosModel,
  ConstantViscosityWithDivergence, SmagorinskyLilly,
  DryModel, EquilMoist, NonEquilMoist,
  NoRadiation,
  NoFluxBC, InitStateBC, RayleighBenardBC,
  FlatOrientation, SphericalOrientation

using LinearAlgebra, StaticArrays
using ..VariableTemplates
using ..MoistThermodynamics
using ..PlanetParameters
using CLIMA.SubgridScaleParameters

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient, vars_diffusive,
  flux!, source!, wavespeed, boundarycondition!, gradvariables!, diffusive!,
  init_aux!, init_state!, update_aux!, LocalGeometry, lengthscale, resolutionmetric

"""
    AtmosModel <: BalanceLaw

A `BalanceLaw` for atmosphere modelling.

# Usage

    AtmosModel(turbulence, moisture, radiation, source, boundarycondition, init_state)

"""
struct AtmosModel{O,T,M,P,R,S,BC,IS} <: BalanceLaw
  orientation::O
  turbulence::T
  moisture::M
  precipitation::P
  radiation::R
  # TODO: a better mechanism than functions.
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
    precipitation::vars_state(m.precipitation,T)
    radiation::vars_state(m.radiation,T)
  end
end
function vars_gradient(m::AtmosModel, T)
  @vars begin
    u::SVector{3,T}
    h_tot::T
    turbulence::vars_gradient(m.turbulence,T)
    moisture::vars_gradient(m.moisture,T)
    precipitation::vars_gradient(m.precipitation,T)
    radiation::vars_gradient(m.radiation,T)
  end
end
function vars_diffusive(m::AtmosModel, T)
  @vars begin
    ρτ::SVector{6,T}
    ρd_h_tot::SVector{3,T}
    turbulence::vars_diffusive(m.turbulence,T)
    moisture::vars_diffusive(m.moisture,T)
    precipitation::vars_diffusive(m.precipitation,T)
    radiation::vars_diffusive(m.radiation,T)
  end
end
function vars_aux(m::AtmosModel, T)
  @vars begin
    coord::SVector{3,T}
    sponge::T
    orientation::vars_aux(m.orientation, T)
    turbulence::vars_aux(m.turbulence,T)
    moisture::vars_aux(m.moisture,T)
    precipitation::vars_aux(m.precipitation,T)
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
  flux_nondiffusive!(m, flux, state, diffusive, aux, t)
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

function diffusive!(m::AtmosModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  ∇u = ∇transform.u

  # strain rate tensor
  # TODO: we use an SVector for this, but should define a "SymmetricSMatrix"?
  S = SVector(∇u[1,1],
              ∇u[2,2],
              ∇u[3,3],
              (∇u[1,2] + ∇u[2,1])/2,
              (∇u[1,3] + ∇u[3,1])/2,
              (∇u[2,3] + ∇u[3,2])/2)

  # kinematic viscosity tensor
  ρν = dynamic_viscosity_tensor(m.turbulence, S, state, aux, t)

  # momentum flux tensor
  diffusive.ρτ = scaled_momentum_flux_tensor(m.turbulence, ρν, S)

  D_T = 440.0
  # diffusive flux of total enthalpy
  diffusive.ρd_h_tot = (-D_T) .* ∇transform.h_tot
  # diffusivity of moisture components
  diffusive!(m.moisture, diffusive, ∇transform, state, aux, t, ρν, D_T)
  diffusive!(m.precipitation, diffusive, ∇transform, state, aux, t, ρν, D_T)
end

function flux_nondiffusive!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end

function flux_diffusive!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  u = ρinv * state.ρu

  # diffusive
  ρτ11, ρτ22, ρτ33, ρτ12, ρτ13, ρτ23 = diffusive.ρτ
  ρτ = SMatrix{3,3}(ρτ11, ρτ12, ρτ13,
                    ρτ12, ρτ22, ρτ23,
                    ρτ13, ρτ23, ρτ33)
  flux.ρu += ρτ
  flux.ρe += ρτ*u

  flux_diffusive!(m.moisture, flux, state, diffusive, aux, t)
  flux_diffusive!(m.precipitation, flux, state, diffusive, aux, t)
end

function wavespeed(m::AtmosModel, nM, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  # TODO: use soundspeed
  return abs(dot(nM, u)) + 300.0
  # return abs(dot(nM, u)) + soundspeed(m.moisture, state, aux)
end

function gradvariables!(m::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1 / state.ρ
  transform.u = ρinv * state.ρu

  gradvariables!(m.moisture, transform, state, aux, t)
  gradvariables!(m.turbulence, transform, state, aux, t)
end


const (xmin, xmax) = (-30000,30000)
const (ymin, ymax) = (0,  5000)
const (zmin, zmax) = (0, 24000)

function diffusive!(m::AtmosModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  ∇u = ∇transform.u

  # strain rate tensor
  # TODO: we use an SVector for this, but should define a "SymmetricSMatrix"?
  S = SVector(∇u[1,1],
              ∇u[2,2],
              ∇u[3,3],
              (∇u[1,2] + ∇u[2,1])/2,
              (∇u[1,3] + ∇u[3,1])/2,
              (∇u[2,3] + ∇u[3,2])/2)

  # kinematic viscosity tensor
  ρν = dynamic_viscosity_tensor(m.turbulence, S, state, diffusive, aux, t)

  # momentum flux tensor
  diffusive.ρτ = scaled_momentum_flux_tensor(m.turbulence, ρν, S)

  # diffusivity of moisture components
  diffusive!(m.moisture, diffusive, ∇transform, state, aux, t, ρν)
  # diffusion terms required for SGS turbulence computations
  diffusive!(m.turbulence, diffusive, ∇transform, state, aux, t, ρν)
end
>>>>>>> upstream/master

function update_aux!(m::AtmosModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  update_aux!(m.moisture, state, diffusive, aux, t)
end

include("turbulence.jl")
include("moisture.jl")
include("precipitation.jl")
include("radiation.jl")
include("orientation.jl")
include("force.jl")

"""
    init_sponge!(m::AtmosModel, aux::Vars, geom::LocalGeometry)

Compute sponge in aux state, for momentum equation.
"""
function init_sponge!(m::AtmosModel, aux::Vars, geom::LocalGeometry)
  DT = eltype(aux)
  #Sponge
  csleft  = DT(0)
  csright = DT(0)
  csfront = DT(0)
  csback  = DT(0)
  ctop    = DT(0)

  cs_left_right = DT(0.0)
  cs_front_back = DT(0.0)
  ct            = DT(0.9)

  #BEGIN  User modification on domain parameters.
  #Only change the first index of brickrange if your axis are
  #oriented differently:
  #x, y, z = aux[_a_x], aux[_a_y], aux[_a_z]
  #TODO z is the vertical coordinate
  #
  domain_left  = xmin
  domain_right = xmax

  domain_front = ymin
  domain_back  = ymax

  domain_bott  = zmin
  domain_top   = zmax

  #END User modification on domain parameters.
  z = aux.orientation.Φ/grav

  # Define Sponge Boundaries
  xc       = DT(0.5) * (domain_right + domain_left)
  yc       = DT(0.5) * (domain_back  + domain_front)
  zc       = DT(0.5) * (domain_top   + domain_bott)

  sponge_type = 2
  if sponge_type == 1

      bc_zscale   = DT(7000.0)
      top_sponge  = DT(0.85) * domain_top
      zd          = domain_top - bc_zscale
      xsponger    = domain_right - DT(0.15) * (domain_right - xc)
      xspongel    = domain_left  + DT(0.15) * (xc - domain_left)
      ysponger    = domain_back  - DT(0.15) * (domain_back - yc)
      yspongel    = domain_front + DT(0.15) * (yc - domain_front)

      #x left and right
      #xsl
      if x <= xspongel
          csleft = cs_left_right * (sinpi(1/2 * (x - xspongel)/(domain_left - xspongel)))^4
      end
      #xsr
      if x >= xsponger
          csright = cs_left_right * (sinpi(1/2 * (x - xsponger)/(domain_right - xsponger)))^4
      end
      #y left and right
      #ysl
      if y <= yspongel
          csfront = cs_front_back * (sinpi(1/2 * (y - yspongel)/(domain_front - yspongel)))^4
      end
      #ysr
      if y >= ysponger
          csback = cs_front_back * (sinpi(1/2 * (y - ysponger)/(domain_back - ysponger)))^4
      end

      #Vertical sponge:
      if z >= top_sponge
          ctop = ct * (sinpi(1/2 * (z - top_sponge)/(domain_top - top_sponge)))^4
      end

  elseif sponge_type == 2


      alpha_coe = DT(0.5)
      bc_zscale = DT(7500.0)
      zd        = domain_top - bc_zscale
      xsponger  = domain_right - DT(0.15) * (domain_right - xc)
      xspongel  = domain_left  + DT(0.15) * (xc - domain_left)
      ysponger  = domain_back  - DT(0.15) * (domain_back - yc)
      yspongel  = domain_front + DT(0.15) * (yc - domain_front)

      #
      # top damping
      # first layer: damp lee waves
      #
      ctop = DT(0.0)
      ct   = DT(0.5)
      if z >= zd
          zid = (z - zd)/(domain_top - zd) # normalized coordinate
          if zid >= 0.0 && zid <= 0.5
              abstaud = alpha_coe*(DT(1) - cos(zid*pi))
          else
              abstaud = alpha_coe*(DT(1) + cos((zid - 1/2)*pi) )
          end
          ctop = ct*abstaud
      end

  end #sponge_type

  beta  = DT(1) - (DT(1) - ctop) #*(1.0 - csleft)*(1.0 - csright)*(1.0 - csfront)*(1.0 - csback)
  aux.sponge  = min(beta, DT(1))
end

# TODO: figure out a nice way to handle this
function init_aux!(m::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.coord = geom.coord
  init_aux!(m.orientation, aux, geom)
  init_aux!(m.turbulence, aux, geom)
  init_sponge!(m, aux, geom)
end

function source_geopot!(m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  DT = eltype(state)
  state.ρu -= SVector(DT(0),DT(0), state.ρ * grav)
end

function source_sponge!(m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  beta     = aux.sponge
  state.ρu -= beta .* state.ρu
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
  source_geopot!(m, source, state, aux, t)
  source_sponge!(m, source, state, aux, t)
  source_microphysics!(m.moisture, source, state, aux, t)
  source_microphysics!(m.precipitation, source, state, aux, t)
end


# TODO: figure out a better interface for this.
# at the moment we can just pass a function, but we should do something better
# need to figure out how subcomponents will interact.
function boundarycondition!(m::AtmosModel, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t)
  m.boundarycondition(stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
end

abstract type BoundaryCondition
end

"""
    NoFluxBC <: BoundaryCondition

Set the momentum at the boundary to be zero.
"""
struct NoFluxBC <: BoundaryCondition
end

function boundarycondition!(m::AtmosModel{O,T,M,P,R,S,BC,IS}, stateP::Vars, diffP::Vars, auxP::Vars,
    nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t) where {O,T,M,P,R,S,BC <: NoFluxBC,IS}
  # TOFIX: This is not strictly a no-flux BC, and needs to be fixed
  DT = eltype(stateP)
  stateP.ρu -= 2 .* dot(stateM.ρu, nM) .* collect(nM)
  stateP.ρe = stateM.ρe
  boundarycondition_moisture!(m.moisture, stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
  boundarycondition_moisture!(m.precipitation, stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
end

"""
    InitStateBC <: BoundaryCondition

Set the value at the boundary to match the `init_state!` function. This is mainly useful for cases where the problem has an explicit solution.
"""
struct InitStateBC <: BoundaryCondition
end
function boundarycondition!(m::AtmosModel{O,T,M,P,R,S,BC,IS}, stateP::Vars, diffP::Vars, auxP::Vars,
    nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t) where {O,T,M,P,R,S,BC <: InitStateBC,IS}
  init_state!(m, stateP, auxP, auxP.coord, t)
end

function init_state!(m::AtmosModel, state::Vars, aux::Vars, coords, t, args)
  m.init_state(state, aux, coords, t, args...)
end

end # module
