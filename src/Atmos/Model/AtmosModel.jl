module Atmos

export AtmosModel,
  
  ConstantViscosityWithDivergence, SmagorinskyLilly, 
  DryModel, EquilMoist,
  NoRadiation,
  NoFluxBC, InitStateBC, DYCOMS_BC,
  FlatOrientation, SphericalOrientation

using LinearAlgebra, StaticArrays
using ..VariableTemplates
using ..MoistThermodynamics
using ..PlanetParameters

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient, vars_diffusive,
  flux!, source!, wavespeed, boundarycondition!, gradvariables!, diffusive!,
  init_aux!, init_state!, update_aux!, LocalGeometry, lengthscale

"""
    AtmosModel <: BalanceLaw

A `BalanceLaw` for atmosphere modelling.

# Usage

    AtmosModel(turbulence, moisture, radiation, source, boundarycondition, init_state)

"""
struct AtmosModel{O,T,M,R,S,BC,IS} <: BalanceLaw
  orientation::O
  turbulence::T
  moisture::M
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
    radiation::vars_state(m.radiation,T)
  end
end
function vars_gradient(m::AtmosModel, T)
  @vars begin
    u::SVector{3,T}
    turbulence::vars_gradient(m.turbulence,T)
    moisture::vars_gradient(m.moisture,T)
    radiation::vars_gradient(m.radiation,T)
  end
end
function vars_diffusive(m::AtmosModel, T)
  @vars begin
    ρτ::SVector{6,T}
    turbulence::vars_diffusive(m.turbulence,T)
    moisture::vars_diffusive(m.moisture,T)
    radiation::vars_diffusive(m.radiation,T)
  end
end
function vars_aux(m::AtmosModel, T)
  @vars begin
    coord::SVector{3,T}
    orientation::vars_aux(m.orientation, T)
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
end

function wavespeed(m::AtmosModel, nM, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  return abs(dot(nM, u)) + soundspeed(m.moisture, state, aux)
end

function gradvariables!(m::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1 / state.ρ
  transform.u = ρinv * state.ρu

  gradvariables!(m.moisture, transform, state, aux, t)
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

  # diffusivity of moisture components
  diffusive!(m.moisture, diffusive, ∇transform, state, aux, t, ρν)
end

function update_aux!(m::AtmosModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  update_aux!(m.moisture, state, diffusive, aux, t)
end

include("turbulence.jl")
include("moisture.jl")
include("radiation.jl")
include("orientation.jl")

# TODO: figure out a nice way to handle this
function init_aux!(m::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.coord = geom.coord
  init_aux!(m.orientation, aux, geom)
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
  m.source(source, state, aux, t)
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
function boundarycondition!(m::AtmosModel{O,T,M,R,S,BC,IS}, stateP::Vars, diffP::Vars, auxP::Vars,
    nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t) where {O,T,M,R,S,BC <: NoFluxBC,IS}
  stateP.ρu -= 2 * dot(stateM.ρu, nM) * nM
end

"""
    InitStateBC <: BoundaryCondition

Set the value at the boundary to match the `init_state!` function. This is mainly useful for cases where the problem has an explicit solution.
"""
struct InitStateBC <: BoundaryCondition
end
function boundarycondition!(m::AtmosModel{O,T,M,R,S,BC,IS}, stateP::Vars, diffP::Vars, auxP::Vars,
    nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t) where {O,T,M,R,S,BC <: InitStateBC,IS}
  init_state!(m, stateP, auxP, auxP.coord, t)
end

function init_state!(m::AtmosModel, state::Vars, aux::Vars, coord, t)
  m.init_state(state, aux, coord, t)
end

"""
  DYCOMS_BC <: BoundaryCondition
  Prescribes boundary conditions for Dynamics of Marine Stratocumulus Case
"""
struct DYCOMS_BC <: BoundaryCondition
  C_drag
  LHF
  SHF
end
function boundarycondition!(bl::AtmosModel{O,T,M,R,S,BC,IS}, stateP::Vars, diffP::Vars, auxP::Vars,
    nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t, state1::Vars, diff1::Vars, aux1::Vars) where {O,T,M,R,S,BC <: DYCOMS_BC,IS}
    # stateM is the 𝐘⁻ state while stateP is the 𝐘⁺ state at an interface. 
    # at the boundaries the ⁻, minus side states are the interior values
    # state1 is 𝐘 at the first interior nodes relative to the bottom wall 
    
    # Get values from minus-side state
    ρM = stateM.ρ 
    UM, VM, WM = stateM.ρu
    EM = stateM.ρe
    QTM = stateM.moisture.ρq_tot
    uM, vM, wM  = UM/ρM, VM/ρM, WM/ρM
    q_totM = QTM/ρM
    UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM
    
    # Assign reflection wall boundaries (top wall)
    stateP.ρu = SVector(UM - 2 * nM[1] * UnM, 
                        VM - 2 * nM[2] * UnM,
                        WM - 2 * nM[3] * UnM)

    # Assign scalar values at the boundaries 
    stateP.ρ = ρM
    stateP.moisture.ρq_tot = QTM
    # Assign diffusive fluxes at boundaries
    diffP = diffM
    xvert = auxM.coord[3]
    
    if bctype == 5
      # ------------------------------------------------------------------------
      # (<var>_FN) First node values (First interior node from bottom wall)
      # ------------------------------------------------------------------------
      z_FN             = aux1.coord[3]
      ρ_FN             = state1.ρ
      U_FN, V_FN, W_FN = state1.ρu
      E_FN             = state1.ρe
      u_FN, v_FN, w_FN = U_FN/ρ_FN, V_FN/ρ_FN, W_FN/ρ_FN
      windspeed_FN     = sqrt(u_FN^2 + v_FN^2 + w_FN^2)
      q_tot_FN         = state1.moisture.ρq_tot / ρ_FN
      e_int_FN         = E_FN/ρ_FN - 0.5*windspeed_FN^2 - grav*z_FN
      TS_FN            = PhaseEquil(e_int_FN, q_tot_FN, ρ_FN) 
      T_FN             = air_temperature(TS_FN)
      q_vap_FN         = q_tot_FN - PhasePartition(TS_FN).liq
      # --------------------------
      # Bottom boundary quantities 
      # --------------------------
      zM          = auxM.coord[3] 
      q_totM      = QTM/ρM
      windspeed   = sqrt(uM^2 + vM^2 + wM^2)
      e_intM      = EM/ρM - 0.5*windspeed^2 - grav*zM
      TSM         = PhaseEquil(e_intM, q_totM, ρM) 
      q_vapM      = q_totM - PhasePartition(TSM).liq
      TM          = air_temperature(TSM)
      # ----------------------------------------------------------
      # Extract components of diffusive momentum flux (minus-side)
      # ----------------------------------------------------------
      ρτ11, ρτ22, ρτ33, ρτ12, ρτ13, ρτ23 = diffM.ρτ
      
      # Case specific for flat bottom topography, normal vector is n⃗ = k⃗ = [0, 0, 1]ᵀ
      # A more general implementation requires (n⃗ ⋅ ∇A) to be defined where A is replaced by the appropriate flux terms
      C_drag = bl.boundarycondition.C_drag
      ρτ13P  = -ρM * C_drag * windspeed_FN * u_FN 
      ρτ23P  = -ρM * C_drag * windspeed_FN * v_FN 
      # Assign diffusive momentum and moisture fluxes
      # (i.e. ρ𝚻 terms)  
      diffP.ρτ = SVector(0,0,0,0, ρτ13P, ρτ23P)
      diffP.moisture.ρd_q_tot  = SVector(diffM.moisture.ρd_q_tot[1],
                                         diffM.moisture.ρd_q_tot[2],
                                         bl.boundarycondition.LHF/(LH_v0))

      # Assign diffusive enthalpy flux (i.e. ρ(𝐉 + 𝐃) terms) 
      diffP.moisture.ρ_SGS_enthalpyflux  = SVector(diffM.moisture.ρ_SGS_enthalpyflux[1],
                                                   diffM.moisture.ρ_SGS_enthalpyflux[2],
                                                   bl.boundarycondition.LHF + bl.boundarycondition.SHF)
  end
end

end # module
