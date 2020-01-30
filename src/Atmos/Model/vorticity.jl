export VorticityModel

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
                        reverse_indefinite_stack_integral!, num_state
import ..DGmethods.NumericalFluxes: boundary_state!, Rusanov,
                                    CentralNumericalFluxGradient,
                                    CentralNumericalFluxDiffusive,
                                    boundary_flux_diffusive!

"""
    VorticityModel <: BalanceLaw

A `BalanceLaw` for computing the vorticity diagnostic.

# Usage

    ....

"""
struct VorticityModel{M} <: BalanceLaw
  atmos::M
end

function vars_state(m::VorticityModel, FT)
  @vars begin
    ω::SVector{3,FT}
  end
end

vars_gradient(m::VorticityModel, FT) = @vars()
vars_diffusive(m::VorticityModel, FT) = @vars()
vars_aux(m::VorticityModel, FT) = vars_state(m.atmos, FT)
init_aux!(m::VorticityModel, aux::Vars, geom::LocalGeometry) = nothing

function init_state!(m::VorticityModel, state::Vars, aux::Vars, coords, t, args...)
  state.ω = SVector(NaN, NaN, NaN)
end

"""
    flux_nondiffusive!(m::VorticityModel, flux::Grad, state::Vars, aux::Vars,
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
@inline function flux_nondiffusive!(m::VorticityModel, flux::Grad, state::Vars,
                                    aux::Vars, t::Real)
  ρ = aux.ρ
  ρinv = 1/ρ
  ρu = aux.ρu
  u = ρinv * ρu
  @inbounds begin
    flux.ω = @SMatrix [ 0     u[3] -u[2];
                       -u[3]  0     u[1];
                        u[2] -u[1]  0    ]
  end
end

function gradvariables!(atmos::VorticityModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  nothing
end

function diffusive!(atmos::VorticityModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  nothing
end

@inline function flux_diffusive!(atmos::VorticityModel, flux::Grad, state::Vars,
                                 diffusive::Vars, aux::Vars, t::Real)
  nothing
end

"""
    source!(m::VorticityModel, source::Vars, state::Vars, aux::Vars, t::Real)
Computes flux `S(Y)` in:
```
∂Y
-- = - ∇ • F + S(Y)
∂t
```
"""
function source!(m::VorticityModel, source::Vars, state::Vars, aux::Vars, t::Real)
  nothing
end
