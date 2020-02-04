using StaticArrays
using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw,
                        vars_aux, vars_state, vars_gradient, vars_diffusive,
                        flux_nondiffusive!, flux_diffusive!, source!,
                        gradvariables!, diffusive!,
                        init_aux!, init_state!,
                        boundary_state!, wavespeed, LocalGeometry

"""
    SoilModel

Computes diffusive flux `F` in:

∂y / ∂t = ∇ ⋅ Flux + Source

```
∂(ρcT)   ∂      ∂T
------ = --(λ * --)
  ∂t     ∂z     ∂z
```
where

 - `ρ` is the density of the soil (kg/m³)
 - `c` is the soil heat capacity (J/(kg K))
 - `λ` is the thermal conductivity (W/(m K))

To write this in the form
```
∂Y
-- + ∇⋅F(Y,t) = 0
∂t
```
we write `Y = ρcT` and `F(Y, t) = -λ ∇T`.

"""
Base.@kwdef struct SoilModel <: BalanceLaw
  ρc::Float64 = 2.49e6
  λ::Float64  = 2.42
  initialT::Float64 = 10.0
  meanT₀::Float64 = 15.0
  A₀::Float64 = 5.0
end

# Stored in the aux state are:
#   `coord` coordinate points (needed for BCs)
#   `u` advection velocity
#   `D` Diffusion tensor
vars_aux(::SoilModel, FT) = @vars(z::FT, ρc::FT, λ::FT) # stored dg.auxstate
vars_state(::SoilModel, FT) = @vars(ρcT::FT) # stored in Q
vars_gradient(::SoilModel, FT) = @vars(T::FT) # not stored
vars_diffusive(::SoilModel, FT) = @vars(∇T::SVector{3,FT}) # stored in dg.diffstate

function gradvariables!(m::SoilModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.T = state.ρcT / (m.ρc)
end
function diffusive!(m::SoilModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  diffusive.∇T = ∇transform.T
end
function flux_nondiffusive!(m::SoilModel, flux::Grad, state::Vars, aux::Vars, t::Real)
end
function flux_diffusive!(m::SoilModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  flux.ρcT -= m.λ * diffusive.∇T
end

function source!(m::SoilModel, state::Vars, _...)
  # state.ρcT += d(ρcT)/dt
end

#=
function wavespeed(m::SoilModel, nM, state::Vars, aux::Vars, t::Real)
  zero(eltype(state))
end
=#

function init_aux!(m::SoilModel, aux::Vars, geom::LocalGeometry)
  aux.z = geom.coord[3]
end

function init_state!(m::SoilModel, state::Vars, aux::Vars, coords, t::Real)
  state.ρcT = m.ρc * m.initialT
end


function boundary_state!(nf, m::SoilModel, state⁺::Vars, aux⁺::Vars,
                         nM, state⁻::Vars, aux⁻::Vars, bctype, t, _...)
  if bctype == 1
    # surface
    t_hours = t/(60*60)
    T = m.meanT₀ + m.A₀ * sinpi(2*(t_hours-8)/24)
    state⁺.ρcT = m.ρc * T
  elseif bctype == 2
    # bottom
    nothing
  end
end

function boundary_state!(nf, m::SoilModel, state⁺::Vars, diff⁺::Vars,
                         aux⁺::Vars, nM, state⁻::Vars, diff⁻::Vars, aux⁻::Vars,
                         bctype, t, _...)
  if bctype == 1
    # surface
    t_hours = t/(60*60)
    T = m.meanT₀ + m.A₀ * sinpi(2*(t_hours-8)/24)
    state⁺.ρcT = m.ρc * T
  elseif bctype == 2
    # bottom
    diff⁺.∇T = -diff⁻.∇T
  end
end
