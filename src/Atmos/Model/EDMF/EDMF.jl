#### Eddy-Diffusivity Mass-Flux (EDMF)

export EDMF
"""
    EDMF <: TurbulenceConvection

The EDMF turbulence convection model. This model extends `AtmosModel`
by adding `1+n_updrafts` prognostic equations, via additional `sub-domain`
to the grid mean equations (`AtmosModel` equations).
"""
struct EDMF{N,AF,MOM,E,M,TKE,SM,ML,ED,P,B} <: TurbulenceConvection{N}
  "Area fraction model"
  area_frac::AF
  "Momentum model"
  momentum::MOM
  "Energy model"
  energy::E
  "Moisture model"
  moisture::M
  "Turbulent kinetic energy model"
  tke::TKE
  "Surface model"
  surface::SM
  "Mixing length model"
  mix_len::ML
  "Entrainment-Detrainment model"
  entr_detr::ED
  "Pressure model"
  pressure::P
  "Buoyancy model"
  buoyancy::B
end

EDMF{N}(args...) where N = EDMF{N,typeof.(args)...}(args...)

function vars_state(edmf::EDMF{N}, T) where N
  @vars begin
    area_frac::vars_state(edmf.area_frac,T, Val(N))
    momentum::vars_state(edmf.momentum,T, Val(N))
    energy::vars_state(edmf.energy,T, Val(N))
    moisture::vars_state(edmf.moisture,T, Val(N))
    tke::vars_state(edmf.tke,T, Val(N))
  end
end
function vars_gradient(edmf::EDMF{N}, T) where N
  @vars begin
    area_frac::vars_gradient(edmf.area_frac,T)
    momentum::vars_gradient(edmf.momentum,T)
    energy::vars_gradient(edmf.energy,T)
    moisture::vars_gradient(edmf.moisture,T)
    tke::vars_gradient(edmf.tke,T)

    surface::vars_gradient(edmf.surface,T)
    mix_len::vars_gradient(edmf.mix_len,T, Val(N))
    entr_detr::vars_gradient(edmf.entr_detr,T, Val(N))
    pressure::vars_gradient(edmf.pressure,T)
    buoyancy::vars_gradient(edmf.buoyancy,T, Val(N))
  end
end
function vars_aux(edmf::EDMF{N}, T) where N
  @vars begin
    area_frac::vars_aux(edmf.area_frac,T)
    momentum::vars_aux(edmf.momentum,T)
    energy::vars_aux(edmf.energy,T)
    moisture::vars_aux(edmf.moisture,T)
    tke::vars_aux(edmf.tke,T)

    surface::vars_aux(edmf.surface,T)
    mix_len::vars_aux(edmf.mix_len,T, Val(N))
    entr_detr::vars_aux(edmf.entr_detr,T, Val(N))
    pressure::vars_aux(edmf.pressure,T)
    buoyancy::vars_aux(edmf.buoyancy,T, Val(N))
  end
end

function update_aux!(       ::TurbulenceConvection, state::Vars, diffusive::Vars, aux::Vars, t::Real);end
function gradvariables!(    ::TurbulenceConvection, transform::Vars, state::Vars, aux::Vars, t::Real);end
function flux!(             ::TurbulenceConvection, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real);end
function source!(           ::TurbulenceConvection, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real);end
function boundarycondition!(::TurbulenceConvection, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t);end

include("subdomain_index_helper.jl")
include("area_frac.jl")
include("momentum.jl")
include("energy.jl")
include("moisture.jl")
include("tke.jl")
include("surface.jl")
include("mixing_length.jl")
include("entr_detr.jl")
include("pressure.jl")
include("buoyancy.jl")

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
function flux!(edmf::EDMF, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  flux_advective!(edmf.area_frac, flux, state, diffusive, aux, t)
  flux_advective!(edmf.momentum , flux, state, diffusive, aux, t)
  flux_advective!(edmf.energy   , flux, state, diffusive, aux, t)
  flux_advective!(edmf.moisture , flux, state, diffusive, aux, t)
  flux_advective!(edmf.tke      , flux, state, diffusive, aux, t)

  flux_pressure!(edmf.momentum , flux, state, diffusive, aux, t)
  flux_pressure!(edmf.tke      , flux, state, diffusive, aux, t)

  flux_nondiffusive!(edmf.area_frac, flux, state, diffusive, aux, t)
  flux_nondiffusive!(edmf.momentum , flux, state, diffusive, aux, t)
  flux_nondiffusive!(edmf.energy   , flux, state, diffusive, aux, t)
  flux_nondiffusive!(edmf.moisture , flux, state, diffusive, aux, t)
  flux_nondiffusive!(edmf.tke      , flux, state, diffusive, aux, t)

  flux_diffusive!(edmf.area_frac, flux, state, diffusive, aux, t)
  flux_diffusive!(edmf.momentum , flux, state, diffusive, aux, t)
  flux_diffusive!(edmf.energy   , flux, state, diffusive, aux, t)
  flux_diffusive!(edmf.moisture , flux, state, diffusive, aux, t)
  flux_diffusive!(edmf.tke      , flux, state, diffusive, aux, t)
end

function gradvariables!(edmf::EDMF, transform::Vars, state::Vars, aux::Vars, t::Real)
  gradvariables!(edmf, edmf.area_frac, transform, state, aux, t)
  gradvariables!(edmf, edmf.momentum , transform, state, aux, t)
  gradvariables!(edmf, edmf.energy   , transform, state, aux, t)
  gradvariables!(edmf, edmf.moisture , transform, state, aux, t)
  gradvariables!(edmf, edmf.tke      , transform, state, aux, t)

  gradvariables!(edmf, edmf.surface  , transform, state, aux, t)
  gradvariables!(edmf, edmf.mix_len  , transform, state, aux, t)
  gradvariables!(edmf, edmf.entr_detr, transform, state, aux, t)
  gradvariables!(edmf, edmf.pressure , transform, state, aux, t)
  gradvariables!(edmf, edmf.buoyancy , transform, state, aux, t)
end

function update_aux!(edmf::EDMF, state::Vars, diffusive::Vars, ∇transform::Grad, aux::Vars, t::Real)
  diagnose_env!(edmf, edmf.area_frac, state, aux)
  diagnose_env!(edmf, edmf.momentum , state, aux)
  diagnose_env!(edmf, edmf.energy   , state, aux)
  diagnose_env!(edmf, edmf.moisture , state, aux)
  diagnose_env!(edmf, edmf.tke      , state, aux)

  update_aux!(edmf, edmf.area_frac, transform, state, aux, t)
  update_aux!(edmf, edmf.momentum , transform, state, aux, t)
  update_aux!(edmf, edmf.energy   , transform, state, aux, t)
  update_aux!(edmf, edmf.moisture , transform, state, aux, t)
  update_aux!(edmf, edmf.tke      , transform, state, aux, t)

  update_aux!(edmf, edmf.surface  , transform, state, aux, t)
  update_aux!(edmf, edmf.mix_len  , transform, state, aux, t)
  update_aux!(edmf, edmf.entr_detr, transform, state, aux, t)
  update_aux!(edmf, edmf.pressure , transform, state, aux, t)
  update_aux!(edmf, edmf.buoyancy , transform, state, aux, t)
end


"""
    source!(edmf::EDMF, source::Vars, state::Vars, aux::Vars, t::Real)

Computes flux `S(Y)` in:

```
∂Y
-- = - ∇ • F + S(Y)
∂t
```
"""
function source!(edmf::EDMF, source::Vars, state::Vars, aux::Vars, t::Real)
  source!(edmf, edmf.area_frac, source, state, aux, t)
  source!(edmf, edmf.momentum , source, state, aux, t)
  source!(edmf, edmf.energy   , source, state, aux, t)
  source!(edmf, edmf.moisture , source, state, aux, t)
  source!(edmf, edmf.tke      , source, state, aux, t)
end

function boundarycondition!(edmf::EDMF, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t)
  boundarycondition!(edmf, edmf.area_frac, stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
  boundarycondition!(edmf, edmf.momentum , stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
  boundarycondition!(edmf, edmf.energy   , stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
  boundarycondition!(edmf, edmf.moisture , stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
  boundarycondition!(edmf, edmf.tke      , stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
end

