#### Hyperdiffusion Model Functions
using DocStringExtensions
using LinearAlgebra
using CLIMA.PlanetParameters
using CLIMA.SubgridScaleParameters
export HyperDiffusion, NoHyperDiffusion, HorizontalHyperDiffusion

abstract type HyperDiffusion end
vars_state(::HyperDiffusion, FT)                = @vars()
vars_aux(::HyperDiffusion, FT)                  = @vars()
vars_gradient(::HyperDiffusion, FT)             = @vars()
vars_diffusive(::HyperDiffusion, FT)            = @vars()
vars_hyperdiffusive(::HyperDiffusion, FT)       = @vars()
vars_gradient_laplacian(::HyperDiffusion, FT)   = @vars()
function atmos_init_aux!(::HyperDiffusion, ::AtmosModel, aux::Vars, geom::LocalGeometry) end
function atmos_nodal_update_aux!(::HyperDiffusion, ::AtmosModel, state::Vars, aux::Vars, t::Real) end
function gradvariables!(::HyperDiffusion, ::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real) end
function hyperdiffusive!(h::HyperDiffusion, hyperdiffusive::Vars, gradvars::Grad,
                         state::Vars, aux::Vars, t::Real) end
function flux_diffusive!(h::HyperDiffusion, flux::Grad, state::Vars,
                         diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real) end
function diffusive!(h::HyperDiffusion, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real) end

"""
  NoHyperDiffusion <: HyperDiffusion
Defines a default hyperdiffusion model with zero diffusive fluxes. 
"""
struct NoHyperDiffusion <: HyperDiffusion end

"""
  HorizontalHyperDiffusion{FT} <: HyperDiffusion
Horizontal hyperdiffusion methods for application in GCM and LES settings
Timescales are prescribed by the user while the diffusion coefficient is 
computed as a function of the grid lengthscale.
"""
struct HorizontalHyperDiffusion{FT} <: HyperDiffusion 
  τ_timescale::FT
end

vars_aux(::HorizontalHyperDiffusion, FT)                = @vars(Δ::FT)
vars_gradient_laplacian(::HorizontalHyperDiffusion, FT) = @vars(u::SVector{3,FT}, h_tot::FT)
vars_hyperdiffusive(::HorizontalHyperDiffusion, FT)     = @vars(ν∇³u_horz::SMatrix{3,3,FT,9}, ν∇³h_tot::SVector{3,FT})
diffusive!(::HorizontalHyperDiffusion, _...)            = nothing

function atmos_init_aux!(::HorizontalHyperDiffusion, ::AtmosModel, aux::Vars, geom::LocalGeometry) 
  aux.hyperdiffusion.Δ = lengthscale(geom)
end

function hyperdiffusive!(h::HorizontalHyperDiffusion, hyperdiffusive::Vars, hypertransform::Grad,
                         state::Vars, aux::Vars, t::Real)
  ∇Δu_horz = hypertransform.u
  ∇Δh_tot = hypertransform.h_tot
  τ_timescale = h.τ_timescale

  ν₄ = (aux.hyperdiffusion.Δ/2)^4 / 2 / τ_timescale
  hyperdiffusive.ν∇³u_horz = state.ρ * ν₄ * ∇Δu_horz
  hyperdiffusive.ν∇³h_tot  = state.ρ * ν₄ * ∇Δh_tot
end

function flux_nondiffusive!(h::HorizontalHyperDiffusion, flux::Grad, state::Vars, aux::Vars, t::Real) end

function flux_diffusive!(h::HorizontalHyperDiffusion, flux::Grad, state::Vars,
                         diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real) 
  flux.ρu += 1/state.ρ * hyperdiffusive.ν∇³u_horz
  flux.ρe += 1/state.ρ * hyperdiffusive.ν∇³h_tot
end
