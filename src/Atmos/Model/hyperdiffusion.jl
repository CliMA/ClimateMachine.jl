#### Hyperdiffusion Model Functions
using DocStringExtensions
using LinearAlgebra
export HyperDiffusion, NoHyperDiffusion, StandardHyperDiffusion

abstract type HyperDiffusion end
vars_state_conservative(::HyperDiffusion, FT) = @vars()
vars_state_auxiliary(::HyperDiffusion, FT) = @vars()
vars_state_gradient(::HyperDiffusion, FT) = @vars()
vars_gradient_laplacian(::HyperDiffusion, FT) = @vars()
vars_state_gradient_flux(::HyperDiffusion, FT) = @vars()
vars_hyperdiffusive(::HyperDiffusion, FT) = @vars()
function atmos_init_aux!(
    ::HyperDiffusion,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) end
function atmos_nodal_update_auxiliary_state!(
    ::HyperDiffusion,
    ::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function compute_gradient_argument!(
    ::HyperDiffusion,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function transform_post_gradient_laplacian!(
    h::HyperDiffusion,
    hyperdiffusive::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function flux_second_order!(
    h::HyperDiffusion,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
) end
function compute_gradient_flux!(
    h::HyperDiffusion,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end

"""
  NoHyperDiffusion <: HyperDiffusion
Defines a default hyperdiffusion model with zero diffusive fluxes.
"""
struct NoHyperDiffusion <: HyperDiffusion end

"""
  StandardHyperDiffusion{FT} <: HyperDiffusion
Horizontal hyperdiffusion methods for application in GCM and LES settings
Timescales are prescribed by the user while the diffusion coefficient is
computed as a function of the grid lengthscale.
"""
struct StandardHyperDiffusion{FT} <: HyperDiffusion
    τ_timescale::FT
end

vars_state_auxiliary(::StandardHyperDiffusion, FT) = @vars(Δ::FT)
vars_state_gradient(::StandardHyperDiffusion, FT) =
    @vars(u::SVector{3, FT}, h_tot::FT)
vars_gradient_laplacian(::StandardHyperDiffusion, FT) =
    @vars(u::SVector{3, FT}, h_tot::FT)
vars_hyperdiffusive(::StandardHyperDiffusion, FT) =
    @vars(ν∇³u::SMatrix{3, 3, FT, 9}, ν∇³h_tot::SVector{3, FT})

function atmos_init_aux!(
    ::StandardHyperDiffusion,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.hyperdiffusion.Δ = lengthscale(geom)
end

function compute_gradient_argument!(
    h::StandardHyperDiffusion,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.hyperdiffusion.u = ρinv * state.ρu
    transform.hyperdiffusion.h_tot = transform.h_tot
end

function transform_post_gradient_laplacian!(
    h::StandardHyperDiffusion,
    hyperdiffusive::Vars,
    hypertransform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ∇Δu = hypertransform.hyperdiffusion.u
    ∇Δh_tot = hypertransform.hyperdiffusion.h_tot
    # Unpack
    τ_timescale = h.τ_timescale
    # Compute hyperviscosity coefficient
    ν₄ = (aux.hyperdiffusion.Δ / 2)^4 / 2 / τ_timescale
    hyperdiffusive.hyperdiffusion.ν∇³u = ν₄ * ∇Δu
    hyperdiffusive.hyperdiffusion.ν∇³h_tot = ν₄ * ∇Δh_tot
end

function flux_second_order!(
    h::StandardHyperDiffusion,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρu += state.ρ * hyperdiffusive.hyperdiffusion.ν∇³u
    flux.ρe += state.ρ * hyperdiffusive.hyperdiffusion.ν∇³h_tot
end
