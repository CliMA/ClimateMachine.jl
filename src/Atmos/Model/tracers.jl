# ## [Tracers](@id tracer-model) 
#
#md # !!! note
#md #       
#md #     Usage: Enable tracers using a keyword argument in the AtmosModel specification\
#md #     `tracers = NoTracer()`\
#md #     `tracers = NTracers{N, FT}(δ_χ)` where N is the number of tracers required.\
#md #     FT is the float-type and $\delta_{\chi}$ is an SVector of diffusivity scaling coefficients 
# 
# In `tracers.jl`, we define the equation sets governing
# tracer dynamics. Specifically, we address the the equations
# of tracer motion in conservation form, 
#
#
using DocStringExtensions
#
# ### [Equations](@id tracer-eqns) 
# ```math
# \frac{\partial \rho\chi}{\partial t} +  \nabla \cdot ( \rho\chi u) = \nabla \cdot (-\rho\delta_{D\chi}\mathrm{D_{T}}\nabla\chi) + \rho \mathrm{S}
# ```
# where  $$\chi$$ represents the tracer species, $$\mathrm{S}$$ represents the tracer source terms and $$\delta_{D\chi} \mathrm{D_{T}}$$ represents the scaled turbulent eddy diffusivity for each tracer. 
# Currently a default scaling of `1` is supported. 
# The equation as written above corresponds to a single scalar tracer, but can be extended to include 
# multiple independent tracer species.
#
# We first define an abstract tracer type, and define the 
# default function signatures. Two options are currently
# supported. [`NoTracers`](@ref no-tracers),
# and [`NTracers`](@ref multiple-tracers).
#
# ### [Abstract Tracer Type](@id abstract-tracer-type) 
#
# Default methods for a generic tracer type are defined here. 
#

abstract type TracerModel end

export NoTracers, NTracers

vars_state(::TracerModel, FT) = @vars()
vars_gradient(::TracerModel, FT) = @vars()
vars_diffusive(::TracerModel, FT) = @vars()
vars_aux(::TracerModel, FT) = @vars()

function atmos_init_aux!(
    ::TracerModel,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    nothing
end
function atmos_nodal_update_aux!(
    ::TracerModel,
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
function flux_tracers!(
    ::TracerModel,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
function diffusive!(
    ::TracerModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
function flux_diffusive!(
    ::TracerModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    nothing
end
function gradvariables!(
    ::TracerModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

# ### [NoTracers](@id no-tracers) 
# The default tracer type in both the LES and GCM configurations is the 
# no tracer model. (This means no state variables for tracers are being 
# carried around). For the purposes of this model, moist variables are 
# considered separately in `moisture.jl`. 
#
"""
    NoTracers <: TracerModel
No tracers. Default model. 
"""
struct NoTracers <: TracerModel end

# ### [NTracers](@id multiple-tracers) 
# Allows users to specify an integer corresponding to the number of 
# tracers required. This can be extended to provide a vector
# corresponding to a scaling factor for each individual tracer 
# component. Note that tracer naming is not currently supported, 
# i.e. the user must track each tracer variable based on its 
# numerical index. Sources can be added to each tracer based on the 
# same numerical index. Initial profiles must be specified using the 
# `init_state!` hook at the experiment level. 

"""
    NTracers{N, FT} <: TracerModel
Currently the simplest way to get n-tracers in an AtmosModel run 
using the existing machinery. Model input: SVector of 
diffusivity scaling coefficients. Length of SVector allows number
of tracers to be inferred. Tracers are currently identified by indices. 

# Fields
#
$(DocStringExtensions.FIELDS)
"""
struct NTracers{N, FT} <: TracerModel
    "N-component `SVector` with scaling ratios for tracer diffusivities"
    δ_χ::SVector{N, FT}
end

vars_state(tr::NTracers, FT) = @vars(ρχ::typeof(tr.δ_χ))
vars_gradient(tr::NTracers, FT) = @vars(χ::typeof(tr.δ_χ))
vars_diffusive(tr::NTracers, FT) =
    @vars(∇χ::SMatrix{3, length(tr.δ_χ), FT, 3 * length(tr.δ_χ)})
vars_aux(tr::NTracers, FT) = @vars(δ_χ::typeof(tr.δ_χ))

function atmos_init_aux!(
    tr::NTracers,
    am::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.tracers.δ_χ = tr.δ_χ
end
function gradvariables!(
    tr::NTracers,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.tracers.χ = state.tracers.ρχ * ρinv
end
function diffusive!(
    tr::NTracers,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.tracers.∇χ = ∇transform.tracers.χ
end
function flux_tracers!(
    tr::NTracers,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    u = state.ρu / state.ρ
    flux.tracers.ρχ += (state.tracers.ρχ .* u')'
end
function flux_diffusive!(
    tr::NTracers,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    d_χ = (-D_t) * aux.tracers.δ_χ' .* diffusive.tracers.∇χ
    flux_diffusive!(tr, flux, state, d_χ)
end
function flux_diffusive!(tr::NTracers, flux::Grad, state::Vars, d_χ)
    flux.tracers.ρχ += d_χ * state.ρ
end
