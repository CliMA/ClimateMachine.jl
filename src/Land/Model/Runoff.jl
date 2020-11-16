module Runoff

using ...VariableTemplates
using DocStringExtensions

export AbstractPrecipModel,
    DrivenConstantPrecip,
    AbstractSurfaceRunoffModel,
    NoRunoff,
    compute_surface_flux

"""
    AbstractPrecipModel{FT <: AbstractFloat}
"""
abstract type AbstractPrecipModel{FT <: AbstractFloat} end

"""
    DrivenConstantPrecip{FT, F} <: AbstractPrecipModel{FT}

Instance of a precipitation distribution where the precipication value
is constant across the domain. However, this value can change in time.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct DrivenConstantPrecip{FT, F} <: AbstractPrecipModel{FT}
    "Mean precipitation in grid"
    mp::F
    function DrivenConstantPrecip{FT}(mp::F) where {FT, F}
        new{FT, F}(mp)
    end
end

function (dcp::DrivenConstantPrecip{FT})(t::Real) where {FT}
    return FT(dcp.mp(t))
end

"""
    AbstractSurfaceRunoffModel

Abstract type for different surface runoff models. Currently, only
`NoRunoff` is supported.
"""
abstract type AbstractSurfaceRunoffModel end

"""
    NoRunoff <: AbstractSurfaceRunoffModel

Chosen when no runoff is to be modeled.
"""
struct NoRunoff <: AbstractSurfaceRunoffModel end


"""
    function compute_surface_runoff(
        runoff_model::NoRunoff,
        precip_model::AbstractPrecipModel,
        state::Vars
    )

Returns zero for net surface runoff when `NoRunoff`
is used.
"""
function compute_surface_runoff(
    runoff_model::NoRunoff,
    precip_model::AbstractPrecipModel{FT},
    state::Vars,
) where {FT}
    return FT(0.0)
end


"""
    compute_surface_flux(
        runoff_model::AbstractSurfaceRunoffModel,
        precip_model::AbstractPrecipModel{FT},
        state::Vars,
        t::Real,
    ) where {FT}

Given a runoff model and a precipitation distribution function, compute 
the surface water flux. This can be a function of time, and state.
"""
function compute_surface_flux(
    runoff_model::AbstractSurfaceRunoffModel,
    precip_model::AbstractPrecipModel{FT},
    state::Vars,
    t::Real,
) where {FT}
    mean_p = precip_model(t)
    net_runoff = compute_surface_runoff(runoff_model, precip_model, state)
    net_flux = net_runoff - mean_p
    return net_flux
end


end
