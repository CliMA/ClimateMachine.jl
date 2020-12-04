module Runoff

using ...VariableTemplates
using DocStringExtensions
using ...Land:SoilModel

export DrivenConstantPrecip,
    compute_surface_flux,
    NoRunoff,
    TopmodelRunoff,
    AbstractPrecipModel,
    AbstractSurfaceRunoffModel


"""
    AbstractPrecipModel{FT <: AbstractFloat}
"""
abstract type AbstractPrecipModel{FT <: AbstractFloat} end

"""
    AbstractCoupledPrecipModel{FT} <: AbstractPrecipModel{FT}

This will be used when we run coupled climate models, where precipitation
is obtained from the atmosphere model at a given location and time.
"""
abstract type AbstractCoupledPrecipModel{FT} <: AbstractPrecipModel{FT} end


"""
    AbstractDrivenPrecipModel{FT} <: AbstractPrecipModel{FT}

This will be used when we drive the land model using
re-analysis data or other user-prescribed functions for
precipitation.
"""
abstract type  AbstractDrivenPrecipModel{FT} <: AbstractPrecipModel{FT} end


"""
    DrivenConstantPrecip{FT, F} <: AbstractDrivenPrecipModel{FT}

To be used when driving the land model using precipitation data, e.g.
from reanalysis. The precipitation is a user-supplied function of time, and 
treated as constant across the grid.

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


#just commenting out for now. possibly simpler to leave out until we are ready, as not needed by Maxwell cases.
#"""
#    AbstractEvapParameterization{FT <: AbstractFloat}
#"""
#abstract type AbstractEvapParameterization{FT <: AbstractFloat} end##

#"""
#    ConstantEvap{FT} <: AbstractEvapParameterization{FT}
#
#Instance of soil surface evaporation, which is constant at subgrid scale.
#Evap can be a function of time.

## Fields
#$(DocStringExtensions.FIELDS)
#"""
#struct ConstantEvap{FT} <: AbstractEvapParameterization{FT}
#    "Mean evaporation in grid"
#    me::Function
#    function ConstantEvap{FT}(; me::Function = (t) -> FT(0.0)) where {FT}
#        new(me)
#    end
#end

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
   function compute_surface_runoff(runoff_model::NoRunoff, precip_model::AbstractPrecipModel, state::Vars)

Returns zero for net surface runoff.
"""
function compute_surface_runoff(soil::SoilModel,
                                runoff_model::NoRunoff,
                                precip_model::AbstractPrecipModel,
                                aux::Vars,
                                state::Vars,
                                t::Real,
                                )
    FT = eltype(state)
    return FT(0.0)
end

"""
    struct TopmodelRunoff :< AbstractSurfaceRunoffModel

The necessary information for determining surface runoff using TOPMODEL assumptions.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct TopmodelRunoff{FT, F1,F2, F3, F4} <: AbstractSurfaceRunoffModel
    "Constant expressing the linear relationship between topographic index and effective saturation"
    m::F1
    "Parameter of gamma distribution; function of space"
    α::F2
    "Parameter of gamma distribution; function of space"
    θ::F3
    "Minimum topographic index; function of space"
    ϕ_min::F4
    function TopmodelRunoff{FT}(m::FT, α::F2, θ::F3, ϕ_min::F4) where {FT, F2, F3, F4}
        f  = (x, y) -> FT(α(x, y))
        g  = (x, y) -> FT(θ(x, y))
        h  = (x, y) -> FT(ϕ_min(x, y))
        new{FT, typeof(m),typeof(f),typeof(g), typeof(h)}(m, f, g, h)
    end
end

"""
    function compute_mean_p(precip_model::DrivenConstantPrecip, t::Real)

Return the precipitation value for the coarse grid as a function of time,
using the user supplied function.
"""
function compute_mean_p(precip_model::DrivenConstantPrecip, t::Real)
    return precip_model(t)
end





"""
  function compute_ic(soilmodel)

Compute infiltration capacity. Currently a stand-in.
"""
function compute_ic(soil::SoilModel, state::Vars)
    ic = soil.soil_param_functions.Ksat# * d_psi_d_sl(Sl=1) * (1 - Sl[2])/dz
    return ic
end
"""
  function compute_horton_runoff(runoff_model, precip_model, state)

Compute Horton runoff.
"""
function compute_horton_runoff(soil::SoilModel,
                               runoff_model::TopmodelRunoff{FT},
                               precip_model::DrivenConstantPrecip{FT},
                               aux::Vars,
                               state::Vars,
                               t::Real,
                               ) where {FT}
    mean_p = precip_model(t)
    mean_e = FT(0.0)
    incident_water_flux = mean_p - mean_e
    ic = compute_ic(soil, state)
    if S_lsfc < FT(1) && incident_water_flux > ic
        horton_runoff = incident_water_flux - ic
    else
        horton_runoff = FT(0.0)
    end
    return horton_runoff
end


"""
  function compute_dunne_runoff(runoff_model, precip_model, state)

Compute Dunne runoff, under the assumptions that the variable
x = ϕ-ϕ_min follows a Gamma distribution.

We further assume that the local value of the effective saturation
is given by S = S̄ - m (ϕ-ϕ̄), where an overbar represents a coarse scale
value, and ϕ is the topographic index. The value of Dunne runoff
is P̄ × ∫dx f(x),
where the limits of the integral over x are from xsat (S = 1) to ∞, 
P̄ is the mean coarse grid value of precipitation, and f(x) is the gamma distribution for
x.

"""
function compute_Dunne_runoff(soil::SoilModel,
                              runoff_model::TopmodelRunoff,
                              precip_model::DrivenConstantPrecip,
                              aux::Vars,
                              state::Vars,
                              t::Real,
                              )
    FT = eltype(state)
    effective_porosity = soil.param_functions.porosity - state.soil.water.θ_i
    coarse_S_l = effective_saturation(
        effective_porosity,
        state.soil.water.ϑ_l,
    )
    x = aux.x
    y = aux.y
    α = runoff_model.α(x,y)
    θ = runoff_model.θ(x,y)
    mean_ϕ = α*θ
    ϕ_min = runoff_model.ϕ_min(x,y)
    xsat = (coarse_S_l-FT(1))/runoff_model.m+mean_ϕ - ϕ_min

    cdf_x = gamma_inc(α, xsat/θ, 2)[2]
    
    mean_p = compute_mean_p(precip_model, t)
    
    dunne_runoff = mean_p*cdf_x

    return dunne_runoff
end


#function compute_Dunne_runoff(soil::SoilModel,
#                              runoff_model::TopmodelRunoff{FT}(),
#                              precip_dist::ConstantPrecip{FT},
#                              #evaporation::ConstantEvap{FT},
#                              state::Vars,
#                              t::Real,
#    ) where {FT}
#    S_l = effective_saturation(
#    soil.param_functions.porosity,
#    state.soil.water.ϑ_l,
#    )
#    S_lsfc = state.soil.water.ϑ_l / soil.soil_param_functions.porosity
#    mean_p = precip_dist.mp(t)
#    mean_e = FT(0.0)#evaporation.me(t)
#    if S_lsfc >= FT(1)
#	dunne_runoff = mean_p - mean_e
#    else#
#	dunne_runoff = FT(0.0)
#    end
#    return dunne_runoff
#end

"""
  function compute_surface_runoff(runoff_model, precip_model, state)

Compute surface runoff, accounting for precipitation, evaporation, according to TOPMODEL assumptions.
"""
function compute_surface_runoff(soil::SoilModel,
                                runoff_model::TopmodelRunoff{FT},
                                precip_model::DrivenConstantPrecip{FT},
                                aux::Vars,
                                state::Vars,
                                t::Real
                               ) where {FT}
   
   #r_l_horton = compute_horton_runoff(soil, runoff_model, precip_model, aux, state, t)
   r_l_dunne = compute_dunne_runoff(soil, runoff_model, precip_model, aux, state, t)
   return r_l_dunne#+r_l_horton
end

function compute_surface_flux(soil::SoilModel,
                              runoff_model::AbstractSurfaceRunoffModel,
                              precip_model::AbstractPrecipModel{FT},
                              aux::Vars,
                              state::Vars,
                              t::Real
                              ) where {FT}
    mean_p = precip_model.mp(t)
    net_runoff = compute_surface_runoff(soil, runoff_model, precip_model, aux, state, t)
    net_flux = net_runoff - mean_p
    return net_flux
end


end
