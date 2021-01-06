module Runoff

using ...VariableTemplates
using DocStringExtensions
using ...Land:SoilModel
using ...Land:matric_potential, effective_saturation, volumetric_liquid_fraction

export DrivenConstantPrecip,
    compute_surface_flux,
    compute_dunne_runoff,
    compute_horton_runoff,
    NoRunoff,
    TopmodelRunoff,
    AbstractPrecipModel,
    AbstractSurfaceRunoffModel,
    CoarseGridRunoff


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


#just commenting out for now. possibly simpler to leave out until we are ready
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
    CoarseGridRunoff <: AbstractSurfaceRunoffModel

Chosen when no subgrid effects are to be modeled.
"""
struct CoarseGridRunoff{FT,F} <: AbstractSurfaceRunoffModel
    "Vertical resolution at the surface"
    Δz::F
    function CoarseGridRunoff{FT}(Δz::F) where {FT, F}
        new{FT, F}(Δz)
    end
end



"""
    struct TopmodelRunoff :< AbstractSurfaceRunoffModel

The TOPMODEL Runoff model for parameterizing subgrid variations in surface water content.

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

Compute infiltration capacity. Positive by convention.
"""
function compute_ic(soil::SoilModel, runoff_model::CoarseGridRunoff,state::Vars)
    FT = eltype(state)
    Δz = runoff_model.Δz
    hydraulics = soil.water.hydraulics
    θ_l = volumetric_liquid_fraction(state.soil.water.ϑ_l, soil.param_functions.porosity)
    S = effective_saturation(
        soil.param_functions.porosity,
        θ_l,
    )
    
    ic = soil.param_functions.Ksat*(FT(1)-matric_potential(hydraulics,S)/Δz)
    return ic
end
"""
  function compute_horton_runoff(runoff_model, precip_model, state)

Compute Horton runoff. Runoff, if nonzero, is positive.
"""
function compute_horton_runoff(soil::SoilModel,
                               runoff_model::CoarseGridRunoff{FT},
                               precip_model::DrivenConstantPrecip{FT},
                               aux::Vars,
                               state::Vars,
                               t::Real,
                               ) where {FT}
    mean_p = precip_model(t)
    incident_water_flux = mean_p
    ic = compute_ic(soil, runoff_model, state)
    # Need to take into account the ice to determine if the soil is truly saturated.
    effective_porosity = soil.param_functions.porosity - state.soil.water.θ_i
    coarse_S_l = effective_saturation(
        effective_porosity,
        state.soil.water.ϑ_l,
    )
    # i_c > 0 by definition. If incident water flux points in -z and is larger in mag,
    # runoff. if it is smaller in mag (or positive), there is no runoff.
    if coarse_S_l < FT(1) && incident_water_flux < -ic
        horton_runoff = abs(incident_water_flux) - ic # defined to be positive
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

    return dunne_runoff#CHECK SIGN
end


function compute_dunne_runoff(soil::SoilModel,
                              runoff_model::CoarseGridRunoff,
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
    
    mean_p = compute_mean_p(precip_model,t)
    incident_water_flux = mean_p
    if coarse_S_l >= FT(1) && incident_water_flux < FT(0)#if evap >precip, no runoff
	dunne_runoff = abs(incident_water_flux) # positive.
    else
	dunne_runoff = FT(0.0)
    end
    return dunne_runoff
end

"""
  function compute_surface_runoff(runoff_model, precip_model, state)

Compute surface runoff, accounting for precipitation, evaporation, according to TOPMODEL assumptions.
"""
function compute_surface_runoff(soil::SoilModel,
                                runoff_model::CoarseGridRunoff,
                                precip_model::DrivenConstantPrecip,
                                aux::Vars,
                                state::Vars,
                                t::Real
                               )
    r_l_dunne = compute_dunne_runoff(soil, runoff_model, precip_model, aux, state, t)
    r_l_horton = compute_horton_runoff(soil, runoff_model, precip_model, aux, state, t)
   return r_l_horton+r_l_dunne
end

function compute_surface_flux(soil::SoilModel,
                              runoff_model::AbstractSurfaceRunoffModel,
                              precip_model::AbstractPrecipModel,
                              aux::Vars,
                              state::Vars,
                              t::Real
                              )
    mean_p = compute_mean_p(precip_model,t)
    incident_water_flux = mean_p
    net_runoff = compute_surface_runoff(soil, runoff_model, precip_model, aux, state, t)
    net_flux = incident_water_flux-(-net_runoff)
    return net_flux
end


end
