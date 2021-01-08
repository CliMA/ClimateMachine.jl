module Runoff

using ...VariableTemplates
using DocStringExtensions
using ...Land:SoilModel
using ...Land:matric_potential, effective_saturation, volumetric_liquid_fraction, pressure_head
using Printf

export DrivenConstantPrecip,
    compute_surface_flux,
    compute_dunne_runoff,
    compute_horton_runoff,
    compute_generalized_runoff,
    NoRunoff,
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


"""
    AbstractSurfaceRunoffModel

Abstract type for different surface runoff models.
""" 
abstract type AbstractSurfaceRunoffModel end

"""
    NoRunoff <: AbstractSurfaceRunoffModel

Chosen when no runoff is to be modeled. 

In this case, the the net surface flux is used as the 
boundary condition for the soil water component.
"""
struct NoRunoff <: AbstractSurfaceRunoffModel end


"""
    function compute_surface_runoff(soil::SoilModel,
                                    runoff_model::NoRunoff,
                                    precip_model::AbstractPrecipModel,
                                    aux::Vars,
                                    state::Vars,
                                    t::Real,
                                    )

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
    function compute_mean_p(precip_model::DrivenConstantPrecip, t::Real)

Return the precipitation value for the coarse grid as a function of time,
using the user supplied function.
"""
function compute_mean_p(precip_model::DrivenConstantPrecip, t::Real)
    return precip_model(t)
end


"""
    function compute_ic(soil::SoilModel,
                      runoff_model::CoardGridRunoff,
                      state::Vars)

Compute infiltration capacity for unsaturated soil. A positive value
implies infiltration into the soil, negative values cannot occur.
"""
function compute_ic(soil::SoilModel, runoff_model::CoarseGridRunoff,state::Vars)
    FT = eltype(state)
    Δz = runoff_model.Δz
    hydraulics = soil.water.hydraulics
    θ_l = volumetric_liquid_fraction(
        state.soil.water.ϑ_l,
        soil.param_functions.porosity
    )
    S = effective_saturation(soil.param_functions.porosity,θ_l)
    ic = soil.param_functions.Ksat*(FT(1)-matric_potential(hydraulics,S)/Δz)
    return ic
end


"""
    function compute_horton_runoff(soil::SoilModel,
                                   runoff_model::CoarseGridRunoff{FT},
                                   precip_model::DrivenConstantPrecip{FT},
                                   aux::Vars,
                                   state::Vars,
                                   t::Real,
                                   ) where {FT}

Compute infiltration excess in unsaturated soil. 

This runoff is positive or zero.
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
    if coarse_S_l <= FT(1) && incident_water_flux < -ic
        horton_runoff = abs(incident_water_flux) - ic # defined to be positive
    else
        horton_runoff = FT(0.0)
    end
    return horton_runoff
end

"""
    function compute_dunne_runoff(soil::SoilModel,
                                   runoff_model::CoarseGridRunoff{FT},
                                   precip_model::DrivenConstantPrecip{FT},
                                   aux::Vars,
                                   state::Vars,
                                   t::Real,
                                   ) where {FT}

Compute saturation excess for soil saturated at the surface.

This runoff is positive or zero.
"""
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
    if coarse_S_l > FT(1) && incident_water_flux < FT(0)#if evap >precip, no runoff
	dunne_runoff = abs(incident_water_flux) # positive.
    else
	dunne_runoff = FT(0.0)
    end
    return dunne_runoff
end



"""
    function compute_surface_runoff(soil::SoilModel,
                                   runoff_model::CoarseGridRunoff{FT},
                                   precip_model::DrivenConstantPrecip{FT},
                                   aux::Vars,
                                   state::Vars,
                                   t::Real,
                                   ) where {FT}

Compute surface runoff, accounting for precipitation, evaporation, and surface runoff.
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



###### VS:


"""
    function compute_generalized_ic(soil::SoilModel,
                                   runoff_model::CoarseGridRunoff{FT},
                                   state::Vars,
                                   )

Compute infiltration capacity in saturated and unsaturated soil. 

Positive implies infiltration into the soil, negative values can 
arise for saturated soils and imply water exiting the surface.
"""
function compute_generalized_ic(soil::SoilModel,
                                runoff_model::CoarseGridRunoff,
                                state::Vars
                                )
    FT = eltype(state)
    Δz = runoff_model.Δz
    hydraulics = soil.water.hydraulics
    i_c = soil.param_functions.Ksat*(FT(1)-pressure_head(
        hydraulics,
        soil.param_functions.porosity,
        soil.param_functions.S_s,
        state.soil.water.ϑ_l,
        state.soil.water.θ_i)/Δz
                                    )
    return i_c
end

"""
    function compute_generalized_runoff(soil::SoilModel,
                                   runoff_model::CoarseGridRunoff{FT},
                                   precip_model::DrivenConstantPrecip{FT},
                                   aux::Vars,
                                   state::Vars,
                                   t::Real,
                                   ) where {FT}

Compute surface runoff, accounting for precipitation, evaporation,
and surface runoff, under the assumption that the infiltration 
excess does not default to zero for saturated soils.
"""
function compute_generalized_runoff(soil::SoilModel,
                               runoff_model::CoarseGridRunoff{FT},
                               precip_model::DrivenConstantPrecip{FT},
                               aux::Vars,
                               state::Vars,
                               t::Real,
                               ) where {FT}
    mean_p = precip_model(t)
    incident_water_flux = mean_p
    ic = compute_generalized_ic(soil, runoff_model, state)
    # Need to take into account the ice to determine if the soil is truly saturated.
    effective_porosity = soil.param_functions.porosity - state.soil.water.θ_i
    coarse_S_l = effective_saturation(
        effective_porosity,
        state.soil.water.ϑ_l,
    )
    # i_c > 0 by definition. If incident water flux points in -z and is larger in mag,
    # runoff. if it is smaller in mag (or positive), there is no runoff.
    
    if incident_water_flux < -ic
        runoff = abs(incident_water_flux) - ic # defined to be positive
    else
        runoff = FT(0.0)
    end
    return runoff
end


"""
    function compute_surface_flux(soil::SoilModel,
                                   runoff_model::CoarseGridRunoff{FT},
                                   precip_model::DrivenConstantPrecip{FT},
                                   aux::Vars,
                                   state::Vars,
                                   t::Real,
                                   ) where {FT}

Computes the (negative of the) flux entering the soil, accounting for
surface runoff, precipitation, and eventually evaporation.

This can then be used as a boundary condition.
"""
function compute_surface_flux(soil::SoilModel,
                              runoff_model::AbstractSurfaceRunoffModel,
                              precip_model::AbstractPrecipModel,
                              aux::Vars,
                              state::Vars,
                              t::Real
                              )
    mean_p = compute_mean_p(precip_model,t)
    incident_water_flux = mean_p
    net_runoff = compute_generalized_runoff(soil, runoff_model, precip_model,
                                            aux, state, t)# or compute_surface_runoff
    net_flux = incident_water_flux-(-net_runoff)
    return net_flux
end


end
