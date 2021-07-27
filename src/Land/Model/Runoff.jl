module Runoff

using LinearAlgebra
using DocStringExtensions


using ...VariableTemplates
using ...Land:
    SoilModel,
    pressure_head,
    hydraulic_conductivity,
    get_temperature,
    effective_saturation,
    impedance_factor,
    viscosity_factor,
    moisture_factor

export AbstractPrecipModel,
    DrivenConstantPrecip,
    AbstractSurfaceRunoffModel,
    NoRunoff,
    compute_surface_grad_bc,
    CoarseGridRunoff

"""
    AbstractPrecipModel{FT <: AbstractFloat}
"""
abstract type AbstractPrecipModel{FT <: AbstractFloat} end

"""
    DrivenConstantPrecip{FT, F} <: AbstractPrecipModel{FT}

Instance of a precipitation distribution where the precipication value
is constant across the domain. However, this value can change in time.

Precipitation is assumed to be aligned with the vertical, i.e. P⃗ = Pẑ,
with P<0.  

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
    CoarseGridRunoff{FT} <: AbstractSurfaceRunoffModel

Chosen when no subgrid effects are to be modeled.
"""
struct CoarseGridRunoff{FT} <: AbstractSurfaceRunoffModel
    "Mean vertical resolution at the surface"
    Δz::FT
end

"""
    function compute_surface_grad_bc(soil::SoilModel,
                                     runoff_model::CoarseGridRunoff,
                                     precip_model::AbstractPrecipModel,
                                     n̂,
                                     state⁻::Vars,
                                     diff⁻::Vars,
                                     aux⁻::Vars,
                                     t::Real
                                     )

Given a runoff model and a precipitation distribution function, compute 
the surface water flux normal to the surface. The sign of the flux
(inwards or outwards) is determined by the magnitude of precipitation,
evaporation, and infiltration.
"""
function compute_surface_grad_bc(
    soil::SoilModel,
    runoff_model::CoarseGridRunoff,
    precip_model::AbstractPrecipModel,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t::Real,
)
    FT = eltype(state⁻)
    precip_vector = (FT(0), FT(0), precip_model(t))
    incident_water_flux = dot(precip_vector, n̂)
    Δz = runoff_model.Δz
    water = soil.water
    param_functions = soil.param_functions
    hydraulics = water.hydraulics(aux⁻)
    ν = param_functions.porosity
    θ_r = param_functions.water.θ_r(aux⁻)
    S_s = param_functions.water.S_s(aux⁻)



    T = get_temperature(soil.heat, aux⁻, t)
    θ_i = state⁻.soil.water.θ_i
    # Ponding Dirichlet BC
    ϑ_bc = FT(ν - θ_i)
    # Value below surface
    ϑ_below = state⁻.soil.water.ϑ_l

    # Approximate derivative of hydraulic head with respect to z
    ∂h∂z =
        FT(1) +
        (
            pressure_head(hydraulics, ν, S_s, θ_r, ϑ_bc, θ_i) -
            pressure_head(hydraulics, ν, S_s, θ_r, ϑ_below, θ_i)
        ) / Δz


    S_l = effective_saturation(ν, ϑ_bc, θ_r)
    f_i = θ_i / (ϑ_bc + θ_i)
    impedance_f = impedance_factor(water.impedance_factor, f_i)
    viscosity_f = viscosity_factor(water.viscosity_factor, T)
    moisture_f = moisture_factor(water.moisture_factor, hydraulics, S_l)
    K = hydraulic_conductivity(
        param_functions.water.Ksat(aux⁻),
        impedance_f,
        viscosity_f,
        moisture_f,
    )

    i_c = n̂ * (K * ∂h∂z)
    if incident_water_flux < -norm(i_c) # More negative if both are negative,
        #ponding BC
        K∇h⁺ = i_c
    else

        K∇h⁺ = n̂ * (-FT(2) * incident_water_flux) - diff⁻.soil.water.K∇h
    end
    return K∇h⁺
end



"""
    function compute_surface_grad_bc(soil::SoilModel,
                                     runoff_model::NoRunoff,
                                     precip_model::AbstractPrecipModel,
                                     n̂,
                                     state⁻::Vars,
                                     aux⁻::Vars,
                                     diff⁻::Vars,
                                     t::Real
                                     )

Given a runoff model and a precipitation distribution function, compute 
the surface water flux normal to the surface. In this case, no runoff is
assumed. The direction of the flux (inwards or outwards) depends on the
magnitude of evaporation, precipitation, and infiltration.
"""
function compute_surface_grad_bc(
    soil::SoilModel,
    runoff_model::NoRunoff,
    precip_model::AbstractPrecipModel,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t::Real,
)
    FT = eltype(state⁻)
    precip_vector = (FT(0), FT(0), precip_model(t))
    incident_water_flux = dot(precip_vector, n̂)
    K∇h⁺ = n̂ * (-FT(2) * incident_water_flux) - diff⁻.soil.water.K∇h
    return K∇h⁺
end

end
