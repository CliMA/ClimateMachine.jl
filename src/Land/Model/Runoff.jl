module Runoff

using LinearAlgebra
using DocStringExtensions
using CLIMAParameters
using CLIMAParameters.Planet:
    ρ_cloud_liq, ρ_cloud_ice, grav, R_v
using CLIMAParameters.Atmos.Microphysics: D_vapor

using ClimateMachine.Thermodynamics: Liquid, q_vap_saturation_generic

using ...VariableTemplates
using ...Land: LandModel,
    SoilModel,
    pressure_head,
    hydraulic_conductivity,
    get_temperature

export AbstractPrecipModel,
    DrivenConstantPrecip,
    AbstractEvapModel,
    NoEvaporation,
    Evaporation,
    DrivenConstantEvap,
    compute_evaporation,
    AbstractSurfaceRunoffModel,
    NoRunoff,
    compute_infiltration,
    CoarseGridRunoff

"""
    AbstractPrecipModel{FT <: AbstractFloat}
"""
abstract type AbstractPrecipModel{FT <: AbstractFloat} end

"""
    AbstractEvapModel{FT <: AbstractFloat}
"""
abstract type AbstractEvapModel{FT <: AbstractFloat} end
"""
    NoEvaporation{FT} <: AbstractEvapModel{FT}

Chosen when no evaporation is to be modeled.
"""
struct NoEvaporation{FT} <: AbstractEvapModel{FT} end
Base.@kwdef struct Evaporation{FT} <: AbstractEvapModel{FT}
    "Transition coefficient, unitless"
    k::FT = 0.8
    "Maximum dry soil layer thickness"
    d::FT = 1.5e-2
    "Density of Moist Air (kg/m^3)"
    ρa::FT  = 1.0
    "Specific humidity of air, unitless"
    q_va::FT = 1.0
    "Surface conductance (m/s)"
    g_ae::FT = 1.0
end
"""
    DrivenConstantEvap{FT, F} <: AbstractEvapModel{FT}

Instance of a evaporation model which is a prescribed function of time.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct DrivenConstantEvap{FT, F} <: AbstractEvapModel{FT}
    "Mean evaporation in grid"
    me::F
    function DrivenConstantEvap{FT}(me::F) where {FT, F}
        new{FT, F}(me)
    end
end

function (dce::DrivenConstantEvap{FT})(t::Real) where {FT}
    return FT(dce.me(t))
end


function compute_evaporation(em::NoEvaporation{FT}, lm::LandModel, state::Vars, aux::Vars, t::Real) where {FT}
    return FT(0.0)
end

function compute_evaporation(em::DrivenConstantEvap{FT}, lm::LandModel, state::Vars, aux::Vars, t::Real) where {FT}
    return em(t)
end

function compute_evaporation(em::Evaporation{FT}, lm::LandModel, state::Vars, aux::Vars, t::Real) where {FT}
    k = em.k
    d = em.d
    ρa = em.ρa
    q_va = em.q_va
    g_ae = em.g_ae
    
    params = lm.param_set
    _g = grav(params)
    _Rv = R_v(params)
    _ρl = ρ_cloud_liq(params)
    _Dν = D_vapor(params)

    ν = lm.soil.param_functions.porosity
    T = get_temperature(lm.soil.heat, aux, t)
    θ_r = lm.soil.water.param_functions.θ_r
    
    ϑ_l = state.soil.water.ϑ_l
    θ_i = state.soil.water.θ_i

    # compute specific humidity inside soil pores near surface
    eff_porosity = ν - θ_i
    θ_l = volumetric_liquid_fraction(ϑ_l,eff_porosity)
    DSL = θ_l < k*ν ? d*(FT(1)-θ_l/(k*ν)) : FT(0)    
    S_l = effective_saturation(ν, ϑ_l, θ_r)
    ψ = matric_potential(lm.soil.water.hydraulics, S_l)
    factor = exp(_g*ψ/_Rv/T)
    q_vstar = q_vap_saturation_generic(params, T, ρa, Liquid())
    q_v = factor*q_vstar

    # compute surface conductance
    τ = FT(2/3)*ν^FT(2)
    g_soil = _Dν*τ/(DSL+eps(FT))
    g_eff = FT(1)/(FT(1)/g_ae+FT(1)/g_soil)
    # compute mass flux of water vapor from evaporation
    e_volume_flux = - ρa/_ρl* g_eff*(q_va-q_v)
    return e_volume_flux
    
end


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
    CoarseGridRunoff{FT} <: AbstractSurfaceRunoffModel

Chosen when no subgrid effects are to be modeled.
"""
struct CoarseGridRunoff{FT} <: AbstractSurfaceRunoffModel
    "Mean vertical resolution at the surface"
    Δz::FT
end


function compute_infiltration(soil::SoilModel,
                              runoff_model::CoarseGridRunoff,
                              incident_water_flux::Real,
                              n̂,
                              state⁻::Vars,
                              diff⁻::Vars,
                              aux⁻::Vars,
                              t::Real)
    
    FT = eltype(state⁻)
    Δz = runoff_model.Δz
    water = soil.water
    param_functions = soil.param_functions
    hydraulics = water.hydraulics
    ν = param_functions.porosity
    
    
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
            pressure_head(hydraulics, param_functions, ϑ_bc, θ_i) -
            pressure_head(hydraulics, param_functions, ϑ_below, θ_i)
        ) / Δz
    
    K =
        soil.param_functions.Ksat * hydraulic_conductivity(
            water.impedance_factor,
            water.viscosity_factor,
            water.moisture_factor,
            hydraulics,
            θ_i,
            param_functions.porosity,
            T,
            ϑ_bc / ν,# when ice is present, K still measured with ν, not νeff.
        )
    
    i_c = n̂ * (K * ∂h∂z)
    if incident_water_flux < -norm(i_c) # More negative if both are negative,
        #ponding BC
        flux⁺ = -i_c
    else
        
        flux⁺ = n̂ * (FT(2) * incident_water_flux) + diff⁻.soil.water.K∇h
    end
    return flux⁺
end
    




"""
    function compute_infiltration(soil::SoilModel,
                                  runoff_model::NoRunoff,
                                  incident_water_flux::Real,
                                  n̂,
                                  state⁻::Vars,
                                  aux⁻::Vars,
                                  diff⁻::Vars,
                                  t::Real
                                  )

Given a runoff model and a precipitation distribution function, compute 
the surface water infiltration. This can be a function of time, and state.
"""
function compute_infiltration(soil::SoilModel,
                              runoff_model::NoRunoff,
                              incident_water_flux::Real,
                              n̂,
                              state⁻::Vars,
                              diff⁻::Vars,
                              aux⁻::Vars,
                              t::Real)
    FT = eltype(state⁻)
    flux⁺ = n̂ * (FT(2) * incident_water_flux) + diff⁻.soil.water.K∇h
    return flux⁺
end

end
