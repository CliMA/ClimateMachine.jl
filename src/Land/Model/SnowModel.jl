module SnowModel

using DocStringExtensions
using UnPack
using ..Land
using ..VariableTemplates
using ..BalanceLaws
import ..BalanceLaws:
    BalanceLaw,
    prognostic_vars,
    flux,
    source,
    precompute,
    eq_tends,
    vars_state,
    Prognostic,
    Auxiliary,
    Gradient,
    GradientFlux

using ...DGMethods: LocalGeometry, DGModel
using StaticArrays: SVector

export SingleLayerSnowModel,
    FRSingleLayerSnowModel,
    NoSnowModel,
    FRSingleLayerSnowModel,
    SnowWaterEquivalent,
    SnowVolumetricInternalEnergy,
    SnowSurfaceTemperature,
    PrescribedForcing,
    SnowParameters,
    FluxDivergence


"""
    SnowWaterEquivalent <: AbstractPrognosticVariable

A prognostic variable type for the snow model. Used only for
dispatching on.
"""
struct SnowWaterEquivalent <: AbstractPrognosticVariable end

"""
    SnowVolumetricInternalEnergy <: AbstractPrognosticVariable

A prognostic variable type for the snow model. Used only for
dispatching on.
"""
struct SnowVolumetricInternalEnergy <: AbstractPrognosticVariable end

"""
    SnowSurfaceTemperature <: AbstractPrognosticVariable

A prognostic variable type for the snow model. Used only for
dispatching on.
"""
struct SnowSurfaceTemperature <: AbstractPrognosticVariable end

"""
    NoSnowModel <: BalanceLaw

The default snow model, which does not add any prognostic variables
to the land model and therefore does not model snow.
"""
struct NoSnowModel <: BalanceLaw end



"""
    AbstractSnowParameters{FT <: AbstractFloat}
"""
abstract type AbstractSnowParameters{FT <: AbstractFloat} end


"""
    SnowParameters{FT, TK, Tρ, Tscf} <: AbstractSnowParameters{FT}

Necessary parameters for the snow model. At present, these are all
floating point numbers, but in the long run we will be able to swap out
different models for the different parameters.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SnowParameters{FT, TK, Tρ, Tz} <: AbstractSnowParameters{FT}
    "Bulk thermal conductivity. W/m/K"
    κ_snow::TK
    "Bulk density. kg/m^3"
    ρ_snow::Tρ
    "Depth of snow"
    z_snow::Tz
end


"""
    AbstractSnowForcingModel{FT <: AbstractFloat}
"""
abstract type AbstractSnowForcingModel{FT <: AbstractFloat} end


"""
    PrescribedForcing{FT, F1, F2} <: AbstractSnowForcingModel{
# Fields
$(DocStringExtensions.FIELDS)
"""
struct PrescribedForcing{FT, F1, F2} <: AbstractSnowForcingModel{FT}
    "surface flux function (of time)"
    Q_surf::F1
    "Bottom flux function of time"
    Q_bottom::F2
end

function PrescribedForcing(::Type{FT};
    Q_surf::Function = (t) -> eltype(t)(0.0),
    Q_bottom::Function = (t) -> eltype(t)(0.0),
) where {FT}
    args = (Q_surf, Q_bottom)
    return PrescribedForcing{FT, typeof.(args)...}(args...)
end

"""
    SingleLayerSnowModel{pt, ft, fs, fe}  <: BalanceLaw 

The surface snow model balance law type, with prognostic variables of
snow water equivalent (SWE) and snow volumetric internal energy (ρe_int).

This single-layer model allows for simulating changes in snow internal energy and water
 mass due to fluxes at the top and bottom of the snowpack, as well as 
liquid water runoff. As the snow model differential equations are
ordinary, there is no need to specify flux methods, or boundary conditions.
Initial conditions are still required.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SingleLayerSnowModel{pt, ft,  fe} <: BalanceLaw
    "Parameter functions"
    parameters::pt
    "Forcing functions"
    forcing::ft
    "Initial condition for ρe_int"
    initial_ρe_int::fe
end

"""
    FRSingleLayerSnowModel{pt, ft, fs, fts,fe}  <: BalanceLaw 

The surface snow model balance law type, with prognostic variables of
snow water equivalent (SWE), surface temperature, and snow volumetric internal energy (ρe_int).

This single-layer model allows for simulating changes in snow internal energy and water
 mass due to fluxes at the top and bottom of the snowpack, as well as 
liquid water runoff. As the snow model differential equations are
ordinary, there is no need to specify flux methods, or boundary conditions.
Initial conditions are still required.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct FRSingleLayerSnowModel{pt, ft, fts, fe} <: BalanceLaw
    "Parameter functions"
    parameters::pt
    "Forcing functions"
    forcing::ft
    "Initial condition for Tsurf"
    initial_tsurf::fts
    "Initial condition for ρe_int"
    initial_ρe_int::fe
end

vars_state(surface:: SingleLayerSnowModel, st::Prognostic, FT) = @vars(ρe_int::FT, swe::FT)
vars_state(surface:: FRSingleLayerSnowModel, st::Prognostic, FT) = @vars(ρe_int::FT, tsurf::FT, swe::FT)
vars_state(surface:: FRSingleLayerSnowModel, st::Auxiliary, FT) = @vars(th::FT)

function Land.land_init_aux!(
    land::LandModel,
    snow::Union{NoSnowModel,  SingleLayerSnowModel},
    aux,
    geom::LocalGeometry,
) end

function Land.land_nodal_update_auxiliary_state!(
    land::LandModel,
    snow::Union{NoSnowModel,  SingleLayerSnowModel},
    state,
    aux,
    t,
) end

function Land.land_init_aux!(
    land::LandModel,
    snow::FRSingleLayerSnowModel,
    aux,
    geom::LocalGeometry,
) end

function Land.land_nodal_update_auxiliary_state!(
    land::LandModel,
    snow::FRSingleLayerSnowModel,
    state,
    aux,
    t,
    
) end



"""
    FluxDivergence <: TendencyDef{Source}

A source term for the snow volumetric internal energy equation. It is equal
to (Q_surf - Q_bottom)/z_snow.
"""
struct FluxDivergence{FT} <: TendencyDef{Source} end

#=
"""
    TsurfFluxDivergence <: TendencyDef{Source}

A source term for the surface temperature, from the force-restore model.
"""
struct TsurfFluxDivergence{FT} <: TendencyDef{Source} end
=#

prognostic_vars(::FluxDivergence) = (SnowVolumetricInternalEnergy(),)
#prognostic_vars(::TsurfFluxDivergence) = (SnowSurfaceTemperature(),)


precompute(source_type::FluxDivergence, land::LandModel, args, tt::Source) =
    NamedTuple()
#precompute(source_type::TsurfFluxDivergence, land::LandModel, args, tt::Source) =
#    NamedTuple()
function source(::SnowVolumetricInternalEnergy, s::FluxDivergence, land::LandModel, args)
    @unpack state, diffusive, aux, t, direction = args
    Q_surf = compute_q_surf(land,land.snow,args)
    Q_bottom = land.snow.forcing.Q_bottom(t)
    divflux = -(Q_surf-Q_bottom)/land.snow.parameters.z_snow
    return divflux
end

function compute_q_surf(land::LandModel, snow::SingleLayerSnowModel, args)
    @unpack t = args
    return land.snow.forcing.Q_surf(t)
end

#=
function compute_q_surf(land::LandModel, snow::FRSingleLayerSnowModel, args)
    @unpack state  = args
    κ = land.snow.parameters.κ_snow
    ρ_snow = land.snow.parameters.ρ_snow
    param_set = land.param_set
    ρc_snow = volumetric_heat_capacity(l, ρ_snow, param_set)
    ρe_int = state.snow.ρe_int
    l = liquid_fraction(ρe_int, ρ_snow, param_set)
    T_snow = T_snow_ave(ρe_int, l, ρ_snow, param_set)

    ν = FT(2.0*π/24/3600)
    d = (FT(2)*κ_snow/(ρc_snow*ν))^FT(0.5)
    flux = -κ/d*(T_snow - state.snow.tsurf)
    return flux
end

    

function source(::SnowSurfaceTemperature, s::TsurfFluxDivergence, land::LandModel, args)
    @unpack state, diffusive, aux, t, direction = args
    κ = land.snow.parameters.κ_snow
    ρ_snow = land.snow.parameters.ρ_snow
    param_set = land.param_set
    ρc_snow = volumetric_heat_capacity(l, ρ_snow, param_set)
    ρe_int = state.snow.ρe_int
    l = liquid_fraction(ρe_int, ρ_snow, param_set)
    T_snow = T_snow_ave(ρe_int, l, ρ_snow, param_set)

    ν = FT(2.0*π/24/3600)
    d = (FT(2)*κ_snow/(ρc_snow*ν))^FT(0.5)
    Q_bottom = -κ/d*(T_snow - state.snow.tsurf)
    divflux = -FT(2)/(d*ρc_snow)*(Q_surf-Q_bottom)
    return divflux
end
=#

end
