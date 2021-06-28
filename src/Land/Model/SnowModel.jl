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
include("SnowModelParameterizations.jl")
using .SnowModelParameterizations

export SingleLayerSnowModel,
    FRSingleLayerSnowModel,
    NoSnowModel,
    FRSingleLayerSnowModel,
    SnowWaterEquivalent,
    SnowVolumetricInternalEnergy,
    SurfaceVolumetricEnergy,
    PrescribedForcing,
    SnowParameters,
    FluxDivergence,
    ForceRestore


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
    SurfaceVolumetricEnergy <: AbstractPrognosticVariable

A prognostic variable type for the snow model. Used only for
dispatching on.
"""
struct SurfaceVolumetricEnergy <: AbstractPrognosticVariable end

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
    FRSingleLayerSnowModel{pt, ft, fs, fes,fe}  <: BalanceLaw 

The surface snow model balance law type, with prognostic variables of
snow water equivalent (SWE), surface volumetric energy (ρe_surf),
 and snow volumetric internal energy (ρe_int).

This single-layer model allows for simulating changes in snow internal energy and water
 mass due to fluxes at the top and bottom of the snowpack, as well as 
liquid water runoff. As the snow model differential equations are
ordinary, there is no need to specify flux methods, or boundary conditions.
Initial conditions are still required.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct FRSingleLayerSnowModel{pt, ft, fes, fe} <: BalanceLaw
    "Parameter functions"
    parameters::pt
    "Forcing functions"
    forcing::ft
    "Initial condition for ρe_surf"
    initial_ρe_surf::fes
    "Initial condition for ρe_int"
    initial_ρe_int::fe
end

vars_state(surface:: SingleLayerSnowModel, st::Prognostic, FT) = @vars(ρe_int::FT, swe::FT)
vars_state(surface:: FRSingleLayerSnowModel, st::Prognostic, FT) = @vars(ρe_int::FT, ρe_surf::FT, swe::FT)
vars_state(surface:: FRSingleLayerSnowModel, st::Auxiliary, FT) = @vars(t_h::FT)

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
    )
    FT = eltype(aux)
    κ_snow = land.snow.parameters.κ_snow
    z_snow = land.snow.parameters.z_snow
    ρ_snow = land.snow.parameters.ρ_snow
    param_set = land.param_set
    
    ρe_int = land.snow.initial_ρe_int(aux)
    ρe_surf = land.snow.initial_ρe_surf(aux)
    l = liquid_fraction(ρe_int, ρ_snow, param_set)

    ρc_snow = volumetric_heat_capacity(l, ρ_snow, param_set)
    #T_snow = snow_temperature(ρe_int, l, ρ_snow, param_set)
    T_snow = usu_bulk_snow_T(ρe_int, ρ_snow, z_snow, param_set)
    T_surf = snow_temperature(ρe_surf,l,ρ_snow, param_set)
    ν = FT(2.0*π/24/3600)
    d = (FT(2)*κ_snow/(ρc_snow*ν))^FT(0.5)
    h = max(FT(0), z_snow-d)
    Q_bottom = land.snow.forcing.Q_bottom(eltype(aux)(0.0))
    num = FT(2.0)*z_snow*T_snow-T_surf*(z_snow-h) - Q_bottom*h^FT(2.0)/κ_snow
    denom = h+z_snow
    aux.snow.t_h = num/denom
end

function Land.land_nodal_update_auxiliary_state!(
    land::LandModel,
    snow::FRSingleLayerSnowModel,
    state,
    aux,
    t,
    )
    FT = eltype(state)
    κ_snow = land.snow.parameters.κ_snow
    z_snow = land.snow.parameters.z_snow
    ρ_snow = land.snow.parameters.ρ_snow
    param_set = land.param_set
    
    ρe_int = state.snow.ρe_int
    ρe_surf = state.snow.ρe_surf
    l = liquid_fraction(ρe_int, ρ_snow, param_set)

    ρc_snow = volumetric_heat_capacity(l, ρ_snow, param_set)
    #    T_snow = snow_temperature(ρe_int, l, ρ_snow, param_set)
    T_snow = usu_bulk_snow_T(ρe_int, ρ_snow, z_snow, param_set)
    T_surf = snow_temperature(ρe_surf,l,ρ_snow, param_set)
    ν = FT(2.0*π/24/3600)
    d = (FT(2)*κ_snow/(ρc_snow*ν))^FT(0.5)
    h = max(FT(0), z_snow-d)
    Q_bottom = land.snow.forcing.Q_bottom(t)
    num = FT(2.0)*z_snow*T_snow-T_surf*(z_snow-h) - Q_bottom*h^FT(2.0)/κ_snow
    denom = h+z_snow
    aux.snow.t_h = num/denom
    
end



"""
    FluxDivergence <: TendencyDef{Source}

A source term for the snow volumetric internal energy equation. It is equal
to (Q_surf - Q_bottom)/z_snow.
"""
struct FluxDivergence{FT} <: TendencyDef{Source} end


"""
    ForceRestore <: TendencyDef{Source}

A source term for the surface volumetrci energy, from the force-restore model.
"""
struct ForceRestore{FT} <: TendencyDef{Source} end


prognostic_vars(::FluxDivergence) = (SnowVolumetricInternalEnergy(),)
prognostic_vars(::ForceRestore) = (SurfaceVolumetricEnergy(),)


precompute(source_type::FluxDivergence, land::LandModel, args, tt::Source) =
    NamedTuple()
precompute(source_type::ForceRestore, land::LandModel, args, tt::Source) =
    NamedTuple()

function source(::SnowVolumetricInternalEnergy, s::FluxDivergence, land::LandModel, args)
    @unpack state, diffusive, aux, t, direction = args
    Q_surf = compute_q_surf(land,land.snow,args)
    Q_bottom = land.snow.forcing.Q_bottom(t)
    divflux = -(Q_surf-Q_bottom)/land.snow.parameters.z_snow
    return divflux
end

function compute_q_surf(land::LandModel, snow::Union{FRSingleLayerSnowModel,SingleLayerSnowModel}, args)
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
=#
    

function source(::SurfaceVolumetricEnergy, s::ForceRestore, land::LandModel, args)
    @unpack state, diffusive, aux, t, direction = args
    FT = eltype(state)

    κ_snow = land.snow.parameters.κ_snow
    ρ_snow = land.snow.parameters.ρ_snow
    param_set = land.param_set

    ρe_int = state.snow.ρe_int
    l = liquid_fraction(ρe_int, ρ_snow, param_set)
    ρc_snow = volumetric_heat_capacity(l, ρ_snow, param_set)
    #T_snow = snow_temperature(ρe_int, l, ρ_snow, param_set)
    T_snow = usu_bulk_snow_T(ρe_int, ρ_snow, z_snow, param_set)
    ρe_surf = state.snow.ρe_surf
    T_surf = snow_temperature(ρe_surf,l,ρ_snow, param_set)
    ν = FT(2.0*π/24/3600)
    T_h = aux.snow.t_h
    d = (FT(2)*κ_snow/(ρc_snow*ν))^FT(0.5)
    Q_surf = land.snow.forcing.Q_surf(t)
    divflux = -ν*ρc_snow*(d/κ_snow*Q_surf + (T_surf -T_h))
                         
    return divflux
end


end
