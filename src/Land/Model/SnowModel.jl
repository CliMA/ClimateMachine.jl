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
    NoSnowModel,
    SnowWaterEquivalent,
    SnowVolumetricInternalEnergy,
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
#    "Initial condition for SWE"
#    initial_swe::fs
    "Initial condition for ρe_int"
    initial_ρe_int::fe
end



vars_state(surface:: SingleLayerSnowModel, st::Prognostic, FT) = @vars(ρe_int::FT, swe::FT)

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



"""
    FluxDivergence <: TendencyDef{Source}

A source term for the snow volumetric internal energy equation. It is equal
to (Q_surf - Q_bottom)/z_snow.
"""
struct FluxDivergence{FT} <: TendencyDef{Source} end


prognostic_vars(::FluxDivergence) = (SnowVolumetricInternalEnergy(),)


precompute(source_type::FluxDivergence, land::LandModel, args, tt::Source) =
    NamedTuple()

function source(::SnowVolumetricInternalEnergy, s::FluxDivergence, land::LandModel, args)
    @unpack state, diffusive, aux, t, direction = args
    Q_surf = land.snow.forcing.Q_surf(t)
    Q_bottom = land.snow.forcing.Q_bottom(t)
    divflux = -(Q_surf-Q_bottom)/land.snow.parameters.z_snow
    return divflux
end


end
