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
    SnowVolumetricInternalEnergy

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
    SingleLayerSnowModel{pt, ft, rm}  <: BalanceLaw 

The surface snow model balance law type, with prognostic variables of
snow water equivalent (SWE) and snow volumetric internal energy (ρe_int).

This single-layer model allows for simulating changes in snow internal energy and water
 mass due to fluxes at the top and bottom of the snowpack, as well as 
liquid water runoff. As the snow model differential equations are
ordinary, there is no need to specify flux methods, or boundary conditions.
# Fields
$(DocStringExtensions.FIELDS)
"""
struct  SingleLayerSnowModel{pt, ft, rm} <: BalanceLaw
    "Parameter functions"
    parameters::pt
    "Forcing functions"
    forcing::ft
    "Runoff model"
    runoff_model::rm
end

vars_state(surface:: SingleLayerSnowModel, st::Prognostic, FT) = @vars(swe::FT, ρe_int::FT)

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



end
