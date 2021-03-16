using ..Atmos
using ..TurbulenceClosures
using ..TurbulenceConvection

# Method definitions for diagnostics collection for all the components
# of `AtmosModel`.

const AtmosComponentTypes = Union{
    MoistureModel,
    PrecipitationModel,
    RadiationModel,
    TracerModel,
    TurbulenceClosureModel,
    TurbulenceConvectionModel,
}

dv_PointwiseDiagnostic(
    ::AtmosConfigType,
    ::PointwiseDiagnostic,
    ::AtmosComponentTypes,
    ::AtmosModel,
    ::States,
    ::AbstractFloat,
    ::Dict{Symbol, Any},
) = 0
dv_HorizontalAverage(
    ::AtmosConfigType,
    ::HorizontalAverage,
    ::AtmosComponentTypes,
    ::AtmosModel,
    ::States,
    ::AbstractFloat,
    ::Dict{Symbol, Any},
) = 0
