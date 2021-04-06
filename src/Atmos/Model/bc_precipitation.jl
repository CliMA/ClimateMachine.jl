abstract type PrecipitationBC end

"""
    OutflowPrecipitation() :: PrecipitationBC

Free flux out of the domain.
"""
struct OutflowPrecipitation <: PrecipitationBC end
function atmos_precipitation_boundary_state!(
    nf,
    bc_precipitation::OutflowPrecipitation,
    atmos,
    _...,
) end
function atmos_precipitation_normal_boundary_flux_second_order!(
    nf,
    bc_precipitation::OutflowPrecipitation,
    atmos,
    _...,
) end
