abstract type PrecipitationBC end

"""
    ImpermeablePrecipitation() :: PrecipitationBC

No precipitation flux.
"""
struct ImpermeablePrecipitation <: PrecipitationBC end
function atmos_precipitation_boundary_state!(
    nf,
    bc_precipitation::ImpermeablePrecipitation,
    atmos,
    args...,
) end
function atmos_precipitation_normal_boundary_flux_second_order!(
    nf,
    bc_precipitation::ImpermeablePrecipitation,
    atmos,
    args...,
) end
