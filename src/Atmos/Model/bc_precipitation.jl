"""
    OutflowPrecipitation{PV <: Union{Rain, Snow}} <: BCDef{PV}

Free flux out of the domain.
"""
struct OutflowPrecipitation{PV <: Union{Rain, Snow}} <: BCDef{PV} end

bc_val(bc::OutflowPrecipitation, atmos::AtmosModel, ::NF12∇, args) =
    DefaultBCValue()

function atmos_precipitation_normal_boundary_flux_second_order!(
    nf,
    bc_precipitation::OutflowPrecipitation,
    atmos,
    args...,
) end
