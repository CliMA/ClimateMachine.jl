#### Subdomain statistics

function compute_subdomain_statistics(m::AtmosModel, args, ts_gm, ts_en)
    turbconv = turbconv_model(m)
    return compute_subdomain_statistics(
        turbconv.micro_phys.statistical_model,
        m,
        args,
        ts_gm,
        ts_en,
    )
end

"""
    compute_subdomain_statistics(
        statistical_model::SubdomainMean,
        m::AtmosModel{FT},
        args,
        ts_gm,
        ts_en,
    ) where {FT}

Returns a cloud fraction and cloudy and dry thermodynamic
states in the subdomain.
"""
function compute_subdomain_statistics(
    statistical_model::SubdomainMean,
    m::AtmosModel{FT},
    args,
    ts_gm,
    ts_en,
) where {FT}
    cloud_frac = has_condensate(ts_en) ? FT(1) : FT(0)
    dry = ts_en
    cloudy = ts_en
    return (dry = dry, cloudy = cloudy, cloud_frac = cloud_frac)
end
