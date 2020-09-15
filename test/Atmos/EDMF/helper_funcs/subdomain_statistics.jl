#### Subdomain statistics

compute_subdomain_statistics(m::AtmosModel, state, aux, t) =
    compute_subdomain_statistics(
        m,
        state,
        aux,
        t,
        m.turbconv.micro_phys.statistical_model,
    )

"""
    compute_subdomain_statistics(
        m::AtmosModel{FT},
        state::Vars,
        aux::Vars,
        t::Real,
        statistical_model::SubdomainMean,
    ) where {FT}

Returns a cloud fraction and cloudy and dry thermodynamic
states in the subdomain.
"""
function compute_subdomain_statistics(
    m::AtmosModel{FT},
    state::Vars,
    aux::Vars,
    t::Real,
    statistical_model::SubdomainMean,
) where {FT}
    ts_en = recover_thermo_state_en(m, state, aux)
    cloud_frac = has_condensate(ts_en) ? FT(1) : FT(0)
    dry = ts_en
    cloudy = ts_en
    return (dry = dry, cloudy = cloudy, cloud_frac = cloud_frac)
end
