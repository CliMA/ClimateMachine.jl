##### Moisture tendencies

#####
##### First order fluxes
#####

function flux(::Advect{TotalMoisture}, m, state, aux, t, ts, direction)
    u = state.ρu / state.ρ
    return u * state.moisture.ρq_tot
end

function flux(::Advect{LiquidMoisture}, m, state, aux, t, ts, direction)
    u = state.ρu / state.ρ
    return u * state.moisture.ρq_liq
end

function flux(::Advect{IceMoisture}, m, state, aux, t, ts, direction)
    u = state.ρu / state.ρ
    return u * state.moisture.ρq_ice
end

#####
##### Sources
#####

function source(
    s::Subsidence{TotalMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    z = altitude(m, aux)
    w_sub = subsidence_velocity(s, z)
    k̂ = vertical_unit_vector(m, aux)
    return -state.ρ * w_sub * dot(k̂, diffusive.moisture.∇q_tot)
end

export CreateClouds
"""
    CreateClouds{PV <: Union{LiquidMoisture,IceMoisture}} <: TendencyDef{Source, PV}

A source/sink to `q_liq` and `q_ice` implemented as a relaxation towards
equilibrium in the Microphysics module.
The default relaxation timescales are defined in CLIMAParameters.jl.
"""
struct CreateClouds{PV <: Union{LiquidMoisture, IceMoisture}} <:
       TendencyDef{Source, PV} end

CreateClouds() = (CreateClouds{LiquidMoisture}(), CreateClouds{IceMoisture}())

function source(
    s::CreateClouds{LiquidMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    # get current temperature and phase partition
    FT = eltype(state)
    q = PhasePartition(ts)
    T = air_temperature(ts)

    # phase partition corresponding to the current T and q.tot
    # (this is not the same as phase partition from saturation adjustment)
    ts_eq = PhaseEquil_ρTq(m.param_set, state.ρ, T, q.tot)
    q_eq = PhasePartition(ts_eq)

    # cloud condensate as relaxation source terms
    S_q_liq = conv_q_vap_to_q_liq_ice(m.param_set.microphys.liq, q_eq, q)

    return state.ρ * S_q_liq
end

function source(
    s::CreateClouds{IceMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    # get current temperature and phase partition
    FT = eltype(state)
    q = PhasePartition(ts)
    T = air_temperature(ts)

    # phase partition corresponding to the current T and q.tot
    # (this is not the same as phase partition from saturation adjustment)
    ts_eq = PhaseEquil_ρTq(m.param_set, state.ρ, T, q.tot)
    q_eq = PhasePartition(ts_eq)

    # cloud condensate as relaxation source terms
    S_q_ice = conv_q_vap_to_q_liq_ice(m.param_set.microphys.ice, q_eq, q)

    return state.ρ * S_q_ice
end

function source(
    s::RemovePrecipitation{TotalMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    if has_condensate(ts)
        nt = remove_precipitation_sources(s, m, state, aux, ts)
        return nt.S_ρ_qt
    else
        FT = eltype(state)
        return FT(0)
    end
end

function source(
    s::WarmRain_1M{TotalMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    nt = warm_rain_sources(m, state, aux, ts)
    return nt.S_ρ_qt
end

function source(
    s::WarmRain_1M{LiquidMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    nt = warm_rain_sources(m, state, aux, ts)
    return nt.S_ρ_ql
end

function source(
    s::RainSnow_1M{TotalMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    nt = rain_snow_sources(m, state, aux, ts)
    return nt.S_ρ_qt
end

function source(
    s::RainSnow_1M{LiquidMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    nt = rain_snow_sources(m, state, aux, ts)
    return nt.S_ρ_ql
end

function source(
    s::RainSnow_1M{IceMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    nt = rain_snow_sources(m, state, aux, ts)
    return nt.S_ρ_qi
end
