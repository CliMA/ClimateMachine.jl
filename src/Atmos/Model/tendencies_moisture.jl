##### Moisture tendencies

#####
##### First order fluxes
#####

function flux(::Advect{TotalMoisture}, atmos, args)
    @unpack state = args
    u = state.ρu / state.ρ
    return u * state.moisture.ρq_tot
end

function flux(::Advect{LiquidMoisture}, atmos, args)
    @unpack state = args
    u = state.ρu / state.ρ
    return u * state.moisture.ρq_liq
end

function flux(::Advect{IceMoisture}, atmos, args)
    @unpack state = args
    u = state.ρu / state.ρ
    return u * state.moisture.ρq_ice
end

#####
##### Second order fluxes
#####

function flux(::MoistureDiffusion{TotalMoisture}, atmos, args)
    @unpack state, aux, t, diffusive = args
    ν, D_t, τ = turbulence_tensors(atmos, state, diffusive, aux, t)
    d_q_tot = (-D_t) .* diffusive.moisture.∇q_tot
    return d_q_tot * state.ρ
end

function flux(::MoistureDiffusion{LiquidMoisture}, atmos, args)
    @unpack state, aux, t, diffusive = args
    ν, D_t, τ = turbulence_tensors(atmos, state, diffusive, aux, t)
    d_q_liq = (-D_t) .* diffusive.moisture.∇q_liq
    return d_q_liq * state.ρ
end

function flux(::MoistureDiffusion{IceMoisture}, atmos, args)
    @unpack state, aux, t, diffusive = args
    ν, D_t, τ = turbulence_tensors(atmos, state, diffusive, aux, t)
    d_q_ice = (-D_t) .* diffusive.moisture.∇q_ice
    return d_q_ice * state.ρ
end

#####
##### Sources
#####

function source(s::Subsidence{TotalMoisture}, m, args)
    @unpack state, aux, diffusive = args
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

function source(s::CreateClouds{LiquidMoisture}, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
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

function source(s::CreateClouds{IceMoisture}, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
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

function source(s::RemovePrecipitation{TotalMoisture}, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
    if has_condensate(ts)
        nt = remove_precipitation_sources(s, m, args)
        return nt.S_ρ_qt
    else
        FT = eltype(state)
        return FT(0)
    end
end

function source(s::WarmRain_1M{TotalMoisture}, m, args)
    nt = warm_rain_sources(m, args)
    return nt.S_ρ_qt
end

function source(s::WarmRain_1M{LiquidMoisture}, m, args)
    nt = warm_rain_sources(m, args)
    return nt.S_ρ_ql
end

function source(s::RainSnow_1M{TotalMoisture}, m, args)
    nt = rain_snow_sources(m, args)
    return nt.S_ρ_qt
end

function source(s::RainSnow_1M{LiquidMoisture}, m, args)
    nt = rain_snow_sources(m, args)
    return nt.S_ρ_ql
end

function source(s::RainSnow_1M{IceMoisture}, m, args)
    nt = rain_snow_sources(m, args)
    return nt.S_ρ_qi
end
