##### Energy tendencies

#####
##### First order fluxes
#####

function flux(::Advect{Energy}, atmos, args)
    @unpack state = args
    return (state.ρu / state.ρ) * state.energy.ρe
end

function flux(::Advect{ρθ_liq_ice}, atmos, args)
    @unpack state = args
    return (state.ρu / state.ρ) * state.energy.ρθ_liq_ice
end

function flux(::Pressure{Energy}, atmos, args)
    @unpack state = args
    @unpack ts = args.precomputed
    return state.ρu / state.ρ * air_pressure(ts)
end

#####
##### Second order fluxes
#####

struct ViscousFlux{PV <: Union{Energy,ρθ_liq_ice}} <: TendencyDef{Flux{SecondOrder}, PV} end
function flux(::ViscousFlux{Energy}, atmos, args)
    @unpack state = args
    @unpack τ = args.precomputed.turbulence
    return τ * state.ρu
end

function flux(::ViscousFlux{ρθ_liq_ice}, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    return state.ρ * (-D_t) .* diffusive.energy.∇θ_liq_ice
end

function flux(::HyperdiffViscousFlux{Energy}, atmos, args)
    @unpack state, hyperdiffusive = args
    return hyperdiffusive.hyperdiffusion.ν∇³u_h * state.ρu
end

function flux(::HyperdiffEnthalpyFlux{Energy}, atmos, args)
    @unpack state, hyperdiffusive = args
    return hyperdiffusive.hyperdiffusion.ν∇³h_tot * state.ρ
end

struct DiffEnthalpyFlux{PV <: Energy} <: TendencyDef{Flux{SecondOrder}, PV} end
function flux(::DiffEnthalpyFlux{Energy}, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    d_h_tot = -D_t .* diffusive.energy.∇h_tot
    return d_h_tot * state.ρ
end

#####
##### Sources
#####

function source(s::Subsidence{Energy}, m, args)
    @unpack state, aux, diffusive = args
    z = altitude(m, aux)
    w_sub = subsidence_velocity(s, z)
    k̂ = vertical_unit_vector(m, aux)
    return -state.ρ * w_sub * dot(k̂, diffusive.∇h_tot)
end

function source(s::RemovePrecipitation{Energy}, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
    if has_condensate(ts)
        nt = remove_precipitation_sources(s, m, args)
        return nt.S_ρ_e
    else
        FT = eltype(state)
        return FT(0)
    end
end

function source(s::WarmRain_1M{Energy}, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_e
end

function source(s::RainSnow_1M{Energy}, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_e
end
