##### Energy tendencies

#####
##### First order fluxes
#####

function flux(::Energy, ::Advect, atmos, args)
    @unpack state = args
    return (state.ρu / state.ρ) * state.energy.ρe
end

function flux(::ρθ_liq_ice, ::Advect, atmos, args)
    @unpack state = args
    return (state.ρu / state.ρ) * state.energy.ρθ_liq_ice
end

function flux(::Energy, ::Pressure, atmos, args)
    @unpack state = args
    @unpack ts = args.precomputed
    return state.ρu / state.ρ * air_pressure(ts)
end

#####
##### Second order fluxes
#####

struct ViscousFlux <: TendencyDef{Flux{SecondOrder}} end
function flux(::Energy, ::ViscousFlux, atmos, args)
    @unpack state = args
    @unpack τ = args.precomputed.turbulence
    return τ * state.ρu
end

function flux(::ρθ_liq_ice, ::ViscousFlux, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    return state.ρ * (-D_t) .* diffusive.energy.∇θ_liq_ice
end

function flux(::Energy, ::HyperdiffViscousFlux, atmos, args)
    @unpack state, hyperdiffusive = args
    return hyperdiffusive.hyperdiffusion.ν∇³u_h * state.ρu
end

function flux(::Energy, ::HyperdiffEnthalpyFlux, atmos, args)
    @unpack state, hyperdiffusive = args
    return hyperdiffusive.hyperdiffusion.ν∇³h_tot * state.ρ
end

struct DiffEnthalpyFlux <: TendencyDef{Flux{SecondOrder}} end
function flux(::Energy, ::DiffEnthalpyFlux, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    d_h_tot = -D_t .* diffusive.energy.∇h_tot
    return d_h_tot * state.ρ
end

#####
##### Sources
#####

function source(::Energy, s::Subsidence, m, args)
    @unpack state, aux, diffusive = args
    z = altitude(m, aux)
    w_sub = subsidence_velocity(s, z)
    k̂ = vertical_unit_vector(m, aux)
    return -state.ρ * w_sub * dot(k̂, diffusive.energy.∇h_tot)
end

function source(::Energy, s::RemovePrecipitation, m, args)
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

function source(::Energy, s::WarmRain_1M, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_e
end

function source(::Energy, s::RainSnow_1M, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_e
end
