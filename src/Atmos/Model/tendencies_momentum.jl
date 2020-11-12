##### Momentum tendencies

#####
##### First order fluxes
#####

function flux(::Advect{Momentum}, m, state, aux, t, ts, direction)
    return state.ρu .* (state.ρu / state.ρ)'
end

function flux(::PressureGradient{Momentum}, m, state, aux, t, ts, direction)
    if m.ref_state isa HydrostaticState
        return (air_pressure(ts) - aux.ref_state.p) * I
    else
        return air_pressure(ts) * I
    end
end

#####
##### Sources
#####

export Gravity
struct Gravity{PV <: Momentum} <: TendencyDef{Source, PV} end
Gravity() = Gravity{Momentum}()
function source(
    s::Gravity{Momentum},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    if m.ref_state isa HydrostaticState
        return -(state.ρ - aux.ref_state.ρ) * aux.orientation.∇Φ
    else
        return -state.ρ * aux.orientation.∇Φ
    end
end

export Coriolis
struct Coriolis{PV <: Momentum} <: TendencyDef{Source, PV} end
Coriolis() = Coriolis{Momentum}()
function source(
    s::Coriolis{Momentum},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    FT = eltype(state)
    _Omega::FT = Omega(m.param_set)
    # note: this assumes a SphericalOrientation
    return -SVector(0, 0, 2 * _Omega) × state.ρu
end

export GeostrophicForcing
struct GeostrophicForcing{PV <: Momentum, FT} <: TendencyDef{Source, PV}
    f_coriolis::FT
    u_geostrophic::FT
    v_geostrophic::FT
end
function GeostrophicForcing(
    ::Type{FT},
    f_coriolis,
    u_geostrophic,
    v_geostrophic,
) where {FT}
    return GeostrophicForcing{Momentum, FT}(
        FT(f_coriolis),
        FT(u_geostrophic),
        FT(v_geostrophic),
    )
end
function source(
    s::GeostrophicForcing{Momentum},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    u_geo = SVector(s.u_geostrophic, s.v_geostrophic, 0)
    ẑ = vertical_unit_vector(m, aux)
    fkvector = s.f_coriolis * ẑ
    return -fkvector × (state.ρu .- state.ρ * u_geo)
end
