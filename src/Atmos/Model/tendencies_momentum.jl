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
