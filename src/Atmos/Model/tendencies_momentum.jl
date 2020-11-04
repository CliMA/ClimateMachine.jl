##### Momentum tendencies

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
