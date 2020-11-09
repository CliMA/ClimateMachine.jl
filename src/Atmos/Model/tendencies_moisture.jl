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
