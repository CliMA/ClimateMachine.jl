##### Mass tendencies

function flux(::Advect{Mass}, m, state, aux, t, ts, direction)
    return state.ρu
end

#####
##### Sources
#####

function source(s::Subsidence{Mass}, m, state, aux, t, ts, direction, diffusive)
    z = altitude(m, aux)
    w_sub = subsidence_velocity(s, z)
    k̂ = vertical_unit_vector(m, aux)
    return -state.ρ * w_sub * dot(k̂, diffusive.moisture.∇q_tot)
end
