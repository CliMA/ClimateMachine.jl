advective_courant(m::AtmosLinearModel, a...) = advective_courant(m.atmos, a...)

nondiffusive_courant(m::AtmosLinearModel, a...) =
    nondiffusive_courant(m.atmos, a...)

diffusive_courant(m::AtmosLinearModel, a...) = diffusive_courant(m.atmos, a...)

function advective_courant(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    direction,
)
    k̂ = vertical_unit_vector(m.orientation, aux)
    if direction isa VerticalDirection
        norm_u = abs(dot(state.ρu, k̂)) / state.ρ
    elseif direction isa HorizontalDirection
        norm_u = norm((state.ρu - dot(state.ρu, k̂) * k̂) / state.ρ)
    else
        norm_u = norm(state.ρu / state.ρ)
    end
    return Δt * norm_u / Δx
end

function nondiffusive_courant(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    direction,
)
    k̂ = vertical_unit_vector(m.orientation, aux)
    if direction isa VerticalDirection
        norm_u = abs(dot(state.ρu, k̂)) / state.ρ
    elseif direction isa HorizontalDirection
        norm_u = norm((state.ρu - dot(state.ρu, k̂) * k̂) / state.ρ)
    else
        norm_u = norm(state.ρu / state.ρ)
    end
    return Δt * (norm_u + soundspeed(m, m.moisture, state, aux)) / Δx
end

function diffusive_courant(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    direction,
)
    ν, τ = turbulence_tensors(m.turbulence, state, diffusive, aux, 0)
    FT = eltype(state)

    if ν isa Real
        return Δt * ν / (Δx * Δx)
    else
        k̂ = vertical_unit_vector(m.orientation, aux)
        ν_vert = dot(ν, k̂)

        if direction isa VerticalDirection
            return Δt * ν_vert / (Δx * Δx)
        elseif direction isa HorizontalDirection
            ν_horz = ν - ν_vert * k̂
            return Δt * norm(ν_horz) / (Δx * Δx)
        else
            return Δt * norm(ν) / (Δx * Δx)
        end
    end
end
