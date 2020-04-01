using ..Mesh.Grids: Direction, HorizontalDirection, VerticalDirection

advective_courant(m::AtmosLinearModel, a...) = advective_courant(m.atmos, a...)

nondiffusive_courant(m::AtmosLinearModel, a...) =
    nondiffusive_courant(m.atmos, a...)

diffusive_courant(m::AtmosLinearModel, a...) = diffusive_courant(m.atmos, a...)

norm_u(state::Vars, k̂::AbstractVector, ::VerticalDirection) =
    abs(dot(state.ρu, k̂)) / state.ρ
norm_u(state::Vars, k̂::AbstractVector, ::HorizontalDirection) =
    norm((state.ρu .- dot(state.ρu, k̂) * k̂) / state.ρ)
norm_u(state::Vars, k̂::AbstractVector, ::Direction) = norm(state.ρu / state.ρ)

norm_ν(ν::Real, k̂::AbstractVector, ::Direction) = ν
norm_ν(ν::AbstractVector, k̂::AbstractVector, ::VerticalDirection) = dot(ν, k̂)
norm_ν(ν::AbstractVector, k̂::AbstractVector, ::HorizontalDirection) =
    norm(ν - dot(ν, k̂) * k̂)
norm_ν(ν::AbstractVector, k̂::AbstractVector, ::Direction) = norm(ν)

function advective_courant(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
    k̂ = vertical_unit_vector(m.orientation, aux)
    normu = norm_u(state, k̂, direction)
    return Δt * normu / Δx
end

function nondiffusive_courant(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
    k̂ = vertical_unit_vector(m.orientation, aux)
    normu = norm_u(state, k̂, direction)
    return Δt * (normu + soundspeed(m, m.moisture, state, aux)) / Δx
end

function diffusive_courant(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
    ν, τ = turbulence_tensors(m.turbulence, state, diffusive, aux, t)
    k̂ = vertical_unit_vector(m.orientation, aux)
    normν = norm_ν(ν, k̂, direction)
    return Δt * normν / (Δx * Δx)
end
