using ..Mesh.Grids: Direction, HorizontalDirection, VerticalDirection
using ..TurbulenceClosures

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
    k̂ = vertical_unit_vector(m, aux)
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
    k̂ = vertical_unit_vector(m, aux)
    normu = norm_u(state, k̂, direction)
    # TODO: Change this to new_thermo_state
    # so that Courant computations do not depend
    # on the aux state.
    ts = recover_thermo_state(m, state, aux)
    ss = soundspeed_air(ts)
    return Δt * (normu + ss) / Δx
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
    ν, _, _ = turbulence_tensors(m, state, diffusive, aux, t)
    ν = ν isa Real ? ν : diag(ν)
    k̂ = vertical_unit_vector(m, aux)
    normν = norm_ν(ν, k̂, direction)
    return Δt * normν / (Δx * Δx)
end
