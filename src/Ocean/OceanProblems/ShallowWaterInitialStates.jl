using ..ShallowWater: TurbulenceClosure, LinearDrag, ConstantViscosity

function null_init_state!(
    ::HomogeneousBox,
    ::TurbulenceClosure,
    state,
    aux,
    coord,
    t,
)
    T = eltype(state.U)
    state.U = @SVector zeros(T, 2)
    state.η = 0
    return nothing
end

η_lsw(x, y, t) = cos(π * x) * cos(π * y) * cos(√2 * π * t)
u_lsw(x, y, t) = 2^(-0.5) * sin(π * x) * cos(π * y) * sin(√2 * π * t)
v_lsw(x, y, t) = 2^(-0.5) * cos(π * x) * sin(π * y) * sin(√2 * π * t)

function lsw_init_state!(
    m::ShallowWaterModel,
    p::HomogeneousBox,
    state,
    aux,
    coords,
    t,
)
    state.U = @SVector [
        u_lsw(coords[1], coords[2], t),
        v_lsw(coords[1], coords[2], t),
    ]

    state.η = η_lsw(coords[1], coords[2], t)

    return nothing
end

v_lkw(x, y, t) = 0
u_lkw(x, y, t) = exp(-0.5 * y^2) * exp(-0.5 * (x - t + 5)^2)
η_lkw(x, y, t) = 1 + u_lkw(x, y, t)

function lkw_init_state!(
    m::ShallowWaterModel,
    p::HomogeneousBox,
    state,
    aux,
    coords,
    t,
)
    state.U = @SVector [
        u_lkw(coords[1], coords[2], t),
        v_lkw(coords[1], coords[2], t),
    ]

    state.η = η_lkw(coords[1], coords[2], t)

    return nothing
end

R₋(ϵ) = (-1 - sqrt(1 + (2 * π * ϵ)^2)) / (2ϵ)
R₊(ϵ) = (-1 + sqrt(1 + (2 * π * ϵ)^2)) / (2ϵ)
D(ϵ) =
    (R₊(ϵ) * (exp(R₋(ϵ)) - 1) + R₋(ϵ) * (1 - exp(R₊(ϵ)))) /
    (exp(R₊(ϵ)) - exp(R₋(ϵ)))
R₂(x₁, ϵ) =
    (1 / D(ϵ)) * (
        (
            (R₊(ϵ) * (exp(R₋(ϵ)) - 1)) * exp(R₊(ϵ) * x₁) +
            (R₋(ϵ) * (1 - exp(R₊(ϵ)))) * exp(R₋(ϵ) * x₁)
        ) / (exp(R₊(ϵ)) - exp(R₋(ϵ)))
    )
R₁(x₁, ϵ) =
    (π / D(ϵ)) * (
        1 .+
        (
            (exp(R₋(ϵ)) - 1) * exp(R₊(ϵ) * x₁) .+
            (1 - exp(R₊(ϵ))) * exp(R₋(ϵ) * x₁)
        ) / (exp(R₊(ϵ)) - exp(R₋(ϵ)))
    )

𝒱(x₁, y₁, ϵ) = R₂(x₁, ϵ) * sin.(π * y₁)
𝒰(x₁, y₁, ϵ) = -R₁(x₁, ϵ) * cos.(π * y₁)
ℋ(x₁, y₁, ϵ, βᵖ, fₒ, γ) =
    (R₂(x₁, ϵ) / (π * fₒ)) * γ * cos(π * y₁) +
    (R₁(x₁, ϵ) / π) *
    (sin(π * y₁) * (1.0 + βᵖ * (y₁ - 0.5)) + (βᵖ / π) * cos(π * y₁))

function gyre_init_state!(
    m::SWModel,
    p::HomogeneousBox,
    T::LinearDrag,
    state,
    aux,
    coords,
    t,
)
    FT = eltype(state)
    τₒ = p.τₒ
    fₒ = m.fₒ
    β = m.β
    Lˣ = p.Lˣ
    Lʸ = p.Lʸ
    H = p.H

    γ = T.λ

    βᵖ = β * Lʸ / fₒ
    ϵ = γ / (Lˣ * β)

    _grav::FT = grav(m.param_set)

    uˢ(ϵ) = (τₒ * D(ϵ)) / (H * γ * π)
    hˢ(ϵ) = (fₒ * Lˣ * uˢ(ϵ)) / _grav

    u = uˢ(ϵ) * 𝒰(coords[1] / Lˣ, coords[2] / Lʸ, ϵ)
    v = uˢ(ϵ) * 𝒱(coords[1] / Lˣ, coords[2] / Lʸ, ϵ)
    h = hˢ(ϵ) * ℋ(coords[1] / Lˣ, coords[2] / Lʸ, ϵ, βᵖ, fₒ, γ)

    state.U = @SVector [H * u, H * v]

    state.η = h

    return nothing
end

t1(x, δᵐ) = cos((√3 * x) / (2 * δᵐ)) + (√3^-1) * sin((√3 * x) / (2 * δᵐ))
t2(x, δᵐ) = 1 - exp((-x) / (2 * δᵐ)) * t1(x, δᵐ)
t3(y, Lʸ) = π * sin(π * y / Lʸ)
t4(x, Lˣ, C) = C * (1 - x / Lˣ)

η_munk(x, y, Lˣ, Lʸ, δᵐ, C) = t4(x, Lˣ, C) * t3(y, Lʸ) * t2(x, δᵐ)

function gyre_init_state!(
    m::SWModel,
    p::HomogeneousBox,
    V::ConstantViscosity,
    state,
    aux,
    coords,
    t,
)
    FT = eltype(state.U)
    _grav::FT = grav(m.param_set)

    τₒ = p.τₒ
    fₒ = m.fₒ
    β = m.β
    Lˣ = p.Lˣ
    Lʸ = p.Lʸ
    H = p.H

    ν = V.ν

    δᵐ = (ν / β)^(1 / 3)
    C = τₒ / (_grav * H) * (fₒ / β)

    state.η = η_munk(coords[1], coords[2], Lˣ, Lʸ, δᵐ, C)
    state.U = @SVector zeros(T, 2)

    return nothing
end
