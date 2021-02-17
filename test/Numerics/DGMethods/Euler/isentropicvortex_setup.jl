import ClimateMachine.Atmos: atmos_init_aux!, vars_state

Base.@kwdef struct IsentropicVortexSetup{FT}
    p∞::FT = 10^5
    T∞::FT = 300
    ρ∞::FT = air_density(param_set, FT(T∞), FT(p∞))
    translation_speed::FT = 150
    translation_angle::FT = pi / 4
    vortex_speed::FT = 50
    vortex_radius::FT = 1 // 200
    domain_halflength::FT = 1 // 20
end

function (setup::IsentropicVortexSetup)(
    problem,
    bl,
    state,
    aux,
    localgeo,
    t,
    args...,
)
    FT = eltype(state)
    x = MVector(localgeo.coord)

    ρ∞ = setup.ρ∞
    p∞ = setup.p∞
    T∞ = setup.T∞
    translation_speed = setup.translation_speed
    α = setup.translation_angle
    vortex_speed = setup.vortex_speed
    R = setup.vortex_radius
    L = setup.domain_halflength

    u∞ = SVector(translation_speed * cos(α), translation_speed * sin(α), 0)

    x .-= u∞ * t
    # make the function periodic
    x .-= floor.((x .+ L) / 2L) * 2L

    @inbounds begin
        r = sqrt(x[1]^2 + x[2]^2)
        δu_x = -vortex_speed * x[2] / R * exp(-(r / R)^2 / 2)
        δu_y = vortex_speed * x[1] / R * exp(-(r / R)^2 / 2)
    end
    u = u∞ .+ SVector(δu_x, δu_y, 0)

    _kappa_d::FT = kappa_d(param_set)
    T = T∞ * (1 - _kappa_d * vortex_speed^2 / 2 * ρ∞ / p∞ * exp(-(r / R)^2))
    # adiabatic/isentropic relation
    p = p∞ * (T / T∞)^(FT(1) / _kappa_d)
    ρ = air_density(bl.param_set, T, p)

    state.ρ = ρ
    state.ρu = ρ * u
    e_kin = u' * u / 2
    state.energy.ρe = ρ * total_energy(bl.param_set, e_kin, FT(0), T)
    if !(bl.moisture isa DryModel)
        state.moisture.ρq_tot = FT(0)
    end
end

struct IsentropicVortexReferenceState{FT} <: ReferenceState
    setup::IsentropicVortexSetup{FT}
end
vars_state(::IsentropicVortexReferenceState, ::Auxiliary, FT) =
    @vars(ρ::FT, ρe::FT, p::FT, T::FT)
function atmos_init_aux!(
    atmos::AtmosModel,
    m::IsentropicVortexReferenceState,
    state_auxiliary::MPIStateArray,
    grid,
    direction,
)
    init_state_auxiliary!(
        atmos,
        (args...) -> init_vortex_ref_state!(m, args...),
        state_auxiliary,
        grid,
        direction,
    )
end
function init_vortex_ref_state!(
    m::IsentropicVortexReferenceState,
    atmos::AtmosModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    setup = m.setup
    ρ∞ = setup.ρ∞
    p∞ = setup.p∞
    T∞ = setup.T∞

    aux.ref_state.ρ = ρ∞
    aux.ref_state.p = p∞
    aux.ref_state.T = T∞
    aux.ref_state.ρe = ρ∞ * internal_energy(atmos.param_set, T∞)
end
