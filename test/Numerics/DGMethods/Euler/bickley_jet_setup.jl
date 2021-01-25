using ClimateMachine
import ClimateMachine.Atmos: atmos_init_aux!, vars_state
using ClimateMachine.Thermodynamics

# Require γ = 2 for BickleyJet problem
CLIMAParameters.Planet.cp_d(ps::AbstractEarthParameterSet) = 717.5060234725578 * 2

Base.@kwdef struct BickleyJetSetup{FT}
    p∞::FT = 10^5
    T∞::FT = 300
    ρ∞::FT = air_density(param_set, FT(T∞), FT(p∞))
    domain_halflength::FT = pi
end

function (setup::BickleyJetSetup)(
    problem,
    bl,
    state,
    aux,
    localgeo,
    t,
    args...,
)
    FT = eltype(state)
    (x,y) = localgeo.coord

    Rd::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    T_ref::FT = T_surf_ref(bl.param_set)
    γ::FT  = c_p / c_v

    k = FT(1/2)
    l = FT(1/2)
    ϵ = FT(0.1)
    ψ₁ = exp(-(y + (l/10))^2/ (2*l^2)) * cos(k*x) * cos(k*y)
    u₀ = sech(y)^2
    v₀ = (k * tan(k*y) + y/l^2)*ψ₁
    u = u₀ + ϵ*u₀
    v = ϵ * v₀ 
        
    u⃗ = SVector{3,FT}(u,v,0)
    T_0 = FT(273.16) # T_ref
    ρ = FT(1.0)
    ts = PhaseDry_ρT(bl.param_set, ρ, T_0)

    e_int = internal_energy(ts)

    state.ρ = ρ
    state.ρu = ρ * u⃗
    e_kin = 0.5 * u⃗' * u⃗ 
    state.ρe = ρ * (e_kin + e_int)
    state.tracers.ρχ = SVector{1,FT}(ρ * FT(sin(y/2)))
end

struct BickleyJetReferenceState{FT} <: ReferenceState
    setup::BickleyJetSetup{FT}
end
vars_state(::BickleyJetReferenceState, ::Auxiliary, FT) =
    @vars(ρ::FT, ρe::FT, p::FT, T::FT)
function atmos_init_aux!(
    atmos::AtmosModel,
    m::BickleyJetReferenceState,
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
    m::BickleyJetReferenceState,
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
