
import ClimateMachine.Atmos: atmos_init_aux!, vars_state

Base.@kwdef struct BickleyJetSetup{FT}
    p∞::FT = 10^5
    T∞::FT = 300
    ρ∞::FT = air_density(param_set, FT(T∞), FT(p∞))
    translation_speed::FT = 150
    translation_angle::FT = pi / 4
    vortex_speed::FT = 50
    vortex_radius::FT = 1 // 200
    domain_halflength::FT = 1 // 20
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

    ## Unpack constant parameters
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
     p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    T_ref::FT = T_surf_ref(bl.param_set) # Why T_surf_ref and not T0 ? 

    k = FT(1/2)
    l = FT(1/2)
    ϵ = FT(1/2)
    ψ₁ = exp(-(u + (l/10))^2/ (2*l^2)) * cos(k*x) * cos(k*y)
    u₀ = sech(y)^2
    v₁ = (k * tan(k*y) + y/l^2)*ψ₁
    u = u₀ + ϵ*u₁
    v = ϵ * v₁ 
        
    u⃗ = SVector{2,FT}(u,v)
    
    ρ = FT(1.0)
    ts = PhaseDry_ρT(bl.param_set, ρ, T)

    state.ρ = ρ
    state.ρu = ρ * u⃗
    e_kin = 0.5 * u⃗' * u⃗ 
    state.ρe = ρ * (e_kin + e_int)
    if !(bl.moisture isa DryModel)
        state.moisture.ρq_tot = FT(0)
    end
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
