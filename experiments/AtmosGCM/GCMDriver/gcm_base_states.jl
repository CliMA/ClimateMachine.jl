# GCM Initial Base State
# This file contains helpers and lists currely avaiable options

abstract type AbstractBaseState end
struct ZeroBaseState <: AbstractBaseState end
struct BCWaveBaseState <: AbstractBaseState end
struct HeldSuarezBaseState <: AbstractBaseState end

# Helper for parsing `--init-base-state`` command line argument
function parse_base_state_arg(arg)
    if arg === nothing
        base_state = nothing
    elseif arg == "bc_wave"
        base_state = BCWaveBaseState()
    elseif arg == "heldsuarez"
        base_state = HeldSuarezBaseState()
    elseif arg == "zero"
        base_state = ZeroBaseState()
    else
        error("unknown base state: " * arg)
    end

    return base_state
end

# Initial base state from rest, independent of the model reference state
function init_base_state(::ZeroBaseState, bl, state, aux, coords, t)
    FT = eltype(state)
    _R_d = R_d(bl.param_set)::FT
    _grav = grav(bl.param_set)::FT
    _a = planet_radius(bl.param_set)::FT
    _p_0::FT = MSLP(bl.param_set)
    T_initial = FT(255)
    r = norm(coords, 2)
    h = r - _a

    scale_height = _R_d * FT(T_initial) / _grav
    p = FT(_p_0) * exp(-h / scale_height)
    u_ref, v_ref, w_ref = (FT(0), FT(0), FT(0))
    return T_initial, p, u_ref, v_ref, w_ref
end

# Initial base state from rest, consistent with the model reference state
function init_base_state(::HeldSuarezBaseState, bl, state, aux, coords, t)
    FT = eltype(state)

    # T_v is dry ref state
    T_v = aux.ref_state.T
    p = aux.ref_state.p
    u_ref, v_ref, w_ref = (FT(0), FT(0), FT(0))

    return T_v, p, u_ref, v_ref, w_ref
end

# Initial base state following
# Ullrich et al. (2016) Dynamical Core Model Intercomparison Project (DCMIP2016) Test Case Document
function init_base_state(::BCWaveBaseState, bl, state, aux, coords, t)
    FT = eltype(state)

    # general parameters
    _grav = grav(bl.param_set)::FT
    _R_d = R_d(bl.param_set)::FT
    _Ω = Omega(bl.param_set)::FT
    _a = planet_radius(bl.param_set)::FT
    _p_0 = MSLP(bl.param_set)::FT

    # grid
    φ = latitude(bl, aux)
    z = altitude(bl, aux)
    γ::FT = 1 # set to 0 for shallow-atmosphere case and to 1 for deep atmosphere case

    # base state parameters
    k::FT = 3
    T_E::FT = 310
    T_P::FT = 240
    T_0::FT = 0.5 * (T_E + T_P)
    Γ::FT = 0.005
    A::FT = 1 / Γ
    B::FT = (T_0 - T_P) / T_0 / T_P
    C::FT = 0.5 * (k + 2) * (T_E - T_P) / T_E / T_P
    b::FT = 2
    H::FT = _R_d * T_0 / _grav
    z_t::FT = 15e3
    λ_c::FT = π / 9
    φ_c::FT = 2 * π / 9
    d_0::FT = _a / 6
    V_p::FT = 1

    # convenience functions for temperature and pressure
    τ_z_1::FT = exp(Γ * z / T_0)
    τ_z_2::FT = 1 - 2 * (z / b / H)^2
    τ_z_3::FT = exp(-(z / b / H)^2)
    τ_1::FT = 1 / T_0 * τ_z_1 + B * τ_z_2 * τ_z_3
    τ_2::FT = C * τ_z_2 * τ_z_3
    τ_int_1::FT = A * (τ_z_1 - 1) + B * z * τ_z_3
    τ_int_2::FT = C * z * τ_z_3
    I_T::FT =
        (cos(φ) * (1 + γ * z / _a))^k -
        k / (k + 2) * (cos(φ) * (1 + γ * z / _a))^(k + 2)

    # base state virtual temperature and pressure
    T_v::FT = (τ_1 - τ_2 * I_T)^(-1)
    p::FT = _p_0 * exp(-_grav / _R_d * (τ_int_1 - τ_int_2 * I_T))

    # base state velocity
    U::FT =
        _grav * k / _a *
        τ_int_2 *
        T_v *
        (
            (cos(φ) * (1 + γ * z / _a))^(k - 1) -
            (cos(φ) * (1 + γ * z / _a))^(k + 1)
        )
    u_ref::FT =
        -_Ω * (_a + γ * z) * cos(φ) +
        sqrt((_Ω * (_a + γ * z) * cos(φ))^2 + (_a + γ * z) * cos(φ) * U)
    v_ref::FT = 0
    w_ref::FT = 0

    return T_v, p, u_ref, v_ref, w_ref
end
