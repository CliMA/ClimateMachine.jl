include("../DryAtmos.jl")

import CLIMAParameters
#const X = 20
const X = 1
CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 6.371229e6 / X
CLIMAParameters.Planet.Omega(::EarthParameterSet) = 7.29212e-5 * X
CLIMAParameters.Planet.MSLP(::EarthParameterSet) = 1e5
CLIMAParameters.Planet.grav(::EarthParameterSet) = 9.80616

struct BaroclinicWave <: AbstractDryAtmosProblem end

vars_state(::DryAtmosModel, ::BaroclinicWave, ::Auxiliary, FT) = @vars(ωk::FT)

function init_state_prognostic!(bl::DryAtmosModel,
                                ::BaroclinicWave,
                                state, aux, localgeo, t)
    coords = localgeo.coord
    FT = eltype(state)

    # parameters
    _grav::FT = grav(param_set)
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _Ω::FT = Omega(param_set)
    _a::FT = planet_radius(param_set)
    _p_0::FT = MSLP(param_set)

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
    M_v::FT = 0.608
    p_w::FT = 34e3             ## Pressure width parameter for specific humidity
    η_crit::FT = 10 * _p_0 / p_w ## Critical pressure coordinate
    q_0::FT = 0                ## Maximum specific humidity (default: 0.018)
    q_t::FT = 1e-12            ## Specific humidity above artificial tropopause
    φ_w::FT = 2π / 9           ## Specific humidity latitude wind parameter

    # grid
    λ = @inbounds atan(coords[2], coords[1])
    φ =  @inbounds asin(coords[3] / norm(coords, 2))
    z =  norm(coords) - _a

    r::FT = z + _a
    γ::FT = 1 # set to 0 for shallow-atmosphere case and to 1 for deep atmosphere case

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

    # base state virtual temperature, pressure, specific humidity, density
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

    # velocity perturbations
    F_z::FT = 1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3
    if z > z_t
        F_z = FT(0)
    end
    d::FT = _a * acos(sin(φ) * sin(φ_c) + cos(φ) * cos(φ_c) * cos(λ - λ_c))
    c3::FT = cos(π * d / 2 / d_0)^3
    s1::FT = sin(π * d / 2 / d_0)
    if 0 < d < d_0 && d != FT(_a * π)
        u′::FT =
            -16 * V_p / 3 / sqrt(3) *
            F_z *
            c3 *
            s1 *
            (-sin(φ_c) * cos(φ) + cos(φ_c) * sin(φ) * cos(λ - λ_c)) /
            sin(d / _a)
        v′::FT =
            16 * V_p / 3 / sqrt(3) * F_z * c3 * s1 * cos(φ_c) * sin(λ - λ_c) /
            sin(d / _a)
    else
        u′ = FT(0)
        v′ = FT(0)
    end
    w′::FT = 0
    u_sphere = SVector{3, FT}(u_ref + u′, v_ref + v′, w_ref + w′)
    u_cart = sphr_to_cart_vec(u_sphere, φ, λ)

    ## temperature & density
    T::FT = T_v
    ρ::FT = p / (_R_d * T)
    ## potential & kinetic energy
    e_pot = aux.Φ
    e_kin::FT = 0.5 * u_cart' * u_cart
    e_int = _cv_d * T

    ## Assign state variables
    state.ρ = ρ
    state.ρu = ρ * u_cart
    state.ρθ = thetadensity(p)
    nothing
end

function sphr_to_cart_vec(
    vec, lat, lon
)
    FT = eltype(vec)
    slat, clat = sin(lat), cos(lat)
    slon, clon = sin(lon), cos(lon)
    u = MVector{3, FT}(
        -slon * vec[1] - slat * clon * vec[2] + clat * clon * vec[3],
        clon * vec[1] - slat * slon * vec[2] + clat * slon * vec[3],
        clat * vec[2] + slat * vec[3],
    )
    return u
end
