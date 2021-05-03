import ClimateMachine.BalanceLaws:
    # declaration
    vars_state,
    # initialization
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    init_state_auxiliary!,
    # rhs computation
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    source!,
    # boundary conditions
    boundary_conditions,
    boundary_state!

struct DryReferenceState{TP}
    temperature_profile::TP
end

"""
    Declaration of state variables

    vars_state returns a NamedTuple of data types.
"""
function vars_state(m::Union{DryAtmosModel,DryAtmosLinearModel}, st::Auxiliary, FT)
    @vars begin
        x::FT
        y::FT
        z::FT
        Φ::FT
        ∇Φ::SVector{3, FT} # TODO: only needed for the linear model
        ref_state::vars_state(m, m.physics.ref_state, st, FT)
    end
end

vars_state(::Union{DryAtmosModel,DryAtmosLinearModel}, ::DryReferenceState, ::Auxiliary, FT) =
    @vars(T::FT, p::FT, ρ::FT, ρe::FT)
vars_state(::Union{DryAtmosModel,DryAtmosLinearModel}, ::NoReferenceState, ::Auxiliary, FT) = @vars()

function vars_state(::Union{DryAtmosModel,DryAtmosLinearModel}, ::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
    end
end

function vars_state(::DryAtmosModel, ::Entropy, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        Φ::FT
    end
end

"""
    Initialization of state variables

    init_state_xyz! sets up the initial fields within our state variables
    (e.g., prognostic, auxiliary, etc.), however it seems to not initialized
    the gradient flux variables by default.
"""
function init_state_prognostic!(
        model::Union{DryAtmosModel,DryAtmosLinearModel},
        # model::DryAtmosLinearModel,
        state::Vars, 
        aux::Vars, 
        localgeo, 
        t
    )
    x = aux.x
    y = aux.y
    z = aux.z

    parameters = model.parameters
    ic = model.initial_conditions

    if !isnothing(ic)
        state.ρ  = ic.ρ(parameters, x, y, z)
        state.ρu = ic.ρu(parameters, x, y, z)
        state.ρe = ic.ρe(parameters, x, y, z)
    end

    return nothing
end

function nodal_init_state_auxiliary!(
    m::Union{DryAtmosModel,DryAtmosLinearModel},
    state_auxiliary,
    tmp,
    geom,
)
    init_state_auxiliary!(m, m.physics.orientation, state_auxiliary, geom)
    init_state_auxiliary!(m, m.physics.ref_state, state_auxiliary, geom)
end

function init_state_auxiliary!(
    ::Union{DryAtmosModel,DryAtmosLinearModel},
    ::SphericalOrientation,
    state_auxiliary,
    geom,
)
    FT = eltype(state_auxiliary)
    _grav = FT(grav(param_set))
    r = norm(geom.coord)
    state_auxiliary.x = geom.coord[1]
    state_auxiliary.y = geom.coord[2]
    state_auxiliary.z = geom.coord[3]
    state_auxiliary.Φ = _grav * r
    state_auxiliary.∇Φ = _grav * geom.coord / r
end

function init_state_auxiliary!(
    ::Union{DryAtmosModel,DryAtmosLinearModel},
    ::NoReferenceState,
    state_auxiliary,
    geom,
) end

function init_state_auxiliary!(
    m::Union{DryAtmosModel,DryAtmosLinearModel},
    ref_state::DryReferenceState,
    state_auxiliary,
    geom,
)
    FT = eltype(state_auxiliary)
    z = altitude(m, m.physics.orientation, geom)
    T, p = ref_state.temperature_profile(param_set, z)

    _R_d::FT = R_d(param_set)
    ρ = p / (_R_d * T)
    Φ = state_auxiliary.Φ
    ρu = SVector{3, FT}(0, 0, 0)

    state_auxiliary.ref_state.T = T
    state_auxiliary.ref_state.p = p
    state_auxiliary.ref_state.ρ = ρ
    state_auxiliary.ref_state.ρe = totalenergy(ρ, ρu, p, Φ)
end

"""
    LHS computations
"""
@inline function flux_first_order!(
    model::Union{DryAtmosModel,DryAtmosLinearModel},
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)

    lhs = model.physics.lhs
    ntuple(Val(length(lhs))) do s
        Base.@_inline_meta
        calc_flux!(flux, lhs[s], state, aux, t)
    end
end

"""
    RHS computations
"""
function source!(m::DryAtmosModel, source, state_prognostic, state_auxiliary, _...)
    sources = m.physics.sources

    ntuple(Val(length(sources))) do s
        Base.@_inline_meta
        calc_force!(source, sources[s], state_prognostic, state_auxiliary)
    end
end

function source!(
    m::DryAtmosLinearModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    ::NTuple{1, Dir},
) where {Dir <: Direction}
    sources = m.physics.sources
    
    ntuple(Val(length(sources))) do s
        Base.@_inline_meta
        calc_force!(source, sources[s], state, aux)
    end    
end

"""
    Boundary conditions
"""
boundary_conditions(model::Union{DryAtmosModel,DryAtmosLinearModel}) = model.boundary_conditions

function boundary_state!(
    ::NumericalFluxFirstOrder,
    bctype,
    ::Union{DryAtmosModel,DryAtmosLinearModel},
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    _...,
)
    state⁺.ρ = state⁻.ρ
    state⁺.ρu -= 2 * dot(state⁻.ρu, n) .* SVector(n)
    state⁺.ρe = state⁻.ρe
    aux⁺.Φ = aux⁻.Φ
end

function boundary_state!(
    nf::NumericalFluxSecondOrder,
    bc,
    lm::Union{DryAtmosModel,DryAtmosLinearModel},
    args...,
)
    nothing
end

"""
    Utils
"""
function vertical_unit_vector(::Union{DryAtmosModel,DryAtmosLinearModel}, aux::Vars)
    FT = eltype(aux)
    aux.∇Φ / FT(grav(param_set))
end

function altitude(::Union{DryAtmosModel,DryAtmosLinearModel}, ::SphericalOrientation, geom)
    FT = eltype(geom)
    _planet_radius::FT = planet_radius(param_set)
    norm(geom.coord) - _planet_radius
end

# function init_state_prognostic!(
#     bl::DryAtmosModel,
#     state,
#     aux,
#     localgeo,
#     t,
# )
#     coords = localgeo.coord
#     FT = eltype(state)

#     # parameters
#     _grav::FT = grav(param_set)
#     _R_d::FT = R_d(param_set)
#     _cv_d::FT = cv_d(param_set)
#     _Ω::FT = Omega(param_set)
#     _a::FT = planet_radius(param_set)
#     _p_0::FT = MSLP(param_set)

#     k::FT = 3
#     T_E::FT = 310
#     T_P::FT = 240
#     T_0::FT = 0.5 * (T_E + T_P)
#     Γ::FT = 0.005
#     A::FT = 1 / Γ
#     B::FT = (T_0 - T_P) / T_0 / T_P
#     C::FT = 0.5 * (k + 2) * (T_E - T_P) / T_E / T_P
#     b::FT = 2
#     H::FT = _R_d * T_0 / _grav
#     z_t::FT = 15e3
#     λ_c::FT = π / 9
#     φ_c::FT = 2 * π / 9
#     d_0::FT = _a / 6
#     V_p::FT = 1
#     M_v::FT = 0.608
#     p_w::FT = 34e3             ## Pressure width parameter for specific humidity
#     η_crit::FT = 10 * _p_0 / p_w ## Critical pressure coordinate
#     q_0::FT = 0                ## Maximum specific humidity (default: 0.018)
#     q_t::FT = 1e-12            ## Specific humidity above artificial tropopause
#     φ_w::FT = 2π / 9           ## Specific humidity latitude wind parameter

#     # grid
#     λ = @inbounds atan(coords[2], coords[1])
#     φ = @inbounds asin(coords[3] / norm(coords, 2))
#     z = norm(coords) - _a

#     r::FT = z + _a
#     γ::FT = 1 # set to 0 for shallow-atmosphere case and to 1 for deep atmosphere case

#     # convenience functions for temperature and pressure
#     τ_z_1::FT = exp(Γ * z / T_0)
#     τ_z_2::FT = 1 - 2 * (z / b / H)^2
#     τ_z_3::FT = exp(-(z / b / H)^2)
#     τ_1::FT = 1 / T_0 * τ_z_1 + B * τ_z_2 * τ_z_3
#     τ_2::FT = C * τ_z_2 * τ_z_3
#     τ_int_1::FT = A * (τ_z_1 - 1) + B * z * τ_z_3
#     τ_int_2::FT = C * z * τ_z_3
#     I_T::FT =
#         (cos(φ) * (1 + γ * z / _a))^k -
#         k / (k + 2) * (cos(φ) * (1 + γ * z / _a))^(k + 2)

#     # base state virtual temperature, pressure, specific humidity, density
#     T_v::FT = (τ_1 - τ_2 * I_T)^(-1)
#     p::FT = _p_0 * exp(-_grav / _R_d * (τ_int_1 - τ_int_2 * I_T))

#     # base state velocity
#     U::FT =
#         _grav * k / _a *
#         τ_int_2 *
#         T_v *
#         (
#             (cos(φ) * (1 + γ * z / _a))^(k - 1) -
#             (cos(φ) * (1 + γ * z / _a))^(k + 1)
#         )
#     u_ref::FT =
#         -_Ω * (_a + γ * z) * cos(φ) +
#         sqrt((_Ω * (_a + γ * z) * cos(φ))^2 + (_a + γ * z) * cos(φ) * U)
#     v_ref::FT = 0
#     w_ref::FT = 0

#     # velocity perturbations
#     F_z::FT = 1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3
#     if z > z_t
#         F_z = FT(0)
#     end
#     d::FT = _a * acos(sin(φ) * sin(φ_c) + cos(φ) * cos(φ_c) * cos(λ - λ_c))
#     c3::FT = cos(π * d / 2 / d_0)^3
#     s1::FT = sin(π * d / 2 / d_0)
#     if 0 < d < d_0 && d != FT(_a * π)
#         u′::FT =
#             -16 * V_p / 3 / sqrt(3) *
#             F_z *
#             c3 *
#             s1 *
#             (-sin(φ_c) * cos(φ) + cos(φ_c) * sin(φ) * cos(λ - λ_c)) /
#             sin(d / _a)
#         v′::FT =
#             16 * V_p / 3 / sqrt(3) * F_z * c3 * s1 * cos(φ_c) * sin(λ - λ_c) /
#             sin(d / _a)
#     else
#         u′ = FT(0)
#         v′ = FT(0)
#     end
#     w′::FT = 0
#     u_sphere = SVector{3, FT}(u_ref + u′, v_ref + v′, w_ref + w′)
#     #u_sphere = SVector{3, FT}(u_ref, v_ref, w_ref)
#     u_cart = sphr_to_cart_vec(u_sphere, φ, λ)

#     ## temperature & density
#     T::FT = I_T
#     ρ::FT = p / (_R_d * T)
#     ## potential & kinetic energy
#     e_pot = aux.Φ
#     e_kin::FT = 0.5 * u_cart' * u_cart
#     e_int = _cv_d * T

#     ## Assign state variables
#     # state.ρ = ρ
#     # if φ == 0.0 && λ == 0.0
#     #     @show _p_0
#     # end
#     state.ρ = I_T
#     state.ρu = ρ * u_cart
#     if total_energy
#         state.ρe = ρ * (e_int + e_kin + e_pot)
#     else
#         state.ρe = ρ * (e_int + e_kin)
#     end

#     nothing
# end

# import ClimateMachine.Orientations: sphr_to_cart_vec
# function sphr_to_cart_vec(vec, lat, lon)
#     FT = eltype(vec)
#     slat, clat = sin(lat), cos(lat)
#     slon, clon = sin(lon), cos(lon)
#     u = MVector{3, FT}(
#         -slon * vec[1] - slat * clon * vec[2] + clat * clon * vec[3],
#         clon * vec[1] - slat * slon * vec[2] + clat * slon * vec[3],
#         clat * vec[2] + slat * vec[3],
#     )
#     return u
# end