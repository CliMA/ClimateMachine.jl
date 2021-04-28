using ClimateMachine.DGMethods.NumericalFluxes: NumericalFluxFirstOrder
import ClimateMachine.DGMethods.NumericalFluxes:
    EntropyConservative,
    numerical_volume_conservative_flux_first_order!,
    numerical_volume_fluctuation_flux_first_order!,
    ave,
    logave,
    numerical_flux_first_order!,
    numerical_flux_second_order!,
    numerical_boundary_flux_second_order!
import ClimateMachine.BalanceLaws:
    entropy_variables_to_state!,
    state_to_entropy,
    wavespeed

@inline gamma(ps::EarthParameterSet) = cp_d(ps) / cv_d(ps)

function numerical_volume_conservative_flux_first_order!(
    ::EntropyConservative,
    ::DryAtmosModel,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(F)
    ρ_1, ρu_1, ρe_1 = state_1.ρ, state_1.ρu, state_1.ρe
    ρ_2, ρu_2, ρe_2 = state_2.ρ, state_2.ρu, state_2.ρe
    Φ_1, Φ_2 = aux_1.Φ, aux_2.Φ
    u_1 = ρu_1 / ρ_1
    u_2 = ρu_2 / ρ_2
    p_1 = pressure(ρ_1, ρu_1, ρe_1, Φ_1)
    p_2 = pressure(ρ_2, ρu_2, ρe_2, Φ_2)
    b_1 = ρ_1 / 2p_1
    b_2 = ρ_2 / 2p_2

    ρ_avg = ave(ρ_1, ρ_2)
    u_avg = ave(u_1, u_2)
    b_avg = ave(b_1, b_2)
    Φ_avg = ave(Φ_1, Φ_2)

    usq_avg = ave(dot(u_1, u_1), dot(u_2, u_2))

    ρ_log = logave(ρ_1, ρ_2)
    b_log = logave(b_1, b_2)
    α = b_avg * ρ_log / 2b_1

    γ = FT(gamma(param_set))

    Fρ = u_avg * ρ_log
    Fρu = u_avg * Fρ' + ρ_avg / 2b_avg * I
    if total_energy
        Fρe =
            (1 / (2 * (γ - 1) * b_log) - usq_avg / 2 + Φ_avg) * Fρ + Fρu * u_avg
    else
        Fρe = (1 / (2 * (γ - 1) * b_log) - usq_avg / 2) * Fρ + Fρu * u_avg
    end

    F.ρ += Fρ
    F.ρu += Fρu
    F.ρe += Fρe
end

function numerical_volume_fluctuation_flux_first_order!(
    ::NumericalFluxFirstOrder,
    ::DryAtmosModel,
    D::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    if fluctuation_gravity
        FT = eltype(D)
        ρ_1, ρu_1, ρe_1 = state_1.ρ, state_1.ρu, state_1.ρe
        ρ_2, ρu_2, ρe_2 = state_2.ρ, state_2.ρu, state_2.ρe
        Φ_1, Φ_2 = aux_1.Φ, aux_2.Φ
        p_1 = pressure(ρ_1, ρu_1, ρe_1, Φ_1)
        p_2 = pressure(ρ_2, ρu_2, ρe_2, Φ_2)
        b_1 = ρ_1 / 2p_1
        b_2 = ρ_2 / 2p_2

        ρ_log = logave(ρ_1, ρ_2)
        b_avg = ave(b_1, b_2)
        α = b_avg * ρ_log / 2b_1

        D.ρu -= α * (Φ_1 - Φ_2) * I
    end
end

struct CentralVolumeFlux <: NumericalFluxFirstOrder end
function numerical_volume_conservative_flux_first_order!(
    ::CentralVolumeFlux,
    m::DryAtmosModel,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(F)
    F_1 = similar(F)
    flux_first_order!(m, F_1, state_1, aux_1, FT(0), EveryDirection())

    F_2 = similar(F)
    flux_first_order!(m, F_2, state_2, aux_2, FT(0), EveryDirection())

    parent(F) .= (parent(F_1) .+ parent(F_2)) ./ 2
end

struct KGVolumeFlux <: NumericalFluxFirstOrder end
function numerical_volume_conservative_flux_first_order!(
    ::KGVolumeFlux,
    ::DryAtmosModel,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)

    Φ_1 = aux_1.Φ
    ρ_1 = state_1.ρ
    ρu_1 = state_1.ρu
    ρe_1 = state_1.ρe
    u_1 = ρu_1 / ρ_1
    e_1 = ρe_1 / ρ_1
    p_1 = pressure(ρ_1, ρu_1, ρe_1, Φ_1)

    Φ_2 = aux_2.Φ
    ρ_2 = state_2.ρ
    ρu_2 = state_2.ρu
    ρe_2 = state_2.ρe
    u_2 = ρu_2 / ρ_2
    e_2 = ρe_2 / ρ_2
    p_2 = pressure(ρ_2, ρu_2, ρe_2, Φ_2)

    ρ_avg = ave(ρ_1, ρ_2)
    u_avg = ave(u_1, u_2)
    e_avg = ave(e_1, e_2)
    p_avg = ave(p_1, p_2)

    F.ρ = ρ_avg * u_avg
    F.ρu = p_avg * I + ρ_avg * u_avg .* u_avg'
    F.ρe = ρ_avg * u_avg * e_avg + p_avg * u_avg
end

struct EntropyConservativeWithPenalty <: NumericalFluxFirstOrder end
function numerical_flux_first_order!(
    numerical_flux::EntropyConservativeWithPenalty,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    FT = eltype(fluxᵀn)

    numerical_flux_first_order!(
        EntropyConservative(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    fluxᵀn = parent(fluxᵀn)

    wavespeed⁻ = wavespeed(
        balance_law,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )
    wavespeed⁺ = wavespeed(
        balance_law,
        normal_vector,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    max_wavespeed = max.(wavespeed⁻, wavespeed⁺)
    penalty =
        max_wavespeed .* (parent(state_prognostic⁻) - parent(state_prognostic⁺))

    fluxᵀn .+= penalty / 2
end

Base.@kwdef struct MatrixFlux{FT} <: NumericalFluxFirstOrder
    Mcut::FT = 0
    low_mach::Bool = false
    kinetic_energy_preserving::Bool = false
end

function numerical_flux_first_order!(
    numerical_flux::MatrixFlux,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    FT = eltype(fluxᵀn)
    numerical_flux_first_order!(
        EntropyConservative(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    fluxᵀn = parent(fluxᵀn)

    γ = FT(gamma(param_set))

    low_mach = numerical_flux.low_mach
    Mcut = numerical_flux.Mcut
    kinetic_energy_preserving = numerical_flux.kinetic_energy_preserving

    ω = FT(π) / 3
    δ = FT(π) / 5
    random_unit_vector = SVector(sin(ω) * cos(δ), cos(ω) * cos(δ), sin(δ))
    # tangent space basis
    τ1 = random_unit_vector × normal_vector
    τ2 = τ1 × normal_vector

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.ρe

    Φ⁻ = state_auxiliary⁻.Φ
    u⁻ = ρu⁻ / ρ⁻
    p⁻ = pressure(ρ⁻, ρu⁻, ρe⁻, Φ⁻)
    β⁻ = ρ⁻ / 2p⁻

    Φ⁺ = state_auxiliary⁺.Φ
    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.ρe

    u⁺ = ρu⁺ / ρ⁺
    p⁺ = pressure(ρ⁺, ρu⁺, ρe⁺, Φ⁺)
    β⁺ = ρ⁺ / 2p⁺

    ρ_log = logave(ρ⁻, ρ⁺)
    β_log = logave(β⁻, β⁺)
    if total_energy
        Φ_avg = ave(Φ⁻, Φ⁺)
    else
        Φ_avg = 0
    end
    u_avg = ave.(u⁻, u⁺)
    p_avg = ave(ρ⁻, ρ⁺) / 2ave(β⁻, β⁺)
    u²_bar = 2 * sum(u_avg .^ 2) - sum(ave(u⁻ .^ 2, u⁺ .^ 2))

    h_bar = γ / (2 * β_log * (γ - 1)) + u²_bar / 2 + Φ_avg
    c_bar = sqrt(γ * p_avg / ρ_log)

    umc = u_avg - c_bar * normal_vector
    upc = u_avg + c_bar * normal_vector
    u_avgᵀn = u_avg' * normal_vector
    R = hcat(
        SVector(1, umc[1], umc[2], umc[3], h_bar - c_bar * u_avgᵀn),
        SVector(1, u_avg[1], u_avg[2], u_avg[3], u²_bar / 2 + Φ_avg),
        SVector(0, τ1[1], τ1[2], τ1[3], τ1' * u_avg),
        SVector(0, τ2[1], τ2[2], τ2[3], τ2' * u_avg),
        SVector(1, upc[1], upc[2], upc[3], h_bar + c_bar * u_avgᵀn),
    )

    if low_mach
        M = abs(u_avg' * normal_vector) / c_bar
        c_bar *= max(min(M, FT(1)), Mcut)
    end

    if kinetic_energy_preserving
        λl = abs(u_avgᵀn) + c_bar
        λr = λl
    else
        λl = abs(u_avgᵀn - c_bar)
        λr = abs(u_avgᵀn + c_bar)
    end

    Λ = SDiagonal(λl, abs(u_avgᵀn), abs(u_avgᵀn), abs(u_avgᵀn), λr)
    #Ξ = sqrt(abs((p⁺ - p⁻) / (p⁺ + p⁻)))
    #Λ = Ξ * abs(u_avgᵀn + c_bar) * I + (1 - Ξ) * ΛM

    T = SDiagonal(ρ_log / 2γ, ρ_log * (γ - 1) / γ, p_avg, p_avg, ρ_log / 2γ)

    entropy⁻ = similar(parent(state_prognostic⁻), Size(6))
    state_to_entropy_variables!(
        balance_law,
        Vars{vars_state(balance_law, Entropy(), FT)}(entropy⁻),
        state_prognostic⁻,
        state_auxiliary⁻,
    )

    entropy⁺ = similar(parent(state_prognostic⁺), Size(6))
    state_to_entropy_variables!(
        balance_law,
        Vars{vars_state(balance_law, Entropy(), FT)}(entropy⁺),
        state_prognostic⁺,
        state_auxiliary⁺,
    )

    Δentropy = parent(entropy⁺) - parent(entropy⁻)

    fluxᵀn .-= R * Λ * T * R' * Δentropy[SOneTo(5)] / 2
end

"""
    state_to_entropy_variables!(
        ::DryAtmosModel,
        entropy::Vars,
        state::Vars,
        aux::Vars,
    )

See [`BalanceLaws.state_to_entropy_variables!`](@ref)
"""
function state_to_entropy_variables!(
    ::DryAtmosModel,
    entropy::Vars,
    state::Vars,
    aux::Vars,
)
    ρ, ρu, ρe, Φ = state.ρ, state.ρu, state.ρe, aux.Φ

    FT = eltype(state)
    γ = FT(gamma(param_set))

    p = pressure(ρ, ρu, ρe, Φ)
    s = log(p / ρ^γ)
    b = ρ / 2p
    u = ρu / ρ

    if total_energy
        entropy.ρ = (γ - s) / (γ - 1) - (dot(u, u) - 2Φ) * b
    else
        entropy.ρ = (γ - s) / (γ - 1) - (dot(u, u)) * b
    end
    entropy.ρu = 2b * u
    entropy.ρe = -2b
    entropy.Φ = 2ρ * b
end

"""
    entropy_variables_to_state!(
        ::DryAtmosModel,
        state::Vars,
        aux::Vars,
        entropy::Vars,
    )

See [`BalanceLaws.entropy_variables_to_state!`](@ref)
"""
function entropy_variables_to_state!(
    ::DryAtmosModel,
    state::Vars,
    aux::Vars,
    entropy::Vars,
)
    FT = eltype(state)
    β = entropy
    γ = FT(gamma(param_set))

    b = -β.ρe / 2
    ρ = β.Φ / (2b)
    ρu = ρ * β.ρu / (2b)

    p = ρ / (2b)
    s = log(p / ρ^γ)
    Φ = dot(ρu, ρu) / (2 * ρ^2) - ((γ - s) / (γ - 1) - β.ρ) / (2b)

    ρe = p / (γ - 1) + dot(ρu, ρu) / (2ρ) + ρ * Φ

    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe
    aux.Φ = Φ
end

function state_to_entropy(::DryAtmosModel, state::Vars, aux::Vars)
    FT = eltype(state)
    ρ, ρu, ρe, Φ = state.ρ, state.ρu, state.ρe, aux.Φ
    p = pressure(ρ, ρu, ρe, Φ)
    γ = FT(gamma(param_set))
    s = log(p / ρ^γ)
    η = -ρ * s / (γ - 1)
    return η
end

# Numerical Flux 
#=
numerical_flux_first_order!(::Nothing, _...) = nothing
numerical_boundary_flux_first_order!(::Nothing, _...) = nothing
numerical_flux_second_order!(::Nothing, _...) = nothing
numerical_boundary_flux_second_order!(::Nothing, _...) = nothing
=#
numerical_flux_first_order!(::Nothing, ::DryAtmosModel, _...) = nothing
numerical_flux_second_order!(::Nothing, ::DryAtmosModel, _...) = nothing
numerical_boundary_flux_second_order!(::Nothing, a, ::DryAtmosModel, _...) = nothing

function wavespeed(
    lm::DryAtmosLinearModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ref = aux.ref_state
    return soundspeed(ref.ρ, ref.p)
end

function wavespeed(
    ::DryAtmosModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ = aux.Φ
    p = pressure(ρ, ρu, ρe, Φ)

    u = ρu / ρ
    uN = abs(dot(nM, u))
    return uN + soundspeed(ρ, p)
end