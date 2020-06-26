import ClimateMachine.ESDGMethods: numerical_volume_fluctuation!, logave, ave

struct DryAtmosphereModel <: BalanceLaw end

const _γ = 7 // 5
function pressure(ρ, ρu, ρe, Φ)
    FT = eltype(ρ)
    γ = FT(_γ)
    (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ - ρ * Φ)
end

function numerical_volume_fluctuation!(
    ::DryAtmosphereModel,
    H,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(H)
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

    α = b_avg * ρ_log / 2b_1

    usq_avg = ave(dot(u_1, u_1), dot(u_2, u_2))

    ρ_log = logave(ρ_1, ρ_2)
    b_log = logave(b_1, b_2)
    γ = FT(_γ)

    Fρ = u_avg' * ρ_log
    Fρu = u_avg * Fρ + ρ_avg / 2b_avg * I
    Fρe = (1 / (2 * (γ - 1) * b_log) - usq_avg / 2 + Φ_avg) * Fρ .+ u_avg' * Fρu

    H.ρ .= Fρ
    H.ρu .= Fρu - α * (Φ_1 - Φ_2)
    H.ρe .= Fρe
end
