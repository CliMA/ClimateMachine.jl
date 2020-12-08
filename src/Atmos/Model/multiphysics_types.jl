##### Multi-physics types

using ..Microphysics_0M
using ..Microphysics

export Subsidence
struct Subsidence{PV, FT} <: TendencyDef{Source, PV}
    D::FT
end

# Subsidence includes tendencies in Mass, Energy and TotalMoisture equations:
Subsidence(D::FT) where {FT} = (
    Subsidence{Mass, FT}(D),
    Subsidence{Energy, FT}(D),
    Subsidence{TotalMoisture, FT}(D),
)

subsidence_velocity(subsidence::Subsidence{PV, FT}, z::FT) where {PV, FT} =
    -subsidence.D * z

struct PressureGradient{PV <: Momentum} <: TendencyDef{Flux{FirstOrder}, PV} end
struct Pressure{PV <: Energy} <: TendencyDef{Flux{FirstOrder}, PV} end

struct Advect{PV} <: TendencyDef{Flux{FirstOrder}, PV} end
struct Diffusion{PV} <: TendencyDef{Flux{SecondOrder}, PV} end


export RemovePrecipitation
"""
    RemovePrecipitation{PV} <: TendencyDef{Source, PV}
A sink to `q_tot` when cloud condensate is exceeding a threshold.
The threshold is defined either in terms of condensate or supersaturation.
The removal rate is implemented as a relaxation term
in the Microphysics_0M module.
The default thresholds and timescale are defined in CLIMAParameters.jl.
"""
struct RemovePrecipitation{PV <: Union{Mass, Energy, TotalMoisture}} <:
       TendencyDef{Source, PV}
    " Set to true if using qc based threshold"
    use_qc_thr::Bool
end

RemovePrecipitation(b::Bool) = (
    RemovePrecipitation{Mass}(b),
    RemovePrecipitation{Energy}(b),
    RemovePrecipitation{TotalMoisture}(b),
)

function compute_precip_params(
    s::RemovePrecipitation{PV},
    aux,
    ts,
) where {PV <: Union{Mass, Energy, TotalMoisture}}
    FT = eltype(aux)
    S_qt::FT = 0
    q = PhasePartition(ts)
    λ::FT = liquid_fraction(ts)
    I_l::FT = internal_energy_liquid(ts)
    I_i::FT = internal_energy_ice(ts)
    Φ::FT = gravitational_potential(atmos.orientation, aux)
    if s.use_qc_thr
        S_qt = remove_precipitation(atmos.param_set, q)
    else
        q_vap_sat = q_vap_saturation(ts)
        S_qt = remove_precipitation(atmos.param_set, q, q_vap_sat)
    end
    return (S_qt = S_qt, λ = λ, I_l = I_l, I_i = I_i, Φ = Φ)
end

export Rain_1M
"""
    Rain_1M{FT} <: AbstractSource
A collection of source/sink terms related to 1-moment rain microphysics.
"""
struct Rain_1M{
    PV <: Union{Mass, Energy, TotalMoisture, LiquidMoisture, Rain},
} <: TendencyDef{Source, PV} end

Rain_1M() = (
    Rain_1M{Mass}(),
    Rain_1M{Energy}(),
    Rain_1M{TotalMoisture}(),
    Rain_1M{LiquidMoisture}(),
    Rain_1M{Rain}(),
)

function compute_rain_params(atmos, state, aux, t, ts)
    FT = eltype(state)

    q = PhasePartition(ts)
    T::FT = air_temperature(ts)
    I_l::FT = internal_energy_liquid(ts)
    q_rai::FT = state.precipitation.ρq_rai / state.ρ
    Φ::FT = gravitational_potential(atmos.orientation, aux)

    # autoconversion
    src_q_rai_acnv = conv_q_liq_to_q_rai(atmos.param_set.microphys.rai, q.liq)
    # accretion
    src_q_rai_accr = accretion(
        atmos.param_set,
        atmos.param_set.microphys.liq,
        atmos.param_set.microphys.rai,
        q.liq,
        q_rai,
        state.ρ,
    )
    # rain evaporation
    src_q_rai_evap = evaporation_sublimation(
        atmos.param_set,
        atmos.param_set.microphys.rai,
        q,
        q_rai,
        state.ρ,
        T,
    )

    S_qr::FT = src_q_rai_acnv + src_q_rai_accr + src_q_rai_evap
    S_ql::FT = -src_q_rai_acnv - src_q_rai_accr
    S_qt::FT = -S_qr

    return (S_qt = S_qt, S_ql = S_ql, S_qr = S_qr, I_l = I_l, Φ = Φ)
end
