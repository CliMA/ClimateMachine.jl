using ..Microphysics_0M
using ..Microphysics
using CLIMAParameters.Planet: Omega

export AbstractSource, RemovePrecipitation, CreateClouds, Rain_1M

# sources are applied additively
@generated function atmos_source!(
    stuple::Tuple,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    N = fieldcount(stuple)
    return quote
        Base.Cartesian.@nexprs $N i -> atmos_source!(
            stuple[i],
            atmos,
            source,
            state,
            diffusive,
            aux,
            t,
            direction,
        )
        return nothing
    end
end

abstract type AbstractSource end

function atmos_source!(
    ::Gravity,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    # Migrated to Σsources
end

function atmos_source!(
    ::Coriolis,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    # Migrated to Σsources
end

function atmos_source!(
    subsidence::Subsidence,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    # Migrated to Σsources
end

function atmos_source!(
    s::GeostrophicForcing,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    # Migrated to Σsources
end

function atmos_source!(
    s::RayleighSponge,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    # Migrated to Σsources
end

function atmos_source!(
    ::CreateClouds,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    # Migrated to Σsources
end

"""
    RemovePrecipitation{FT} <: AbstractSource

A sink to `q_tot` when cloud condensate is exceeding a threshold.
The threshold is defined either in terms of condensate or supersaturation.
The removal rate is implemented as a relaxation term
in the Microphysics_0M module.
The default thresholds and timescale are defined in CLIMAParameters.jl.
"""
struct RemovePrecipitation <: AbstractSource
    " Set to true if using qc based threshold"
    use_qc_thr::Bool
end
function atmos_source!(
    s::RemovePrecipitation,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)
    ts = recover_thermo_state(atmos, state, aux)
    if has_condensate(ts)

        q = PhasePartition(ts)
        T::FT = air_temperature(ts)
        λ::FT = liquid_fraction(ts)
        I_l::FT = internal_energy_liquid(ts)
        I_i::FT = internal_energy_ice(ts)
        Φ::FT = gravitational_potential(atmos.orientation, aux)

        S_qt::FT = 0
        if s.use_qc_thr
            S_qt = remove_precipitation(atmos.param_set, q)
        else
            q_vap_sat = q_vap_saturation(ts)
            S_qt = remove_precipitation(atmos.param_set, q, q_vap_sat)
        end

        source.moisture.ρq_tot += state.ρ * S_qt
        source.ρ += state.ρ * S_qt
        source.ρe += (λ * I_l + (1 - λ) * I_i + Φ) * state.ρ * S_qt
    end
end

"""
    Rain_1M{FT} <: AbstractSource

A collection of source/sink terms related to 1-moment rain microphysics.
"""
struct Rain_1M <: AbstractSource end
function atmos_source!(
    ::Rain_1M,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)

    ts = recover_thermo_state(atmos, state, aux)
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

    source.ρ += state.ρ * S_qt
    source.moisture.ρq_tot += state.ρ * S_qt

    if atmos.moisture isa NonEquilMoist
        source.moisture.ρq_liq += state.ρ * S_ql
    end

    source.precipitation.ρq_rai += state.ρ * S_qr

    source.ρe += state.ρ * S_qt * (Φ + I_l)
end
