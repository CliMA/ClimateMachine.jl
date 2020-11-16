using ..Microphysics_0M
using CLIMAParameters.Planet: Omega, e_int_i0, R_v, cp_v, cp_l, cv_l, cv_i, T_0, LH_v0

export AbstractSource,
    RayleighSponge, GeostrophicForcing, RemovePrecipitation, CreateClouds

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

"""
    RayleighSponge{FT} <: AbstractSource

Rayleigh Damping (Linear Relaxation) for top wall momentum components
Assumes laterally periodic boundary conditions for LES flows. Momentum components
are relaxed to reference values (zero velocities) at the top boundary.
"""
struct RayleighSponge{FT} <: AbstractSource
    "Maximum domain altitude (m)"
    z_max::FT
    "Altitude at with sponge starts (m)"
    z_sponge::FT
    "Sponge Strength 0 ⩽ α_max ⩽ 1"
    α_max::FT
    "Relaxation velocity components"
    u_relaxation::SVector{3, FT}
    "Sponge exponent"
    γ::FT
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
    z = altitude(atmos, aux)
    if z >= s.z_sponge
        r = (z - s.z_sponge) / (s.z_max - s.z_sponge)
        β_sponge = s.α_max * sinpi(r / 2)^s.γ
        source.ρu -= β_sponge * (state.ρu .- state.ρ * s.u_relaxation)
    end
end

"""
    CreateClouds{FT} <: AbstractSource

A source/sink to `q_liq` and `q_ice` implemented as a relaxation towards
equilibrium in the Microphysics module.
The default relaxation timescales are defined in CLIMAParameters.jl.
"""
struct CreateClouds <: AbstractSource end
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
    # get current temperature and phase partition
    FT = eltype(state)
    ts = recover_thermo_state(atmos, state, aux)
    q = PhasePartition(ts)
    T = air_temperature(ts)
#------------    
    _R_v::FT = R_v(atmos.param_set)
    _cp_v::FT = cp_v(atmos.param_set)
    _cp_l::FT = cp_l(atmos.param_set)
    _LH_v0::FT = LH_v0(atmos.param_set)
    q_v = (q.tot - q.liq)
    p_vs = saturation_vapor_pressure(atmos.param_set, T, _LH_v0, _cp_v - _cp_l)
    z = altitude(atmos, aux)
    f1 = (p_vs / (_R_v * state.ρ * T) - q_v) 
    f2 = q.liq
    RelCloud = 1
    S_q_liq = -RelCloud * (f1 + f2 - sqrt(f1 * f1 + f2 * f2))
    S_q_ice = 0
#--------------
    # phase partition corresponding to the current T and q.tot
    # (this is not the same as phase partition from saturation adjustment)
    ts_eq = PhaseEquil_ρTq(atmos.param_set, state.ρ, T, q.tot)

    # cloud condensate as relaxation source terms
#   S_q_liq = conv_q_vap_to_q_liq_ice(atmos.param_set.microphys.liq, q_eq, q)
#   S_q_ice = conv_q_vap_to_q_liq_ice(atmos.param_set.microphys.ice, q_eq, q)

    source.moisture.ρq_liq += state.ρ * S_q_liq
    source.moisture.ρq_ice += state.ρ * S_q_ice
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

        _e_int_i0::FT = e_int_i0(atmos.param_set)
        _cv_l::FT = cv_l(atmos.param_set)
        _cv_i::FT = cv_i(atmos.param_set)
        _T_0::FT = T_0(atmos.param_set)

        S_qt::FT = 0
        if s.use_qc_thr
            S_qt = remove_precipitation(atmos.param_set, q)
        else
            q_vap_sat = q_vap_saturation(ts)
            S_qt = remove_precipitation(atmos.param_set, q, q_vap_sat)
        end

        source.moisture.ρq_tot += state.ρ * S_qt

        source.ρ += state.ρ * S_qt

        source.ρe +=
            (
                λ * _cv_l * (T - _T_0) +
                (1 - λ) * (_cv_i * (T - _T_0) - _e_int_i0) +
                gravitational_potential(atmos.orientation, aux)
            ) *
            state.ρ *
            S_qt
    end
end
