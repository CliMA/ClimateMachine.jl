using ClimateMachine.Microphysics_0M
using CLIMAParameters.Planet: Omega, e_int_i0, cv_d, cv_l, cv_i, T_0

using Printf

export Source, Gravity, RayleighSponge, Subsidence, GeostrophicForcing, Coriolis, RemovePrecipitation, NudgeToSaturation

# kept for compatibility
# can be removed if no functions are using this
function atmos_source!(
    f::Function,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    f(atmos, source, state, diffusive, aux, t, direction)
end
function atmos_source!(
    ::Nothing,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) end
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

abstract type Source end

struct Gravity <: Source end
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
    if atmos.ref_state isa HydrostaticState
        source.ρu -= (state.ρ - aux.ref_state.ρ) * aux.orientation.∇Φ
    else
        source.ρu -= state.ρ * aux.orientation.∇Φ
    end
end

struct Coriolis <: Source end
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
    FT = eltype(state)
    _Omega::FT = Omega(atmos.param_set)
    # note: this assumes a SphericalOrientation
    source.ρu -= SVector(0, 0, 2 * _Omega) × state.ρu
end

struct Subsidence{FT} <: Source
    D::FT
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
    ρ = state.ρ
    z = altitude(atmos, aux)
    w_sub = subsidence_velocity(subsidence, z)
    k̂ = vertical_unit_vector(atmos, aux)

    source.ρe -= ρ * w_sub * dot(k̂, diffusive.∇h_tot)
    source.moisture.ρq_tot -= ρ * w_sub * dot(k̂, diffusive.moisture.∇q_tot)
end

subsidence_velocity(subsidence::Subsidence{FT}, z::FT) where {FT} =
    -subsidence.D * z


struct GeostrophicForcing{FT} <: Source
    f_coriolis::FT
    u_geostrophic::FT
    v_geostrophic::FT
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
    u_geo = SVector(s.u_geostrophic, s.v_geostrophic, 0)
    ẑ = vertical_unit_vector(atmos, aux)
    fkvector = s.f_coriolis * ẑ
    source.ρu -= fkvector × (state.ρu .- state.ρ * u_geo)
end

"""
    RayleighSponge{FT} <: Source

Rayleigh Damping (Linear Relaxation) for top wall momentum components
Assumes laterally periodic boundary conditions for LES flows. Momentum components
are relaxed to reference values (zero velocities) at the top boundary.
"""
struct RayleighSponge{FT} <: Source
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

struct RemovePrecipitation <: Source end
function atmos_source!(
    ::RemovePrecipitation,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    # TODO - should I be using aux here? Or do another saturation adjustement?
    FT = eltype(state)
    #@info @sprintf("""some parameter info: %s""", eps(FT))
    if aux.moisture.q_liq + aux.moisture.q_ice > eps(FT) #2e-16

        _e_int_i0::FT = e_int_i0(atmos.param_set)
        _cv_d::FT = cv_d(atmos.param_set)
        _cv_l::FT = cv_l(atmos.param_set)
        _cv_i::FT = cv_i(atmos.param_set)
        _T_0::FT = T_0(atmos.param_set)

        q = PhasePartition(state.moisture.ρq_tot / state.ρ, aux.moisture.q_liq, aux.moisture.q_ice)
        T::FT = aux.moisture.temperature
        
        #@info @sprintf("""qliq info: %s""", aux.moisture.q_liq )
        #@info @sprintf("""qice info: %s""", aux.moisture.q_ice )
        dqt_dt::FT = remove_precipitation(atmos.param_set, q)

        source.moisture.ρq_tot += state.ρ * dqt_dt

        source.ρ  += state.ρ / (FT(1) - q.tot) * dqt_dt

        source.ρe += (q.liq / (q.liq + q.ice) * (_cv_l - _cv_d) * (T - _T_0)
                      +
                      q.ice / (q.liq + q.ice) * ((_cv_i - _cv_d) * (T - _T_0) - _e_int_i0)) * state.ρ * dqt_dt
    end
end

"""
    NudgeToSaturation{FT} <: Source

Linear damping of moisture q_tot to RH = 100% at the surface, where RH = q_vap / q_vap_saturation
Parametrises bottom surface moisture fluxes. 
    Following Ming and Held (2018): 
"""
struct NudgeToSaturation <: Source end
function atmos_source!(
    ::NudgeToSaturation,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    # constants
    FT = eltype(state)
    tau_n::FT = 30 * 600 # nudging timescale (default = 30 mins)
    phase_type = PhaseEquil # this may need to be generalised
    #H::FT = 2000 # heoght below which this is applied
    p_nudge::FT = 85000 # pressure (Pa) above which this fuctions in pplied 

    # get relaxation 
    #z = altitude(atmos, aux)
    T = aux.moisture.temperature
    p = air_pressure(atmos.param_set, T, state.ρ)
    
    q = PhasePartition(state.moisture.ρq_tot / state.ρ, aux.moisture.q_liq, aux.moisture.q_ice)
    q_vap = vapor_specific_humidity(q)
    q_vap_sat = q_vap_saturation(atmos.param_set, T, state.ρ , phase_type)
    
    if p >= p_nudge
      dqt_dt = - ( q_vap - q_vap_sat ) / tau_n 
    else
      dqt_dt = FT(0)
    end
    # apply the moisture source 
    source.moisture.ρq_tot += state.ρ * dqt_dt

    ## apply changes to density an energy due to mass addition
    #@info @sprintf("""some parameter info: %s""", eps(FT))
    #source.ρ  += state.ρ / (FT(1) - q.tot) * dqt_dt
    #source.ρe += (q.liq / (q.liq + q.ice) * (_cv_l - _cv_d) * (T - _T_0)
    #                  +
    #                  q.ice / (q.liq + q.ice) * ((_cv_i - _cv_d) * (T - _T_0) - _e_int_i0)) * state.ρ * dqt_dt
end




