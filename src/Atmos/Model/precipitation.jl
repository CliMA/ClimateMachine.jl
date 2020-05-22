#### Precipitation component in atmosphere model
abstract type PrecipitationModel end

export NoPrecipitation, Rain

using ..Microphysics

vars_state_conservative(::PrecipitationModel, FT) = @vars()
vars_state_gradient(::PrecipitationModel, FT) = @vars()
vars_state_gradient_flux(::PrecipitationModel, FT) = @vars()
vars_state_auxiliary(::PrecipitationModel, FT) = @vars()

function atmos_nodal_update_auxiliary_state!(
    ::PrecipitationModel,
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function flux_precipitation!(
    ::PrecipitationModel,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function compute_gradient_flux!(
    ::PrecipitationModel,
    diffusive,
    ∇transform,
    state,
    aux,
    t,
    ρD_t,
) end
function flux_second_order!(
    ::PrecipitationModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
) end
function flux_first_order!(
    ::PrecipitationModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function compute_gradient_argument!(
    ::PrecipitationModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) end

"""
    NoPrecipitation <: PrecipitationModel

No precipitation.
"""
struct NoPrecipitation <: PrecipitationModel end


"""
    Rain <: PrecipitationModel

Precipitation model with rain only.
"""
struct Rain <: PrecipitationModel end

vars_state_conservative(::Rain, FT) = @vars(ρq_rain::FT)
vars_state_gradient(::Rain, FT) = @vars(q_rain::FT)
vars_state_gradient_flux(::Rain, FT) = @vars(ρd_q_rain::SVector{3, FT})
vars_state_auxiliary(::Rain, FT) =
    @vars(terminal_velocity::FT, src_q_rai_tot::FT)

function atmos_nodal_update_auxiliary_state!(
    rain::Rain,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    q_rain = state.precipitation.ρq_rain / state.ρ
    if q_rain > FT(0) # TODO - need a way to prevent negative values
        aux.precipitation.terminal_velocity = terminal_velocity(q_rain, state.ρ)
    else
        aux.precipitation.terminal_velocity = FT(0)
    end

    ρ = state.ρ
    ρinv = 1 / state.ρ
    p = aux.pressure
    # TODO - ensure positive definite
    q_tot = max(FT(0), state.ρq_tot * ρinv)
    q_rai = max(FT(0), state.ρq_rain * ρinv)

    # current state
    ts = thermo_state(rain, state, aux)
    # q     = PhasePartition(q_tot, q_liq, q_ice)
    q = PhasePartition(ts)
    # T     = air_temperature(e_int, q)
    T = air_temperature(ts)
    # equilibrium state at current T
    q_eq = PhasePartition_equil(ts)

    # tendency from cloud water condensation/evaporation
    # src_q_liq = conv_q_vap_to_q_liq(q_eq, q)# TODO - temporary handling ice

    # tendencies from rain
    src_q_rai_evap = conv_q_rai_to_q_vap(atmos.param_set, q_rai, q, T, p, ρ)
    src_q_rai_acnv = conv_q_liq_to_q_rai_acnv(atmos.param_set, q.liq)
    src_q_rai_accr = conv_q_liq_to_q_rai_accr(atmos.param_set, q.liq, q_rai, ρ)

    aux.precipitation.src_q_rai_tot =
        src_q_rai_acnv + src_q_rai_accr + src_q_rai_evap
end

function compute_gradient_argument!(
    rain::Rain,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.precipitation.q_rain = state.precipitation.ρq_rain / state.ρ
end

function compute_gradient_flux!(
    rain::Rain,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    ρν::Union{Real, AbstractMatrix},
    D_T,
)
    # diffusive flux
    diffusive.precipitation.ρd_q_rain =
        state.ρ .* (-D_T) .* ∇transform.precipitation.q_rain
end

function flux_precipitation!(
    rain::Rain,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    u = state.ρu / state.ρ
    flux.precipitation.ρq_rain += state.precipitation.ρq_rain * u
end

function flux_second_order!(
    rain::Rain,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_T,
)
    flux.precipitation.ρq_rain = diffusive.precipitation.ρd_q_rain
end

function boundarycondition_precipitation!(
    rain::Rain,
    stateP::Vars,
    diffP::Vars,
    auxP::Vars,
    nM,
    stateM::Vars,
    diffM::Vars,
    auxM::Vars,
    bctype::BC,
    t,
) where {BC}
    stateP.precipitation.ρq_rain = eltype(stateP)(0)
end
