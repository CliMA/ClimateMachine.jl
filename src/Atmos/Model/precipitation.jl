#### Precipitation component in atmosphere model
abstract type PrecipitationModel end

export NoPrecipitation, Rain#, RainSnow

using ..Microphysics

vars_state(::PrecipitationModel, ::AbstractStateType, FT) = @vars()

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
) end
function flux_second_order!(
    ::PrecipitationModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
) end
function flux_first_order!(::PrecipitationModel, _...) end
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

vars_state(::Rain, ::Prognostic, FT) = @vars(ρq_rai::FT)
vars_state(::Rain, ::Gradient, FT) = @vars(q_rai::FT)
vars_state(::Rain, ::GradientFlux, FT) = @vars(∇q_rai::SVector{3, FT})

function atmos_nodal_update_auxiliary_state!(
    precip::Rain,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end

function compute_gradient_argument!(
    precip::Rain,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.precipitation.q_rai = state.precipitation.ρq_rai * ρinv
end

function compute_gradient_flux!(
    precip::Rain,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    # diffusive flux of q_tot
    diffusive.precipitation.∇q_rai = ∇transform.precipitation.q_rai
end

function flux_precipitation!(
    precip::Rain,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    u = state.ρu / state.ρ
    q_rai = state.precipitation.ρq_rai / state.ρ

    v_term::FT = FT(0)
    if q_rai > FT(0)
        v_term = terminal_velocity(
            atmos.param_set,
            atmos.param_set.microphys.rai,
            state.ρ,
            q_rai,
        )
    end

    k̂ = vertical_unit_vector(atmos, aux)
    flux.precipitation.ρq_rai += state.precipitation.ρq_rai * (u - k̂ * v_term)
end

function flux_second_order!(
    precip::Rain,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    d_q_rai = (-D_t) .* diffusive.precipitation.∇q_rai
    flux_second_order!(precip, flux, state, d_q_rai)
end
function flux_second_order!(precip::Rain, flux::Grad, state::Vars, d_q_rai)
    flux.precipitation.ρq_rai += d_q_rai * state.ρ
end
