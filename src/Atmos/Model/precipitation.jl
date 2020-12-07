#### Precipitation component in atmosphere model
abstract type PrecipitationModel end

export NoPrecipitation, RainModel#, RainSnow

eq_tends(pv::PV, m::PrecipitationModel, ::Flux{O}) where {PV, O} = ()

using ..Microphysics

vars_state(::PrecipitationModel, ::AbstractStateType, FT) = @vars()

function atmos_nodal_update_auxiliary_state!(
    ::PrecipitationModel,
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function flux_first_order!(
    ::PrecipitationModel,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    ts,
    direction,
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

source!(::PrecipitationModel, args...) = nothing

"""
    NoPrecipitation <: PrecipitationModel

No precipitation.
"""
struct NoPrecipitation <: PrecipitationModel end


"""
    RainModel <: PrecipitationModel

Precipitation model with rain only.
"""
struct RainModel <: PrecipitationModel end

vars_state(::RainModel, ::Prognostic, FT) = @vars(ρq_rai::FT)
vars_state(::RainModel, ::Gradient, FT) = @vars(q_rai::FT)
vars_state(::RainModel, ::GradientFlux, FT) = @vars(∇q_rai::SVector{3, FT})

function atmos_nodal_update_auxiliary_state!(
    precip::RainModel,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end

function compute_gradient_argument!(
    precip::RainModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.precipitation.q_rai = state.precipitation.ρq_rai * ρinv
end

function compute_gradient_flux!(
    precip::RainModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    # diffusive flux of q_tot
    diffusive.precipitation.∇q_rai = ∇transform.precipitation.q_rai
end

"""
    RainFlux{PV} <: TendencyDef{Flux{FirstOrder}, PV <: Rain}

Computes the rain flux as a sum of air velocity and rain terminal velocity
multiplied by the advected variable
"""
struct RainFlux{PV <: Rain} <: TendencyDef{Flux{FirstOrder}, PV} end

function flux(::RainFlux{Rain}, m, state, aux, t, ts, direction)
    FT = eltype(state)
    u = state.ρu / state.ρ
    q_rai = state.precipitation.ρq_rai / state.ρ

    v_term::FT = FT(0)
    if q_rai > FT(0)
        v_term = terminal_velocity(
            m.param_set,
            m.param_set.microphys.rai,
            state.ρ,
            q_rai,
        )
    end

    k̂ = vertical_unit_vector(m, aux)
    return state.precipitation.ρq_rai * (u - k̂ * v_term)
end

function flux_first_order!(
    precip::RainModel,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    ts,
    direction,
)
    tend = Flux{FirstOrder}()
    args = (atmos, state, aux, t, ts, direction)
    flux.precipitation.ρq_rai = Σfluxes(eq_tends(Rain(), atmos, tend), args...)
end

function flux_second_order!(
    precip::RainModel,
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
function flux_second_order!(precip::RainModel, flux::Grad, state::Vars, d_q_rai)
    flux.precipitation.ρq_rai += d_q_rai * state.ρ
end

function source!(
    m::RainModel,
    source::Vars,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
    ts,
    direction,
    diffusive::Vars,
)
    tend = Source()
    args = (atmos, state, aux, t, ts, direction, diffusive)
    source.precipitation.ρq_rai =
        Σsources(eq_tends(Rain(), atmos, tend), args...)
end

#####
##### Tendency specifications
#####

eq_tends(pv::PV, ::RainModel, ::Flux{FirstOrder}) where {PV <: Rain} =
    (RainFlux{PV}(),)
