#### Precipitation component in atmosphere model
abstract type PrecipitationModel end

export NoPrecipitation, RainModel, RainSnowModel

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
    ::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
    ts,
    diffusive::Vars,
    hyperdiffusive::Vars,
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
    PrecipitationFlux{PV <: Union{Rain, Snow}} <: TendencyDef{Flux{FirstOrder}, PV}

Computes the precipitation flux as a sum of air velocity and terminal velocity
multiplied by the advected variable.
"""
struct PrecipitationFlux{PV <: Union{Rain, Snow}} <:
       TendencyDef{Flux{FirstOrder}, PV} end

PrecipitationFlux() = (PrecipitationFlux{Rain()}, PrecipitationFlux{Snow()})

function flux(::PrecipitationFlux{Rain}, m, state, aux, t, ts, direction)
    FT = eltype(state)
    u = state.ρu / state.ρ
    q_rai = state.precipitation.ρq_rai / state.ρ

    v_term_rai::FT = FT(0)
    if q_rai > FT(0)
        v_term_rai = terminal_velocity(
            m.param_set,
            m.param_set.microphys.rai,
            state.ρ,
            q_rai,
        )
    end

    k̂ = vertical_unit_vector(m, aux)
    return state.precipitation.ρq_rai * (u - k̂ * v_term_rai)
end
function flux(::PrecipitationFlux{Snow}, m, state, aux, t, ts, direction)
    FT = eltype(state)
    u = state.ρu / state.ρ
    q_sno = state.precipitation.ρq_sno / state.ρ

    v_term_sno::FT = FT(0)
    if q_sno > FT(0)
        v_term_sno = terminal_velocity(
            m.param_set,
            m.param_set.microphys.sno,
            state.ρ,
            q_sno,
        )
    end

    k̂ = vertical_unit_vector(m, aux)
    return state.precipitation.ρq_sno * (u - k̂ * v_term_sno)
end

"""
    NoPrecipitation <: PrecipitationModel

No precipitation.
"""
struct NoPrecipitation <: PrecipitationModel end

"""
    RainModel <: PrecipitationModel

Precipitation model with rain.
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
    diffusive.precipitation.∇q_rai = ∇transform.precipitation.q_rai
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

function source!(m::RainModel, source::Vars, atmos::AtmosModel, args)
    tend = Source()
    source.precipitation.ρq_rai =
        Σsources(eq_tends(Rain(), atmos, tend), atmos, args)
end

"""
    RainSnowModel <: PrecipitationModel

Precipitation model with rain and snow.
"""
struct RainSnowModel <: PrecipitationModel end

vars_state(::RainSnowModel, ::Prognostic, FT) = @vars(ρq_rai::FT, ρq_sno::FT)
vars_state(::RainSnowModel, ::Gradient, FT) = @vars(q_rai::FT, q_sno::FT)
vars_state(::RainSnowModel, ::GradientFlux, FT) =
    @vars(∇q_rai::SVector{3, FT}, ∇q_sno::SVector{3, FT})

function atmos_nodal_update_auxiliary_state!(
    precip::RainSnowModel,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end

function compute_gradient_argument!(
    precip::RainSnowModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.precipitation.q_rai = state.precipitation.ρq_rai * ρinv
    transform.precipitation.q_sno = state.precipitation.ρq_sno * ρinv
end

function compute_gradient_flux!(
    precip::RainSnowModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.precipitation.∇q_rai = ∇transform.precipitation.q_rai
    diffusive.precipitation.∇q_sno = ∇transform.precipitation.q_sno
end

function flux_first_order!(
    precip::RainSnowModel,
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
    flux.precipitation.ρq_sno = Σfluxes(eq_tends(Snow(), atmos, tend), args...)
end

function flux_second_order!(
    precip::RainSnowModel,
    flux::Grad,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
    ts,
    diffusive::Vars,
    hyperdiffusive::Vars,
)
    tend = Flux{SecondOrder}()
    args = (atmos, state, aux, t, ts, diffusive, hyperdiffusive)
    flux.precipitation.ρq_rai = Σfluxes(eq_tends(Rain(), atmos, tend), args...)
    flux.precipitation.ρq_sno = Σfluxes(eq_tends(Snow(), atmos, tend), args...)
end

function source!(m::RainSnowModel, source::Vars, atmos::AtmosModel, args)
    tend = Source()
    source.precipitation.ρq_rai =
        Σsources(eq_tends(Rain(), atmos, tend), atmos, args)
    source.precipitation.ρq_sno =
        Σsources(eq_tends(Snow(), atmos, tend), atmos, args)
end

#####
##### Tendency specifications
#####

eq_tends(pv::PV, ::RainModel, ::Flux{FirstOrder}) where {PV <: Rain} =
    (PrecipitationFlux{PV}(),)
eq_tends(pv::PV, ::RainModel, ::Flux{SecondOrder}) where {PV <: Rain} =
    (Diffusion{PV}(),)

eq_tends(
    pv::PV,
    ::RainSnowModel,
    ::Flux{FirstOrder},
) where {PV <: Precipitation} = (PrecipitationFlux{PV}(),)
eq_tends(
    pv::PV,
    ::RainSnowModel,
    ::Flux{SecondOrder},
) where {PV <: Precipitation} = (Diffusion{PV}(),)
