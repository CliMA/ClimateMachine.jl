#### Precipitation component in atmosphere model
abstract type PrecipitationModel end

export PrecipitationModel, NoPrecipitation, RainModel, RainSnowModel

eq_tends(pv::PV, m::PrecipitationModel, ::Flux{O}) where {PV, O} = ()

vars_state(::PrecipitationModel, ::AbstractStateType, FT) = @vars()

function atmos_nodal_update_auxiliary_state!(
    ::PrecipitationModel,
    m::AtmosModel,
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

precompute(::NoPrecipitation, atmos::AtmosModel, args, ts, ::Source) =
    NamedTuple()

"""
    RainModel <: PrecipitationModel

Precipitation model with rain.
"""
struct RainModel <: PrecipitationModel end

vars_state(::RainModel, ::Prognostic, FT) = @vars(ρq_rai::FT)
vars_state(::RainModel, ::Gradient, FT) = @vars(q_rai::FT)
vars_state(::RainModel, ::GradientFlux, FT) = @vars(∇q_rai::SVector{3, FT})

precompute(::RainModel, atmos::AtmosModel, args, ts, ::Source) =
    (cache = warm_rain_sources(atmos, args, ts),)

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

"""
    RainSnowModel <: PrecipitationModel

Precipitation model with rain and snow.
"""
struct RainSnowModel <: PrecipitationModel end

vars_state(::RainSnowModel, ::Prognostic, FT) = @vars(ρq_rai::FT, ρq_sno::FT)
vars_state(::RainSnowModel, ::Gradient, FT) = @vars(q_rai::FT, q_sno::FT)
vars_state(::RainSnowModel, ::GradientFlux, FT) =
    @vars(∇q_rai::SVector{3, FT}, ∇q_sno::SVector{3, FT})

precompute(::RainSnowModel, atmos::AtmosModel, args, ts, ::Source) =
    (cache = rain_snow_sources(atmos, args, ts),)

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

#####
##### Tendency specifications
#####

eq_tends(::Rain, ::RainModel, ::Flux{FirstOrder}) = (PrecipitationFlux(),)
eq_tends(::Rain, ::RainModel, ::Flux{SecondOrder}) = (Diffusion(),)

eq_tends(::AbstractPrecipitationVariable, ::RainSnowModel, ::Flux{FirstOrder}) =
    (PrecipitationFlux(),)
eq_tends(
    ::AbstractPrecipitationVariable,
    ::RainSnowModel,
    ::Flux{SecondOrder},
) = (Diffusion(),)
