export DryModel, EquilMoist, NonEquilMoist

#### Moisture component in atmosphere model
abstract type MoistureModel end

vars_state(::MoistureModel, ::AbstractStateType, FT) = @vars()

function atmos_nodal_update_auxiliary_state!(
    ::MoistureModel,
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function flux_moisture!(
    ::MoistureModel,
    ::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function compute_gradient_flux!(
    ::MoistureModel,
    diffusive,
    ∇transform,
    state,
    aux,
    t,
) end
function flux_second_order!(
    ::MoistureModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
) end
function flux_first_order!(::MoistureModel, _...) end
function compute_gradient_argument!(
    ::MoistureModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) end

internal_energy(atmos::AtmosModel, state::Vars, aux::Vars) =
    internal_energy(atmos.moisture, atmos.orientation, state, aux)

@inline function internal_energy(
    moist::MoistureModel,
    orientation::Orientation,
    state::Vars,
    aux::Vars,
)
    Thermodynamics.internal_energy(
        state.ρ,
        state.ρe,
        state.ρu,
        gravitational_potential(orientation, aux),
    )
end

temperature(atmos::AtmosModel, ::MoistureModel, state::Vars, aux::Vars) =
    air_temperature(recover_thermo_state(atmos, state, aux))
pressure(atmos::AtmosModel, ::MoistureModel, state::Vars, aux::Vars) =
    air_pressure(recover_thermo_state(atmos, state, aux))
soundspeed(atmos::AtmosModel, ::MoistureModel, state::Vars, aux::Vars) =
    soundspeed_air(recover_thermo_state(atmos, state, aux))

@inline function total_specific_enthalpy(
    atmos::AtmosModel,
    moist::MoistureModel,
    state::Vars,
    aux::Vars,
)
    phase = recover_thermo_state(atmos, state, aux)
    e_tot = state.ρe * (1 / state.ρ)
    return total_specific_enthalpy(phase, e_tot)
end

"""
    DryModel

Assumes the moisture components is in the dry limit.
"""
struct DryModel <: MoistureModel end

vars_state(::DryModel, ::Auxiliary, FT) = @vars(θ_v::FT, air_T::FT)
@inline function atmos_nodal_update_auxiliary_state!(
    moist::DryModel,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ts = new_thermo_state(atmos, state, aux)
    aux.moisture.θ_v = virtual_pottemp(ts)
    aux.moisture.air_T = air_temperature(ts)
    nothing
end

"""
    EquilMoist

Assumes the moisture components are computed via thermodynamic equilibrium.
"""
struct EquilMoist{FT} <: MoistureModel
    maxiter::Int
    tolerance::FT
end
EquilMoist{FT}(;
    maxiter::IT = 3,
    tolerance::FT = FT(1e-1),
) where {FT <: AbstractFloat, IT <: Int} = EquilMoist{FT}(maxiter, tolerance)


vars_state(::EquilMoist, ::Prognostic, FT) = @vars(ρq_tot::FT)
vars_state(::EquilMoist, ::Gradient, FT) = @vars(q_tot::FT)
vars_state(::EquilMoist, ::GradientFlux, FT) = @vars(∇q_tot::SVector{3, FT})
vars_state(::EquilMoist, ::Auxiliary, FT) =
    @vars(temperature::FT, θ_v::FT, q_liq::FT, q_ice::FT)

@inline function atmos_nodal_update_auxiliary_state!(
    moist::EquilMoist,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ts = new_thermo_state(atmos, state, aux)
    aux.moisture.temperature = air_temperature(ts)
    aux.moisture.θ_v = virtual_pottemp(ts)
    aux.moisture.q_liq = PhasePartition(ts).liq
    aux.moisture.q_ice = PhasePartition(ts).ice
    nothing
end

function compute_gradient_argument!(
    moist::EquilMoist,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.moisture.q_tot = state.moisture.ρq_tot * ρinv
end

function compute_gradient_flux!(
    moist::EquilMoist,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    # diffusive flux of q_tot
    diffusive.moisture.∇q_tot = ∇transform.moisture.q_tot
end

function flux_moisture!(
    moist::EquilMoist,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρ = state.ρ
    u = state.ρu / ρ
    flux.moisture.ρq_tot += u * state.moisture.ρq_tot
end

function flux_second_order!(
    moist::EquilMoist,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    d_q_tot = (-D_t) .* diffusive.moisture.∇q_tot
    flux_second_order!(moist, flux, state, d_q_tot)
end
#TODO: Consider whether to not pass ρ and ρu (not state), foc BCs reasons
function flux_second_order!(moist::EquilMoist, flux::Grad, state::Vars, d_q_tot)
    flux.ρ += d_q_tot * state.ρ
    flux.ρu += d_q_tot .* state.ρu'
    flux.moisture.ρq_tot += d_q_tot * state.ρ
end

"""
    NonEquilMoist

Does not assume that the moisture components are in equilibrium.
"""
struct NonEquilMoist <: MoistureModel end

vars_state(::NonEquilMoist, ::Prognostic, FT) =
    @vars(ρq_tot::FT, ρq_liq::FT, ρq_ice::FT)
vars_state(::NonEquilMoist, ::Gradient, FT) =
    @vars(q_tot::FT, q_liq::FT, q_ice::FT)
vars_state(::NonEquilMoist, ::GradientFlux, FT) = @vars(
    ∇q_tot::SVector{3, FT},
    ∇q_liq::SVector{3, FT},
    ∇q_ice::SVector{3, FT}
)
vars_state(::NonEquilMoist, ::Auxiliary, FT) = @vars(temperature::FT, θ_v::FT)

@inline function atmos_nodal_update_auxiliary_state!(
    moist::NonEquilMoist,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ts = new_thermo_state(atmos, state, aux)
    aux.moisture.temperature = air_temperature(ts)
    aux.moisture.θ_v = virtual_pottemp(ts)
    nothing
end

function compute_gradient_argument!(
    moist::NonEquilMoist,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.moisture.q_tot = state.moisture.ρq_tot * ρinv
    transform.moisture.q_liq = state.moisture.ρq_liq * ρinv
    transform.moisture.q_ice = state.moisture.ρq_ice * ρinv
end

function compute_gradient_flux!(
    moist::NonEquilMoist,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    # diffusive fluxes of moisture variables
    diffusive.moisture.∇q_tot = ∇transform.moisture.q_tot
    diffusive.moisture.∇q_liq = ∇transform.moisture.q_liq
    diffusive.moisture.∇q_ice = ∇transform.moisture.q_ice
end

function flux_moisture!(
    moist::NonEquilMoist,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρ = state.ρ
    u = state.ρu / ρ
    flux.moisture.ρq_tot += u * state.moisture.ρq_tot
    flux.moisture.ρq_liq += u * state.moisture.ρq_liq
    flux.moisture.ρq_ice += u * state.moisture.ρq_ice
end

function flux_second_order!(
    moist::NonEquilMoist,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    d_q_tot = (-D_t) .* diffusive.moisture.∇q_tot
    d_q_liq = (-D_t) .* diffusive.moisture.∇q_liq
    d_q_ice = (-D_t) .* diffusive.moisture.∇q_ice

    flux_second_order!(moist, flux, state, d_q_tot, d_q_liq, d_q_ice)
end

function flux_second_order!(
    moist::NonEquilMoist,
    flux::Grad,
    state::Vars,
    d_q_tot,
    d_q_liq,
    d_q_ice,
)
    flux.ρ += d_q_tot * state.ρ
    flux.ρu += d_q_tot .* state.ρu'
    flux.moisture.ρq_tot += d_q_tot * state.ρ
    flux.moisture.ρq_liq += d_q_liq * state.ρ
    flux.moisture.ρq_ice += d_q_ice * state.ρ
end
