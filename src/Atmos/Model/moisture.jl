export DryModel, EquilMoist

#### Moisture component in atmosphere model
abstract type MoistureModel end

vars_state(::MoistureModel, FT) = @vars()
vars_gradient(::MoistureModel, FT) = @vars()
vars_diffusive(::MoistureModel, FT) = @vars()
vars_aux(::MoistureModel, FT) = @vars()

function atmos_nodal_update_aux!(
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
function diffusive!(::MoistureModel, diffusive, ∇transform, state, aux, t) end
function flux_diffusive!(
    ::MoistureModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
) end
function flux_nondiffusive!(
    ::MoistureModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function gradvariables!(
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
    MoistThermodynamics.internal_energy(
        state.ρ,
        state.ρe,
        state.ρu,
        gravitational_potential(orientation, aux),
    )
end

temperature(atmos::AtmosModel, ::MoistureModel, state::Vars, aux::Vars) =
    air_temperature(thermo_state(atmos, state, aux))
pressure(atmos::AtmosModel, ::MoistureModel, state::Vars, aux::Vars) =
    air_pressure(thermo_state(atmos, state, aux))
soundspeed(atmos::AtmosModel, ::MoistureModel, state::Vars, aux::Vars) =
    soundspeed_air(thermo_state(atmos, state, aux))

@inline function total_specific_enthalpy(
    atmos::AtmosModel,
    moist::MoistureModel,
    state::Vars,
    aux::Vars,
)
    phase = thermo_state(atmos, state, aux)
    R_m = gas_constant_air(phase)
    T = air_temperature(phase)
    e_tot = state.ρe * (1 / state.ρ)
    return e_tot + R_m * T
end

"""
    DryModel

Assumes the moisture components is in the dry limit.
"""
struct DryModel <: MoistureModel end

vars_aux(::DryModel, FT) = @vars(θ_v::FT, air_T::FT)
@inline function atmos_nodal_update_aux!(
    moist::DryModel,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    e_int = internal_energy(atmos, state, aux)
    ts = PhaseDry(e_int, state.ρ, atmos.param_set)
    aux.moisture.θ_v = virtual_pottemp(ts)
    aux.moisture.air_T = air_temperature(ts)
    nothing
end

thermo_state(atmos::AtmosModel, state::Vars, aux::Vars) =
    thermo_state(atmos, atmos.moisture, state, aux)

function thermo_state(
    atmos::AtmosModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
)
    return PhaseDry(
        internal_energy(atmos, state, aux),
        state.ρ,
        atmos.param_set,
    )
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


vars_state(::EquilMoist, FT) = @vars(ρq_tot::FT)
vars_gradient(::EquilMoist, FT) = @vars(q_tot::FT)
vars_diffusive(::EquilMoist, FT) = @vars(∇q_tot::SVector{3, FT})
vars_aux(::EquilMoist, FT) = @vars(temperature::FT, θ_v::FT, q_liq::FT)

@inline function atmos_nodal_update_aux!(
    moist::EquilMoist,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ps = atmos.param_set
    e_int = internal_energy(atmos, state, aux)
    ts = PhaseEquil(
        e_int,
        state.ρ,
        state.moisture.ρq_tot / state.ρ,
        moist.maxiter,
        moist.tolerance,
        ps,
    )
    aux.moisture.temperature = air_temperature(ts)
    aux.moisture.θ_v = virtual_pottemp(ts)
    aux.moisture.q_liq = PhasePartition(ts).liq
    nothing
end

function thermo_state(
    atmos::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    e_int = internal_energy(atmos, state, aux)
    ps = atmos.param_set
    PS = typeof(ps)
    FT = eltype(state)
    return PhaseEquil{FT, PS}(
        ps,
        e_int,
        state.ρ,
        state.moisture.ρq_tot / state.ρ,
        aux.moisture.temperature,
    )
end

function gradvariables!(
    moist::EquilMoist,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.moisture.q_tot = state.moisture.ρq_tot * ρinv
end

function diffusive!(
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

function flux_diffusive!(
    moist::EquilMoist,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    d_q_tot = (-D_t) .* diffusive.moisture.∇q_tot
    flux_diffusive!(moist, flux, state, d_q_tot)
end
#TODO: Consider whether to not pass ρ and ρu (not state), foc BCs reasons
function flux_diffusive!(moist::EquilMoist, flux::Grad, state::Vars, d_q_tot)
    flux.ρ += d_q_tot * state.ρ
    flux.ρu += d_q_tot .* state.ρu'
    flux.moisture.ρq_tot += d_q_tot * state.ρ
end
