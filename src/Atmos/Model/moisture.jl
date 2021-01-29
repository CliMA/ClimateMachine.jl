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
function flux_first_order!(::MoistureModel, ::AtmosModel, flux::Grad, args) end
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
    atmos::AtmosModel,
    args,
) end

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

"""
    DryModel

Assumes the moisture components is in the dry limit.
"""
struct DryModel <: MoistureModel end

vars_state(::DryModel, ::Auxiliary, FT) = @vars(θ_v::FT, temperature::FT)
@inline function atmos_nodal_update_auxiliary_state!(
    moist::DryModel,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ts = new_thermo_state(atmos, state, aux)
    aux.moisture.θ_v = virtual_pottemp(ts)
    aux.moisture.temperature = air_temperature(ts)
    nothing
end

source!(::DryModel, args...) = nothing

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
vars_state(::EquilMoist, ::Primitive, FT) = @vars(q_tot::FT)
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

function flux_first_order!(
    moist::EquilMoist,
    atmos::AtmosModel,
    flux::Grad,
    args,
)
    tend = Flux{FirstOrder}()
    flux.moisture.ρq_tot =
        Σfluxes(eq_tends(TotalMoisture(), atmos, tend), atmos, args)
end

function flux_second_order!(
    moist::EquilMoist,
    flux::Grad,
    atmos::AtmosModel,
    args,
)
    tend = Flux{SecondOrder}()
    flux.moisture.ρq_tot =
        Σfluxes(eq_tends(TotalMoisture(), atmos, tend), atmos, args)
end

function source!(m::EquilMoist, source::Vars, atmos::AtmosModel, args)
    tend = Source()
    source.moisture.ρq_tot =
        Σsources(eq_tends(TotalMoisture(), atmos, tend), atmos, args)
end

"""
    NonEquilMoist

Does not assume that the moisture components are in equilibrium.
"""
struct NonEquilMoist <: MoistureModel end

vars_state(::NonEquilMoist, ::Prognostic, FT) =
    @vars(ρq_tot::FT, ρq_liq::FT, ρq_ice::FT)
vars_state(::NonEquilMoist, ::Primitive, FT) =
    @vars(q_tot::FT, q_liq::FT, q_ice::FT)
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

function flux_first_order!(
    moist::NonEquilMoist,
    atmos::AtmosModel,
    flux::Grad,
    args,
)
    tend = Flux{FirstOrder}()
    flux.moisture.ρq_tot =
        Σfluxes(eq_tends(TotalMoisture(), atmos, tend), atmos, args)
    flux.moisture.ρq_liq =
        Σfluxes(eq_tends(LiquidMoisture(), atmos, tend), atmos, args)
    flux.moisture.ρq_ice =
        Σfluxes(eq_tends(IceMoisture(), atmos, tend), atmos, args)
end

function flux_second_order!(
    ::NonEquilMoist,
    flux::Grad,
    atmos::AtmosModel,
    args,
)
    tend = Flux{SecondOrder}()
    flux.moisture.ρq_tot =
        Σfluxes(eq_tends(TotalMoisture(), atmos, tend), atmos, args)
    flux.moisture.ρq_liq =
        Σfluxes(eq_tends(LiquidMoisture(), atmos, tend), atmos, args)
    flux.moisture.ρq_ice =
        Σfluxes(eq_tends(IceMoisture(), atmos, tend), atmos, args)
end

function source!(m::NonEquilMoist, source::Vars, atmos::AtmosModel, args)
    tend = Source()
    source.moisture.ρq_tot =
        Σsources(eq_tends(TotalMoisture(), atmos, tend), atmos, args)
    source.moisture.ρq_liq =
        Σsources(eq_tends(LiquidMoisture(), atmos, tend), atmos, args)
    source.moisture.ρq_ice =
        Σsources(eq_tends(IceMoisture(), atmos, tend), atmos, args)
end
