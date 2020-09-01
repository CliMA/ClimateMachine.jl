### Soil heat model

export SoilHeatModel, PrescribedTemperatureModel

abstract type AbstractHeatModel <: AbstractSoilComponentModel end

"""
    PrescribedTemperatureModel{F1} <: AbstractHeatModel

Model structure for a prescribed temperature model.
"""
struct PrescribedTemperatureModel{F1} <: AbstractHeatModel
    "Temperature"
    T::F1
end

"""
    PrescribedTemperatureModel(
        T::Function = (aux,t) -> eltype(aux)(0.0)
    )

Outer constructor for the PrescribedTemperatureModel defining default values.

The functions supplied by the user are point-wise evaluated and are
evaluated in the Balance Law functions compute_gradient_argument,
 nodal_update, etc. whenever the prescribed temperature content variables are
needed by the water model.
"""
function PrescribedTemperatureModel(T::Function = (aux, t) -> eltype(aux)(0.0))
    return PrescribedTemperatureModel{typeof(T)}(T)
end

"""
    SoilHeatModel{FT, FiT, BCD, BCN} <: AbstractHeatModel

The necessary components for the Heat Equation in a soil water matrix.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SoilHeatModel{FT, FiT, BCD, BCN} <: AbstractHeatModel
    "Initial conditions for temperature"
    initialT::FiT
    "Dirichlet BC structure"
    dirichlet_bc::BCD
    "Neumann BC structure"
    neumann_bc::BCN
end

"""
    SoilHeatModel(
        ::Type{FT};
        initialT::FT = FT(NaN),
        dirichlet_bc::AbstractBoundaryFunctions = nothing,
        neumann_bc::AbstractBoundaryFunctions = nothing
    ) where {FT}

Constructor for the SoilHeatModel.
"""
function SoilHeatModel(
    ::Type{FT};
    initialT = (aux) -> FT(NaN),
    dirichlet_bc::AbstractBoundaryFunctions = nothing,
    neumann_bc::AbstractBoundaryFunctions = nothing,
) where {FT}
    args = (initialT, dirichlet_bc, neumann_bc)
    return SoilHeatModel{FT, typeof.(args)...}(args...)
end

"""
    function get_temperature(
        heat::SoilHeatModel
        aux::Vars,
        t::Real
    )
Returns the temperature when the heat model chosen is the dynamical SoilHeatModel.
This is necessary for fully coupling heat and water.
"""
function get_temperature(heat::SoilHeatModel, aux::Vars, t::Real)
    T = aux.soil.heat.T
    return T
end

"""
    function get_temperature(
        heat::PrescribedTemperatureModel,
        aux::Vars,
        t::Real
    )
Returns the temperature when the heat model chosen is a user prescribed one.
This is useful for driving Richard's equation without a back reaction on temperature.
"""
function get_temperature(heat::PrescribedTemperatureModel, aux::Vars, t::Real)
    T = heat.T(aux, t)
    return T
end

"""
    function get_initial_temperature(
        m::SoilHeatModel
        aux::Vars,
        t::Real
    )    
Returns the temperature from the SoilHeatModel.
Needed for soil_init_aux! of SoilWaterModel.
"""
function get_initial_temperature(m::SoilHeatModel, aux::Vars, t::Real)
    return m.initialT(aux)
end


"""
    function get_initial_temperature(
        m::PrescribedTemperatureModel,
        aux::Vars,
        t::Real
    )    
Returns the temperature from the prescribed model.
Needed for soil_init_aux! of SoilWaterModel.
"""
function get_initial_temperature(
    m::PrescribedTemperatureModel,
    aux::Vars,
    t::Real,
)
    return m.T(aux, t)
end

vars_state(heat::SoilHeatModel, st::Prognostic, FT) = @vars(ρe_int::FT)
vars_state(heat::SoilHeatModel, st::Auxiliary, FT) = @vars(T::FT)
vars_state(heat::SoilHeatModel, st::Gradient, FT) = @vars(T::FT)
vars_state(heat::SoilHeatModel, st::GradientFlux, FT) =
    @vars(κ∇T::SVector{3, FT})

function soil_init_aux!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.soil.heat.T = heat.initialT(aux)
end

function land_nodal_update_auxiliary_state!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    state::Vars,
    aux::Vars,
    t::Real,
)

    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, t)
    θ_l = volumetric_liquid_fraction(ϑ_l, soil.param_functions.porosity)
    ρc_ds = soil.param_functions.ρc_ds
    ρcs = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, land.param_set)
    aux.soil.heat.T = temperature_from_ρe_int(
        state.soil.heat.ρe_int,
        θ_i,
        ρcs,
        land.param_set,
    )
end

function compute_gradient_argument!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)

    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, t)
    θ_l = volumetric_liquid_fraction(ϑ_l, soil.param_functions.porosity)
    ρc_ds = soil.param_functions.ρc_ds
    ρcs = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, land.param_set)
    transform.soil.heat.T = temperature_from_ρe_int(
        state.soil.heat.ρe_int,
        θ_i,
        ρcs,
        land.param_set,
    )
end

function compute_gradient_flux!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, t)
    θ_l = volumetric_liquid_fraction(ϑ_l, soil.param_functions.porosity)
    κ_dry = soil.param_functions.κ_dry
    S_r = relative_saturation(θ_l, θ_i, soil.param_functions.porosity)
    kersten = kersten_number(θ_i, S_r, soil.param_functions)
    κ_sat = saturated_thermal_conductivity(
        θ_l,
        θ_i,
        soil.param_functions.κ_sat_unfrozen,
        soil.param_functions.κ_sat_frozen,
    )
    diffusive.soil.heat.κ∇T =
        thermal_conductivity(κ_dry, kersten, κ_sat) * ∇transform.soil.heat.T
end

function flux_second_order!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    ρe_int_l = volumetric_internal_energy_liq(aux.soil.heat.T, land.param_set)
    diffusive_water_flux =
        -ρe_int_l .* get_diffusive_water_flux(soil.water, diffusive)
    diffusive_heat_flux = -diffusive.soil.heat.κ∇T
    flux.soil.heat.ρe_int += diffusive_heat_flux + diffusive_water_flux
end
