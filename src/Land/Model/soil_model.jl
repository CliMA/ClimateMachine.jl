#### Soil model

export SoilModel, SoilParamFunctions

"""
    AbstractSoilParameterFunctions{FT <: AbstractFloat}
"""
abstract type AbstractSoilParameterFunctions{FT <: AbstractFloat} end

"""
    SoilParamFunctions{FT} <: AbstractSoilParameterFunctions{FT}

Necessary parameters for the soil model. These will eventually be prescribed
functions of space (and time).
# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct SoilParamFunctions{FT} <: AbstractSoilParameterFunctions{FT}
    "Aggregate porosity of the soil"
    porosity::FT = FT(NaN)
    "Hydraulic conductivity at saturation. Units of m s-1."
    Ksat::FT = FT(NaN)
    "Specific storage of the soil. Units of m s-1."
    S_s::FT = FT(NaN)
    "Volume fraction of gravels, relative to soil solids only. Units of m-3 m-3."
    ν_ss_gravel::FT = FT(NaN)
    "Volume fraction of SOM, relative to soil solids only. Units of m-3 m-3."
    ν_ss_om::FT = FT(NaN)
    "Volume fraction of quartz, relative to soil solids only. Units of m-3 m-3."
    ν_ss_quartz::FT = FT(NaN)
    "Bulk volumetric heat capacity of dry soil. Units of J m-3 K-1."
    ρc_ds::FT = FT(NaN)
    "Particle density for soil solids. Units of kg/m^3"
    ρp::FT = FT(NaN)
    "Thermal conductivity of the soil solids. Units of W m-1 K-1."
    κ_solid::FT = FT(NaN)
    "Saturated thermal conductivity for unfrozen soil. Units of W m-1 K-1."
    κ_sat_unfrozen::FT = FT(NaN)
    "Saturated thermal conductivity for frozen soil. Units of W m-1 K-1."
    κ_sat_frozen::FT = FT(NaN)
    "Adjustable scale parameter for determining Kersten number in the Balland and Arp formulation; unitless."
    a::FT = FT(0.24)
    "Adjustable scale parameter for determining Kersten number in the Balland and Arp formulation; unitless."
    b::FT = FT(18.1)
    "Parameter used in the Balland and Arp formulation for κ_dry; unitless"
    κ_dry_parameter::FT = FT(0.053)

end


"""
    SoilModel{PF, W, H} <: BalanceLaw

A BalanceLaw for soil modeling.
Users may over-ride prescribed default values for each field.

# Usage

    SoilModel(
        param_functions,
        water,
        heat,
    )


# Fields
$(DocStringExtensions.FIELDS)
"""
struct SoilModel{PF, W, H} <: BalanceLaw
    "Soil Parameter Functions"
    param_functions::PF
    "Water model"
    water::W
    "Heat model"
    heat::H
end


function vars_state(soil::SoilModel, st::Prognostic, FT)
    @vars begin
        water::vars_state(soil.water, st, FT)
        heat::vars_state(soil.heat, st, FT)
    end
end

function vars_state(soil::SoilModel, st::Auxiliary, FT)
    @vars begin
        water::vars_state(soil.water, st, FT)
        heat::vars_state(soil.heat, st, FT)
    end
end


function vars_state(soil::SoilModel, st::Gradient, FT)
    @vars begin
        water::vars_state(soil.water, st, FT)
        heat::vars_state(soil.heat, st, FT)
    end
end


function vars_state(soil::SoilModel, st::GradientFlux, FT)
    @vars begin
        water::vars_state(soil.water, st, FT)
        heat::vars_state(soil.heat, st, FT)
    end
end


function flux_first_order!(
    land::LandModel,
    soil::SoilModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    directions,
) end


function compute_gradient_argument!(
    land::LandModel,
    soil::SoilModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    compute_gradient_argument!(land, soil, soil.heat, transform, state, aux, t)
    compute_gradient_argument!(land, soil, soil.water, transform, state, aux, t)
end


function compute_gradient_flux!(
    land::LandModel,
    soil::SoilModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    compute_gradient_flux!(
        land,
        soil,
        soil.water,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
    compute_gradient_flux!(
        land,
        soil,
        soil.heat,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )

end


function flux_second_order!(
    land::LandModel,
    soil::SoilModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux_second_order!(
        land,
        soil,
        soil.water,
        flux,
        state,
        diffusive,
        hyperdiffusive,
        aux,
        t,
    )
    flux_second_order!(
        land,
        soil,
        soil.heat,
        flux,
        state,
        diffusive,
        hyperdiffusive,
        aux,
        t,
    )

end


function land_nodal_update_auxiliary_state!(
    land::LandModel,
    soil::SoilModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    land_nodal_update_auxiliary_state!(land, soil, soil.water, state, aux, t)
    land_nodal_update_auxiliary_state!(land, soil, soil.heat, state, aux, t)
end


function land_init_aux!(
    land::LandModel,
    soil::SoilModel,
    aux::Vars,
    geom::LocalGeometry,
)
    soil_init_aux!(land, soil, soil.water, aux, geom)
    soil_init_aux!(land, soil, soil.heat, aux, geom)
end

"""
    abstract type AbstractSoilComponentModel <: BalanceLaw
"""
abstract type AbstractSoilComponentModel <: BalanceLaw end

## When PrescribedModels are chosen, all balance law functions and boundary
## condition functions should do nothing. Since these models are of super
## type AbstractSoilComponentModel, use AbstractSoilComponentModel in
## argument type. The more specific Water and Heat models will have
## different methods (see soil_water.jl and soil_heat.jl).

vars_state(m::AbstractSoilComponentModel, st::AbstractStateType, FT) = @vars()


function soil_init_aux!(
    land::LandModel,
    soil::SoilModel,
    m::AbstractSoilComponentModel,
    aux::Vars,
    geom::LocalGeometry,
) end


function land_nodal_update_auxiliary_state!(
    land::LandModel,
    soil::SoilModel,
    m::AbstractSoilComponentModel,
    state::Vars,
    aux::Vars,
    t::Real,
)

end


function compute_gradient_argument!(
    land::LandModel,
    soil::SoilModel,
    m::AbstractSoilComponentModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)

end


function compute_gradient_flux!(
    land::LandModel,
    soil::SoilModel,
    m::AbstractSoilComponentModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

end


function flux_second_order!(
    land::LandModel,
    soil::SoilModel,
    m::AbstractSoilComponentModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)

end
