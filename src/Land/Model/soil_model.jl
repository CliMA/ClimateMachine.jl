#### Soil model

export SoilModel, SoilParamFunctions, WaterParamFunctions

"""
    AbstractSoilParameterFunctions{FT <: AbstractFloat}
"""
abstract type AbstractSoilParameterFunctions{FT <: AbstractFloat} end


"""
    WaterParamFunctions{FT, TK, TS, TR} <: AbstractSoilParameterFunctions{FT}

Necessary parameters for the soil water model. These can be floating point
 parameters or functions of space (via the auxiliary variable `aux`). Internally,
they will be converted to type FT or altered so that the functions return type FT.

This is not a complete list - the hydraulic parameters necessary for specifying
the van Genuchten or Brooks and Corey functions are stored in the hydraulics model.
# Fields
$(DocStringExtensions.FIELDS)
"""
struct WaterParamFunctions{FT, TK, TS, TR} <: AbstractSoilParameterFunctions{FT}
    "Saturated conductivity. Units of m s-1."
    Ksat::TK
    "Specific storage. Units of m s-1."
    S_s::TS
    "Residual Water Fraction - default is zero; unitless."
    θ_r::TR
end


"""
    HeatParamFunctions{FT, TK, TS, TR} <: AbstractSoilParameterFunctions{FT}

Necessary parameters for the soil water model. These can be floating point
 parameters or functions of space (via the auxiliary variable `aux`). Internally,
they will be converted to type FT or altered so that the functions return type FT.

This is not a complete list - the hydraulic parameters necessary for specifying
the van Genuchten or Brooks and Corey functions are stored in the hydraulics model.
# Fields
$(DocStringExtensions.FIELDS)
"""

function WaterParamFunctions(
    ::Type{FT};
    Ksat::Union{Function, AbstractFloat} = (aux) -> eltype(aux)(0.0),
    S_s::Union{Function, AbstractFloat} = (aux) -> eltype(aux)(1e-3),
    θ_r::Union{Function, AbstractFloat} = (aux) -> eltype(aux)(0.0),
) where {FT}
    fKsat = Ksat isa AbstractFloat ? (aux) -> FT(Ksat) : (aux) -> FT(Ksat(aux))
    fS_s = S_s isa AbstractFloat ? (aux) -> FT(S_s) : (aux) -> FT(S_s(aux))
    fθ = θ_r isa AbstractFloat ? (aux) -> FT(θ_r) : (aux) -> FT(θ_r(aux))
    args = (fKsat, fS_s, fθ)
    return WaterParamFunctions{FT, typeof.(args)...}(args...)
end


"""
    SoilParamFunctions{FT, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, WP}
               <: AbstractSoilParameterFunctions{FT}

Necessary parameters for the soil model. Heat parameters to be moved to their 
own structure in next iteration.
# Fields
$(DocStringExtensions.FIELDS)
"""
struct SoilParamFunctions{
    FT,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    WP,
} <: AbstractSoilParameterFunctions{FT}
    "Aggregate porosity of the soil"
    porosity::F1
    "Volume fraction of gravels, relative to soil solids only; unitless."
    ν_ss_gravel::F2
    "Volume fraction of SOM, relative to soil solids only; unitless."
    ν_ss_om::F3
    "Volume fraction of quartz, relative to soil solids only; unitless."
    ν_ss_quartz::F4
    "Bulk volumetric heat capacity of dry soil. Units of J m-3 K-1."
    ρc_ds::F5
    "Particle density for soil solids. Units of kg m-3"
    ρp::F6
    "Thermal conductivity of the soil solids. Units of W m-1 K-1."
    κ_solid::F7
    "Saturated thermal conductivity for unfrozen soil. Units of W m-1 K-1."
    κ_sat_unfrozen::F8
    "Saturated thermal conductivity for frozen soil. Units of W m-1 K-1."
    κ_sat_frozen::F9
    "Adjustable scale parameter for determining Kersten number in the Balland and Arp formulation; unitless."
    a::F10
    "Adjustable scale parameter for determining Kersten number in the Balland and Arp formulation; unitless."
    b::F11
    "Parameter used in the Balland and Arp formulation for κ_dry; unitless"
    κ_dry_parameter::F12
    "Hydrology parameter functions"
    water::WP
end

function SoilParamFunctions(
    ::Type{FT};
    porosity::FT = FT(NaN),
    ν_ss_gravel::FT = FT(NaN),
    ν_ss_om::FT = FT(NaN),
    ν_ss_quartz::FT = FT(NaN),
    ρc_ds::FT = FT(NaN),
    ρp::FT = FT(NaN),
    κ_solid::FT = FT(NaN),
    κ_sat_unfrozen::FT = FT(NaN),
    κ_sat_frozen::FT = FT(NaN),
    a::FT = FT(0.24),
    b::FT = FT(18.1),
    κ_dry_parameter::FT = FT(0.053),
    water::AbstractSoilParameterFunctions{FT} = WaterParamFunctions(FT;),
) where {FT}
    args = (
        porosity,
        ν_ss_gravel,
        ν_ss_om,
        ν_ss_quartz,
        ρc_ds,
        ρp,
        κ_solid,
        κ_sat_unfrozen,
        κ_sat_frozen,
        a,
        b,
        κ_dry_parameter,
        water,
    )

    return SoilParamFunctions{FT, typeof.(args)...}(args...)
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
