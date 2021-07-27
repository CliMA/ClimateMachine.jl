export SoilWaterModel, PrescribedWaterModel, get_water_content

abstract type AbstractWaterModel <: AbstractSoilComponentModel end

"""
    PrescribedWaterModel{F1, F2} <: AbstractWaterModel

Model structure for a prescribed water content model.

The user supplies functions of space and time for both `ϑ_l` and
`θ_i`. No auxiliary or state variables are added, no PDE is solved.
The defaults are no moisture anywhere, for all time.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct PrescribedWaterModel{FN1, FN2} <: AbstractWaterModel
    "Augmented liquid fraction"
    ϑ_l::FN1
    "Volumetric fraction of ice"
    θ_i::FN2
end

"""
    PrescribedWaterModel(
        ϑ_l::Function = (aux, t) -> eltype(aux)(0.0),
        θ_i::Function = (aux, t) -> eltype(aux)(0.0),
    )

Outer constructor for the PrescribedWaterModel defining default values.

The functions supplied by the user are point-wise evaluated and are
evaluated in the Balance Law functions compute_gradient_argument,
nodal_update, etc. whenever the prescribed water content variables are
needed by the heat model.
"""
function PrescribedWaterModel(
    ϑ_l::Function = (aux, t) -> eltype(aux)(0.0),
    θ_i::Function = (aux, t) -> eltype(aux)(0.0),
)
    args = (ϑ_l, θ_i)
    return PrescribedWaterModel{typeof.(args)...}(args...)
end



"""
    SoilWaterModel{FT, IF, VF, MF, HM, Fiϑl, Fiθi} <: AbstractWaterModel

The necessary components for solving the equations for water (liquid or ice) in soil.

Without freeze/thaw source terms added (separately), this model reduces to
Richard's equation for liquid water. Note that the default for `θ_i` is zero.
Without freeze/thaw source terms added to both the liquid and ice equations,
the default should never be changed, because we do not enforce that the total
volumetric water fraction is less than or equal to porosity otherwise.

When freeze/thaw source terms are included, this model encompasses water in both
liquid and ice form, and water content is conserved upon phase change.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SoilWaterModel{FT, IF, VF, MF, HM, Fiϑl, Fiθi} <: AbstractWaterModel
    "Impedance Factor - will be 1 or ice dependent"
    impedance_factor::IF
    "Viscosity Factor - will be 1 or temperature dependent"
    viscosity_factor::VF
    "Moisture Factor - will be 1 or moisture dependent"
    moisture_factor::MF
    "Hydraulics Model - used in matric potential and moisture factor of hydraulic conductivity"
    hydraulics::HM
    "Initial condition: augmented liquid fraction"
    initialϑ_l::Fiϑl
    "Initial condition: volumetric ice fraction"
    initialθ_i::Fiθi
end

"""
    SoilWaterModel(
        ::Type{FT};
        impedance_factor::AbstractImpedanceFactor{FT} = NoImpedance{FT}(),
        viscosity_factor::AbstractViscosityFactor{FT} = ConstantViscosity{FT}(),
        moisture_factor::AbstractMoistureFactor{FT} = MoistureIndependent{FT}(),
        hydraulics::AbstractHydraulicsModel{FT} = vanGenuchten(FT;),
        initialϑ_l = (aux) -> FT(NaN),
        initialθ_i = (aux) -> FT(0.0),
    ) where {FT}

Constructor for the SoilWaterModel. Defaults imply a constant K = K_sat model.
"""
function SoilWaterModel(
    ::Type{FT};
    impedance_factor::AbstractImpedanceFactor{FT} = NoImpedance{FT}(),
    viscosity_factor::AbstractViscosityFactor{FT} = ConstantViscosity{FT}(),
    moisture_factor::AbstractMoistureFactor{FT} = MoistureIndependent{FT}(),
    hydraulics::AbstractHydraulicsModel{FT} = vanGenuchten(FT;),
    initialϑ_l::Function = (aux) -> eltype(aux)(NaN),
    initialθ_i::Function = (aux) -> eltype(aux)(0.0),
) where {FT}

    args = (
        impedance_factor,
        viscosity_factor,
        moisture_factor,
        hydraulics,
        initialϑ_l,
        initialθ_i,
    )
    return SoilWaterModel{FT, typeof.(args)...}(args...)
end


"""
    get_water_content(
        water::SoilWaterModel,
        aux::Vars,
        state::Vars,
        t::Real
    )

Return the moisture variables for the balance law soil water model.
"""
function get_water_content(
    water::SoilWaterModel,
    aux::Vars,
    state::Vars,
    t::Real,
)
    FT = eltype(state)
    return FT(state.soil.water.ϑ_l), FT(state.soil.water.θ_i)
end



"""
    get_water_content(
        water::PrescribedWaterModel,
        aux::Vars,
        state::Vars,
        t::Real
    )

Return the moisture variables for the prescribed soil water model.
"""
function get_water_content(
    water::PrescribedWaterModel,
    aux::Vars,
    state::Vars,
    t::Real,
)
    FT = eltype(aux)
    ϑ_l = water.ϑ_l(aux, t)
    θ_i = water.θ_i(aux, t)
    return FT(ϑ_l), FT(θ_i)
end

vars_state(water::SoilWaterModel, st::Prognostic, FT) = @vars(ϑ_l::FT, θ_i::FT)

vars_state(water::SoilWaterModel, st::Auxiliary, FT) = @vars(h::FT, K::FT)


vars_state(water::SoilWaterModel, st::Gradient, FT) = @vars(h::FT)


vars_state(::SoilWaterModel, st::GradientFlux, FT) = @vars(K∇h::SVector{3, FT})#really, the flux is - K∇h

function soil_init_aux!(
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    aux::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    T = get_initial_temperature(land.soil.heat, aux, FT(0.0))
    θ_r = soil.param_functions.water.θ_r(aux)
    ν = soil.param_functions.porosity #(aux)
    S_s = soil.param_functions.water.S_s(aux)
    Ksat = soil.param_functions.water.Ksat(aux)

    ϑ_l = water.initialϑ_l(aux)
    θ_i = water.initialθ_i(aux)
    θ_l = volumetric_liquid_fraction(ϑ_l, ν - θ_i)

    S_l = effective_saturation(ν, ϑ_l, θ_r)
    hydraulics = water.hydraulics(aux)
    ψ = pressure_head(hydraulics, ν, S_s, θ_r, ϑ_l, θ_i)
    aux.soil.water.h = hydraulic_head(aux.z, ψ)

    # compute conductivity
    f_i = θ_i / (θ_l + θ_i)
    impedance_f = impedance_factor(water.impedance_factor, f_i)
    viscosity_f = viscosity_factor(water.viscosity_factor, T)
    moisture_f = moisture_factor(water.moisture_factor, hydraulics, S_l)
    aux.soil.water.K =
        hydraulic_conductivity(Ksat, impedance_f, viscosity_f, moisture_f)
end


function land_nodal_update_auxiliary_state!(
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ν = soil.param_functions.porosity #(aux)
    S_s = soil.param_functions.water.S_s(aux)
    Ksat = soil.param_functions.water.Ksat(aux)
    θ_r = soil.param_functions.water.θ_r(aux)

    ϑ_l = state.soil.water.ϑ_l
    θ_i = state.soil.water.θ_i
    θ_l = volumetric_liquid_fraction(ϑ_l, ν - θ_i)

    T = get_temperature(land.soil.heat, aux, t)
    S_l = effective_saturation(ν, ϑ_l, θ_r)
    hydraulics = water.hydraulics(aux)
    ψ = pressure_head(hydraulics, ν, S_s, θ_r, ϑ_l, θ_i)
    aux.soil.water.h = hydraulic_head(aux.z, ψ)

    f_i = θ_i / (θ_l + θ_i)
    impedance_f = impedance_factor(water.impedance_factor, f_i)
    viscosity_f = viscosity_factor(water.viscosity_factor, T)
    moisture_f = moisture_factor(water.moisture_factor, hydraulics, S_l)
    aux.soil.water.K =
        hydraulic_conductivity(Ksat, impedance_f, viscosity_f, moisture_f)
end


function compute_gradient_argument!(
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ν = soil.param_functions.porosity #(aux)
    S_s = soil.param_functions.water.S_s(aux)
    θ_r = soil.param_functions.water.θ_r(aux)
    ϑ_l = state.soil.water.ϑ_l
    θ_i = state.soil.water.θ_i

    S_l = effective_saturation(ν, ϑ_l, θ_r)
    hydraulics = water.hydraulics(aux)
    ψ = pressure_head(hydraulics, ν, S_s, θ_r, ϑ_l, θ_i)
    transform.soil.water.h = hydraulic_head(aux.z, ψ)

end


function compute_gradient_flux!(
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.soil.water.K∇h = aux.soil.water.K * ∇transform.soil.water.h
end

struct DarcyFlux <: TendencyDef{Flux{SecondOrder}} end

function flux(::VolumetricLiquidFraction, ::DarcyFlux, ::LandModel, args)
    @unpack diffusive = args
    return -diffusive.soil.water.K∇h
end
