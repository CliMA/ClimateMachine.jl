export SoilWaterModel, PrescribedWaterModel

abstract type AbstractWaterModel <: AbstractSoilComponentModel end

"""
    struct PrescribedWaterModel{F1, F2} <: AbstractWaterModel

Model structure for a prescribed water content model.

The user supplies functions of space and time for both `ϑ_l` and
`θ_ice`. No auxiliary or state variables are added, no PDE is solved. 
The defaults are no moisture anywhere, for all time. 

# Fields
$(DocStringExtensions.FIELDS)
"""
struct PrescribedWaterModel{F1, F2} <: AbstractWaterModel
    "Augmented liquid fraction"
    ϑ_l::F1
    "Volumetric fraction of ice"
    θ_ice::F2
end

"""
    PrescribedWaterModel(
        ϑ_l::Function = (aux,t) -> eltype(aux)(0.0),
        θ_ice::Function = (aux,t) -> eltype(aux)(0.0),
    ) where {FT}

Outer constructor for the PrescribedWaterModel defining default values, and
making it so changes to those defaults are supplied via keyword args.

The functions supplied by the user are point-wise evaluated and are 
evaluated in the Balance Law functions (kernels?) compute_gradient_argument,
 nodal_update, etc. whenever the prescribed water content variables are 
needed by the heat model.
"""
function PrescribedWaterModel(
    ϑ_l::Function = (aux, t) -> eltype(aux)(0.0),
    θ_ice::Function = (aux, t) -> eltype(aux)(0.0),
) where {FT}
    args = (ϑ_l, θ_ice)
    return PrescribedWaterModel{typeof.(args)...}(args...)
end


"""
    SoilWaterModel{FT, IF, VF, MF, HM, Fiϑl, Fiθi, BCD, BCN} <: AbstractWaterModel

The necessary components for solving the equations for water (liquid or ice) in soil. 

Without freeze/thaw source terms added (separately), this model reduces to
Richard's equation for liquid water. Note that the default for `θ_ice` is zero. 
Without freeze/thaw source terms added to both the liquid and ice equations, 
the default should never be changed, because we do not enforce that the total 
volumetric water fraction is less than or equal to porosity otherwise.

When freeze/thaw source terms are included, this model encompasses water in both
liquid and ice form, and water content is conserved upon phase change. 

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SoilWaterModel{FT, IF, VF, MF, HM, Fiϑl, Fiθi, BCD, BCN} <:
       AbstractWaterModel
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
    initialθ_ice::Fiθi
    "Dirichlet boundary condition structure"
    dirichlet_bc::BCD
    "Neumann boundary condition  structure"
    neumann_bc::BCN
end

"""
    SoilWaterModel(
        ::Type{FT};
        impedance_factor::AbstractImpedanceFactor{FT} = NoImpedance{FT}(),
        viscosity_factor::AbstractViscosityFactor{FT} = ConstantViscosity{FT}(),
        moisture_factor::AbstractMoistureFactor{FT} = MoistureIndependent{FT}(),
        hydraulics::AbstractHydraulicsModel{FT} = vanGenuchten{FT}(),
        initialϑ_l = (aux) -> FT(NaN),
        initialθ_ice = (aux) -> FT(NaN),
        dirichlet_bc::AbstractBoundaryFunctions = nothing,
        neumann_bc::AbstractBoundaryFunctions = nothing,
    ) where {FT}

Constructor for the SoilWaterModel. Defaults imply a constant K = K_sat model.
"""
function SoilWaterModel(
    ::Type{FT};
    impedance_factor::AbstractImpedanceFactor{FT} = NoImpedance{FT}(),
    viscosity_factor::AbstractViscosityFactor{FT} = ConstantViscosity{FT}(),
    moisture_factor::AbstractMoistureFactor{FT} = MoistureIndependent{FT}(),
    hydraulics::AbstractHydraulicsModel{FT} = vanGenuchten{FT}(),
    initialϑ_l::Function = (aux) -> eltype(aux)(NaN),
    initialθ_ice::Function = (aux) -> eltype(aux)(0.0),
    dirichlet_bc::AbstractBoundaryFunctions = nothing,
    neumann_bc::AbstractBoundaryFunctions = nothing,
) where {FT}
    args = (
        impedance_factor,
        viscosity_factor,
        moisture_factor,
        hydraulics,
        initialϑ_l,
        initialθ_ice,
        dirichlet_bc,
        neumann_bc,
    )
    return SoilWaterModel{FT, typeof.(args)...}(args...)
end


"""
    get_water_content(
        aux::Vars,
        state::Vars,
        t::Real,
        water::SoilWaterModel,
    )

Return the moisture variables for the balance law soil water model.
"""
function get_water_content(
    aux::Vars,
    state::Vars,
    t::Real,
    water::SoilWaterModel,
)
    return state.soil.water.ϑ_l, state.soil.water.θ_ice
end



"""
    get_water_content(
        aux::Vars,
        state::Vars,
        t::Real,
        water::PrescribedWaterModel,
    )

Return the moisture variables for the prescribed soil water model.
"""
function get_water_content(
    aux::Vars,
    state::Vars,
    t::Real,
    water::PrescribedWaterModel,
)
    ϑ_l = water.ϑ_l(aux, t)
    θ_ice = water.θ_ice(aux, t)
    return ϑ_l, θ_ice
end


vars_state(water::SoilWaterModel, st::Prognostic, FT) =
    @vars(ϑ_l::FT, θ_ice::FT)


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
    T = get_temperature(land.soil.heat)
    S_l = effective_saturation(
        soil.param_functions.porosity,
        water.initialϑ_l(aux),
    )
    ψ = pressure_head(
        water.hydraulics,
        soil.param_functions.porosity,
        soil.param_functions.S_s,
        water.initialϑ_l(aux),
    )
    aux.soil.water.h = hydraulic_head(aux.z, ψ)
    aux.soil.water.K =
        soil.param_functions.Ksat * hydraulic_conductivity(
            water.impedance_factor,
            water.viscosity_factor,
            water.moisture_factor,
            water.hydraulics,
            water.initialθ_ice(aux),
            soil.param_functions.porosity,
            T,
            S_l,
        )
end


function land_nodal_update_auxiliary_state!(
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    T = get_temperature(land.soil.heat)
    S_l = effective_saturation(
        soil.param_functions.porosity,
        state.soil.water.ϑ_l,
    )
    ψ = pressure_head(
        water.hydraulics,
        soil.param_functions.porosity,
        soil.param_functions.S_s,
        state.soil.water.ϑ_l,
    )
    aux.soil.water.h = hydraulic_head(aux.z, ψ)
    aux.soil.water.K =
        soil.param_functions.Ksat * hydraulic_conductivity(
            water.impedance_factor,
            water.viscosity_factor,
            water.moisture_factor,
            water.hydraulics,
            state.soil.water.θ_ice,
            soil.param_functions.porosity,
            T,
            S_l,
        )
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

    S_l = effective_saturation(
        soil.param_functions.porosity,
        state.soil.water.ϑ_l,
    )
    ψ = pressure_head(
        water.hydraulics,
        soil.param_functions.porosity,
        soil.param_functions.S_s,
        state.soil.water.ϑ_l,
    )
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

function flux_second_order!(
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux.soil.water.ϑ_l -= diffusive.soil.water.K∇h
end
