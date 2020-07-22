#### Soil model

export SoilModel

"""
    SoilModel{W, H} <: BalanceLaw

A BalanceLaw for soil modeling.
Users may over-ride prescribed default values for each field.

# Usage

    SoilModel{W, H} <: BalanceLaw

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SoilModel{W, H} <: BalanceLaw
    "Water model"
    water::W
    "Heat model"
    heat::H
end

"""
    vars_state(soil::SoilModel, st::Prognostic, FT)

Conserved state variables (Prognostic Variables)
"""
function vars_state(soil::SoilModel, st::Prognostic, FT)
    @vars begin
        water::vars_state(soil.water, st, FT)
        heat::vars_state(soil.heat, st, FT)
    end
end

"""
    vars_state(soil::SoilModel, st::Auxiliary, FT)

Names of variables required for the balance law that aren't related to
derivatives of the state variables (e.g. spatial coordinates or various
integrals) or those needed to solve expensive auxiliary equations
(e.g., temperature via a non-linear equation solve)
"""
function vars_state(soil::SoilModel, st::Auxiliary, FT)
    @vars begin
        water::vars_state(soil.water, st, FT)
        heat::vars_state(soil.heat, st, FT)
    end
end

"""
    vars_state(soil::SoilModel, st::Gradient, FT)

Names of the gradients of functions of the conservative state variables.
Used to represent values before **and** after differentiation
"""
function vars_state(soil::SoilModel, st::Gradient, FT)
    @vars begin
        water::vars_state(soil.water, st, FT)
        heat::vars_state(soil.heat, st, FT)
    end
end

"""
    vars_state(soil::SoilModel, st::GradientFlux, FT)

Names of the gradient fluxes necessary to impose Neumann boundary conditions
"""
function vars_state(soil::SoilModel, st::GradientFlux, FT)
    @vars begin
        water::vars_state(soil.water, st, FT)
        heat::vars_state(soil.heat, st, FT)
    end
end

"""
    flux_first_order!(
        Land::LandModel,
        soil::SoilModel,
        flux::Grad,
        state::Vars,
        aux::Vars,
        t::Real
    )

Computes and assembles non-diffusive fluxes in the model equations.
"""
function flux_first_order!(
    land::LandModel,
    soil::SoilModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end

"""
    compute_gradient_argument!(
        land::LandModel,
        soil::SoilModel,
        transform::Vars,
        state::Vars,
        aux::Vars,
        t::Real,
    )

Specify how to compute the arguments to the gradients.
"""
function compute_gradient_argument!(
    land::LandModel,
    soil::SoilModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)

    #   compute_gradient_argument!(
    #     land,
    #     soil.heat,
    #     transform,
    #     state,
    #     aux,
    #     t,
    # )
    #   compute_gradient_argument!(
    #     land,
    #     soil.water,
    #     transform,
    #     state,
    #     aux,
    #     t,
    # )
end

"""
    compute_gradient_flux!(
        land::LandModel,
        soil::SoilModel,
        diffusive::Vars,
        ∇transform::Grad,
        state::Vars,
        aux::Vars,
        t::Real,
    )

Specify how to compute gradient fluxes.
"""
function compute_gradient_flux!(
    land::LandModel,
    soil::SoilModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    #   compute_gradient_flux!(
    #     land,
    #     soil.heat,
    #     diffusive,
    #     ∇transform,
    #     state,
    #     aux,
    #     t,
    # )
    #   compute_gradient_flux!(
    #     land,
    #     soil.water,
    #     diffusive,
    #     ∇transform,
    #     state,
    #     aux,
    #     t,
    # )

end

"""
    flux_second_order!(
        land::LandModel,
        soil::SoilModel,
        flux::Grad,
        state::Vars,
        diffusive::Vars,
        hyperdiffusive::Vars,
        aux::Vars,
        t::Real,
    )

Specify the second order flux for each conservative state variable
"""
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
    # flux_second_order!(
    #     land,
    #     soil.heat,
    #     flux,
    #     state,
    #     diffusive,
    #     hyperdiffusive,
    #     aux,
    #     t,
    # )
    # flux_second_order!(
    #     land,
    #     soil.water,
    #     flux,
    #     state,
    #     diffusive,
    #     hyperdiffusive,
    #     aux,
    #     t,
    # )

end

"""
    land_nodal_update_auxiliary_state!(
        land::LandModel,
        soil::SoilModel,
        state::Vars,
        aux::Vars,
        t::Real,
    )

Update the auxiliary state array
"""
function land_nodal_update_auxiliary_state!(
    land::LandModel,
    soil::SoilModel,
    state::Vars,
    aux::Vars,
    t::Real,
)

end
