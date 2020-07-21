module Land

using CLIMAParameters
using DocStringExtensions
using LinearAlgebra, StaticArrays
using ..VariableTemplates
using ..MPIStateArrays

using ..BalanceLaws
import ..BalanceLaws:
    BalanceLaw,
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    init_state_auxiliary!,
    init_state_prognostic!,
    update_auxiliary_state!,
    nodal_update_auxiliary_state!

using ..DGMethods: LocalGeometry, DGModel

export LandModel

"""
    LandModel{PS, S, SRC} <: BalanceLaw

A BalanceLaw for land modeling.
Users may over-ride prescribed default values for each field.

# Usage

    LandModel{PS, S, SRC} <: BalanceLaw

# Fields
$(DocStringExtensions.FIELDS)
"""
struct LandModel{PS, S, SRC} <: BalanceLaw
    "Parameter set"
    param_set::PS
    "Soil model"
    soil::S
    "Source Terms (Problem specific source terms)"
    source::SRC
end

"""
    vars_state(land::LandModel, ::Prognostic, FT)

Conserved state variables (Prognostic Variables)
"""
function vars_state(land::LandModel, st::Prognostic, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end

"""
    vars_state(land::LandModel, st::Auxiliary, FT)

Names of variables required for the balance law that aren't related to
derivatives of the state variables (e.g. spatial coordinates or various
integrals) or those needed to solve expensive auxiliary equations
(e.g., temperature via a non-linear equation solve)
"""
function vars_state(land::LandModel, st::Auxiliary, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end

"""
    vars_state(land::LandModel, st::Gradient, FT)

Names of the gradients of functions of the conservative state
variables.

Used to represent values before **and** after differentiation.
"""
function vars_state(land::LandModel, st::Gradient, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end

"""
    vars_state(land::LandModel, st::GradientFlux, FT)

Names of the gradient fluxes necessary to impose Neumann boundary
conditions.
"""
function vars_state(land::LandModel, st::GradientFlux, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end


"""
    flux_first_order!(
        Land::LandModel,
        flux::Grad,
        state::Vars,
        aux::Vars,
        t::Real
    )

Computes and assembles non-diffusive fluxes in the model equations.
"""
function flux_first_order!(
    land::LandModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end


"""
    compute_gradient_argument!(
        land::LandModel,
        transform::Vars,
        state::Vars,
        aux::Vars,
        t::Real,
    )

Specify how to compute the arguments to the gradients.
"""
function compute_gradient_argument!(
    land::LandModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)

    compute_gradient_argument!(land, land.soil, transform, state, aux, t)
end

"""
    compute_gradient_flux!(
        land::LandModel,
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
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    compute_gradient_flux!(
        land,
        land.soil,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )

end

"""
    flux_second_order!(
        land::LandModel,
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
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux_second_order!(
        land,
        land.soil,
        flux,
        state,
        diffusive,
        hyperdiffusive,
        aux,
        t,
    )

end

"""
    update_auxiliary_state!(
        dg::DGModel,
        land::LandModel,
        Q::MPIStateArray,
        t::Real,
        elems::UnitRange,
    )

Perform any updates to the auxiliary variables needed at the
beginning of each time-step.
"""
function update_auxiliary_state!(
    dg::DGModel,
    land::LandModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_auxiliary_state!(
        land_nodal_update_auxiliary_state!,
        dg,
        m,
        Q,
        t,
        elems,
    )
end

"""
    land_nodal_update_auxiliary_state!(
        land::LandModel,
        state::Vars,
        aux::Vars,
        t::Real,
    )

Update the auxiliary state array.
"""
function land_nodal_update_auxiliary_state!(
    land::LandModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    land_nodal_update_auxiliary_state!(land, land.soil, state, aux, t)
end

"""
    source!(
        land::LandModel,
        source::Vars,
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        direction,n
    )
Computes (and assembles) source terms `S(Y)` in the balance law.
"""
function source!(
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    land_source!(land.source, land, source, state, diffusive, aux, t, direction)
end

include("SoilWaterParameterizations.jl")
using .SoilWaterParameterizations
include("source.jl")
include("soil_model.jl")
include("soil_heat.jl")
include("soil_water.jl")

end # Module
