using ...DGMethods.NumericalFluxes

import ...BalanceLaws: boundary_state!

export OceanBC

"""
    OceanBC(velocity    = Impenetrable(NoSlip())
            temperature = Insulating())

The standard boundary condition for OceanModel. The default options imply a "no flux" boundary condition.
"""
Base.@kwdef struct OceanBC{M, T}
    velocity::M = Impenetrable(NoSlip())
    temperature::T = Insulating()
end

@inline function boundary_conditions(ocean::HBModel)
    return ocean.problem.boundary_condition
end

@inline function boundary_conditions(linear::LinearHBModel)
    return linear.ocean.problem.boundary_condition
end

"""
    ocean_boundary_state!(nf, bc::OceanBC, ::HBModel)

splits boundary condition application into velocity and temperature conditions
"""
function boundary_state!(nf, bc::OceanBC, ocean::HBModel, args...)
    ocean_velocity_boundary_state!(nf, bc.velocity, ocean, args...)
    ocean_temperature_boundary_state!(nf, bc.temperature, ocean, args...)
end

"""
    boundary_state!(nf, ::LinearHBModel, args...)

applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, bc::OceanBC, lm::LinearHBModel, args...)
    return boundary_state!(nf, bc, lm.ocean, args...)
end

include("bc_velocity.jl")
include("bc_temperature.jl")
