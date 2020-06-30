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

"""
    boundary_state!(nf, ::HBModel, args...)

applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, ocean::HBModel, args...)
    boundary_condition = ocean.problem.boundary_condition
    return ocean_boundary_state!(nf, boundary_condition, ocean, args...)
end

"""
    boundary_state!(nf, ::LinearHBModel, args...)

applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, linear::LinearHBModel, args...)
    ocean = linear.ocean
    boundary_condition = ocean.problem.boundary_condition

    return ocean_boundary_state!(nf, boundary_condition, ocean, args...)
end

"""
    ocean_boundary_state!(nf, boundaries::Tuple, ::HBModel,
                          Q⁺, A⁺, n, Q⁻, A⁻, bctype)
applies boundary conditions for the first-order and gradient fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@generated function ocean_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    boundaries::Tuple,
    ocean,
    Q⁺,
    A⁺,
    n,
    Q⁻,
    A⁻,
    bctype,
    t,
    args...,
)
    N = fieldcount(boundaries)
    return quote
        Base.Cartesian.@nif(
            $(N + 1),
            i -> bctype == i, # conditionexpr
            i -> ocean_boundary_state!(
                nf,
                boundaries[i],
                ocean,
                Q⁺,
                A⁺,
                n,
                Q⁻,
                A⁻,
                t,
            ), # expr
            i -> error("Invalid boundary tag")
        ) # elseexpr
        return nothing
    end
end

"""
    ocean_boundary_state!(nf, boundaries::Tuple, ::HBModel,
                          Q⁺, A⁺, D⁺, n, Q⁻, A⁻, D⁻, bctype)
applies boundary conditions for the second-order fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@generated function ocean_boundary_state!(
    nf::NumericalFluxSecondOrder,
    boundaries::Tuple,
    ocean,
    Q⁺,
    D⁺,
    A⁺,
    n,
    Q⁻,
    D⁻,
    A⁻,
    bctype,
    t,
    args...,
)
    N = fieldcount(boundaries)
    return quote
        Base.Cartesian.@nif(
            $(N + 1),
            i -> bctype == i, # conditionexpr
            i -> ocean_boundary_state!(
                nf,
                boundaries[i],
                ocean,
                Q⁺,
                D⁺,
                A⁺,
                n,
                Q⁻,
                D⁻,
                A⁻,
                t,
            ), # expr
            i -> error("Invalid boundary tag")
        ) # elseexpr
        return nothing
    end
end

"""
    ocean_boundary_state!(nf, bc::OceanBC, ::HBModel)

splits boundary condition application into velocity and temperature conditions
"""
function ocean_boundary_state!(nf, bc::OceanBC, ocean, args...)
    ocean_velocity_boundary_state!(nf, bc.velocity, ocean, args...)
    ocean_temperature_boundary_state!(nf, bc.temperature, ocean, args...)
end

include("bc_velocity.jl")
include("bc_temperature.jl")
