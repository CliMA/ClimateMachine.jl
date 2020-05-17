using ..DGmethods.NumericalFluxes:
    NumericalFluxFirstOrder, NumericalFluxGradient, NumericalFluxSecondOrder

import ..DGmethods: boundary_state!

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

include("OceanBoundaryConditions.jl")
