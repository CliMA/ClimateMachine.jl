using ..DGmethods.NumericalFluxes:
    NumericalFluxNonDiffusive, NumericalFluxGradient, NumericalFluxDiffusive

import ..DGmethods: boundary_state!

"""
    boundary_state!(nf, ::OceanModel, args...)

applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, ocean::OceanModel, args...)
    boundary_condition = ocean.problem.boundary_condition
    return ocean_boundary_state!(nf, boundary_condition, ocean, args...)
end

"""
    boundary_state!(nf, ::HorizontalModel, args...)

applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, horizontal::HorizontalModel, args...)
    ocean = horizontal.ocean
    boundary_condition = ocean.problem.boundary_condition

    return ocean_boundary_state!(nf, boundary_condition, ocean, args...)
end

"""
    boundary_state!(nf, ::BarotropiclModel, args...)

applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, barotropic::BarotropicModel, args...)
    ocean = barotropic.baroclinic
    boundary_condition = ocean.problem.boundary_condition

    return ocean_boundary_state!(nf, boundary_condition, barotropic, args...)
end

"""
    boundary_state!(nf, ::LinearVerticalModel, args...)

applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, linear::LinearVerticalModel, args...)
    ocean = linear.ocean
    boundary_condition = ocean.problem.boundary_condition

    return ocean_boundary_state!(nf, boundary_condition, ocean, args...)
end

"""
    ocean_boundary_state!(nf, boundaries::Tuple, ::OceanModel,
                          Q⁺, A⁺, n, Q⁻, A⁻, bctype)
applies boundary conditions for the first-order and gradient fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@generated function ocean_boundary_state!(
    nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
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
    ocean_boundary_state!(nf, boundaries::Tuple, ::OceanModel,
                          Q⁺, A⁺, D⁺, n, Q⁻, A⁻, D⁻, bctype)
applies boundary conditions for the second-order fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@generated function ocean_boundary_state!(
    nf::NumericalFluxDiffusive,
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
