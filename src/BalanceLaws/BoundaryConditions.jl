"""
    BoundaryCondition

An abstract type representing a boundary condition of a [`BalanceLaw`](@ref).
"""
abstract type BoundaryCondition end

"""
    boundary_conditions(::BalanceLaw)

Either:
- a `BoundaryCondition` object, or
- a `Tuple` of `BoundaryCondition` objects: grid boundaries tagged with integer `i` will use the `i`th entry of the tuple.
"""
function boundary_conditions end

