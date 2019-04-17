module SpaceMethods

export AbstractSpaceMethod, AbstractDGMethod, odefun!

"""
    AbstractSpaceMethod

Supertype for spatial discretizations
"""
abstract type AbstractSpaceMethod end

"""
    AbstractDGMethod <: AbstractSpaceMethod

Supertype for discontinuous Galerkin spatial discretizations
"""
abstract type AbstractDGMethod <: AbstractSpaceMethod end

"""
    odefun!(disc::AbstractSpaceMethod, dQ, Q, t)

Evaluates the right-hand side of the spatial discretization defined by `disc` at
time `t` with state `Q`. The result is added into `dQ`. Namely, the
semi-discretization is of the form
``
    Q̇ = F(Q, t)
``
and after the call `dQ += F(Q, t)`

!!! note

    There is no generic implementation of this function. This must be
    implemented for each subtype of `AbstractSpaceMethod`
"""
odefun!(m::AbstractSpaceMethod, dQ, Q, t) =
throw(MethodError(odefun!, (m, dQ, Q, t)))

end
