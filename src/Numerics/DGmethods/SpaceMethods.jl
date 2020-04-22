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
    odefun!(disc::AbstractSpaceMethod, dQ, Q, t; increment)

Evaluates the right-hand side of the spatial discretization defined by `disc` at
time `t` with state `Q`.
The result is either added into `dQ` if `increment` is true or stored in `dQ` if it is false.
Namely, the semi-discretization is of the form
``
  \\dot{Q} = F(Q, t)
``
and after the call `dQ += F(Q, t)` if `increment == true`
or `dQ = F(Q, t)` if `increment == false`

!!! note

    There is no generic implementation of this function. This must be
    implemented for each subtype of `AbstractSpaceMethod`
"""
function odefun! end

end
