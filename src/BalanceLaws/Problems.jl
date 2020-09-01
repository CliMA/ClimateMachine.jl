module Problems

export AbstractProblem,
    init_state_prognostic!, init_state_auxiliary!, boundary_state!

"""
    AbstractProblem

An abstract type representing the initial conditions and
the boundary conditions for a `BalanceLaw`.

Subtypes `P` should define the methods below.
"""
abstract type AbstractProblem end

"""
    init_state_prognostic!(
        ::P,
        ::BalanceLaw,
        state_prognostic::Vars,
        state_auxiliary::Vars,
        coords,
        t,
        args...,
    )

Initialize the prognostic state variables at ``t = 0``.
"""
function init_state_prognostic! end

"""
    init_state_auxiliary!(
        ::P,
        ::BalanceLaw,
        state_auxiliary::Vars,
        geom::LocalGeometry,
    )

Initialize the auxiliary state, at ``t = 0``.
"""
function init_state_auxiliary! end

end # module
