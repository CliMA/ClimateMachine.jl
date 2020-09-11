export AbstractAtmosProblem, AtmosProblem

include("boundaryconditions.jl")

abstract type AbstractAtmosProblem <: AbstractProblem end

"""
    AtmosProblem

The default problem definition (initial and boundary conditions)
for `AtmosModel`.
"""
struct AtmosProblem{BC, ISP, ISA} <: AbstractAtmosProblem
    "Boundary condition specification"
    boundarycondition::BC
    "Initial condition (function to assign initial values of prognostic state variables)"
    init_state_prognostic::ISP
    "Initial condition (function to assign initial values of auxiliary state variables)"
    init_state_auxiliary::ISA
end

function AtmosProblem(;
    boundarycondition::BC = AtmosBC(),
    init_state_prognostic::ISP = nothing,
    init_state_auxiliary::ISA = atmos_problem_init_state_auxiliary,
) where {BC, ISP, ISA}
    @assert init_state_prognostic ≠ nothing

    problem = (boundarycondition, init_state_prognostic, init_state_auxiliary)

    return AtmosProblem{typeof.(problem)...}(problem...)
end

atmos_problem_init_state_auxiliary(
    problem::AtmosProblem,
    model::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) = nothing
