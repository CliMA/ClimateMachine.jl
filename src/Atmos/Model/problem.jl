export AbstractAtmosProblem, AtmosProblem

include("boundaryconditions.jl")

abstract type AbstractAtmosProblem <: AbstractProblem end

"""
    AtmosProblem

The default problem definition (initial and boundary conditions)
for `AtmosModel`.
"""
struct AtmosProblem{BCS, ISP, ISA} <: AbstractAtmosProblem
    "Boundary condition specification"
    boundaryconditions::BCS
    "Initial condition (function to assign initial values of prognostic state variables)"
    init_state_prognostic::ISP
    "Initial condition (function to assign initial values of auxiliary state variables)"
    init_state_auxiliary::ISA
end

function AtmosProblem(;
    boundaryconditions::BCS = (AtmosBC(), AtmosBC()),
    init_state_prognostic::ISP = nothing,
    init_state_auxiliary::ISA = atmos_problem_init_state_auxiliary,
) where {BCS, ISP, ISA}
    @assert init_state_prognostic ≠ nothing

    problem = (boundaryconditions, init_state_prognostic, init_state_auxiliary)

    return AtmosProblem{typeof.(problem)...}(problem...)
end

atmos_problem_init_state_auxiliary(
    problem::AtmosProblem,
    model::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) = nothing
