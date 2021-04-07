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
    physics = nothing,
    boundaryconditions = nothing,
    init_state_prognostic::ISP = nothing,
    init_state_auxiliary::ISA = atmos_problem_init_state_auxiliary,
) where {ISP, ISA}
    @assert init_state_prognostic ≠ nothing
    if boundaryconditions == nothing
        @assert physics ≠ nothing
        boundaryconditions = (AtmosBC(physics), AtmosBC(physics))
    end

    problem = (boundaryconditions, init_state_prognostic, init_state_auxiliary)

    return AtmosProblem{typeof.(problem)...}(problem...)
end

atmos_problem_init_state_auxiliary(
    problem::AtmosProblem,
    model::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) = nothing
