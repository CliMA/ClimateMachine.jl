#####
##### Initial value problem
#####

struct InitialValueProblem{FT, IC, BC} <: AbstractSimpleBoxProblem
    Lˣ::FT
    Lʸ::FT
    H::FT
    initial_conditions::IC
    boundary_conditions::BC

    """
        InitialValueProblem(FT=Float64; dimensions, initial_conditions=InitialConditions(),
                            boundary_conditions = (OceanBC(Impenetrable(FreeSlip()), Insulating()),
                                                   OceanBC(Penetrable(FreeSlip()), Insulating())))
    
    Returns an `InitialValueProblem` with `dimensions = (Lˣ, Lʸ, H)`, `initial_conditions`,
    and `boundary_conditions`.
    
    The default `initial_conditions` are resting with no temperature perturbation;
    the default `boundary_conditions` are horizontally-periodic with `Insulating()`
    and `FreeSlip()` conditions at the top and bottom.
    """
    function InitialValueProblem{FT}(;
        dimensions,
        initial_conditions = InitialConditions(),
        boundary_conditions = (
            OceanBC(Impenetrable(FreeSlip()), Insulating()),
            OceanBC(Penetrable(FreeSlip()), Insulating()),
        ),
    ) where FT
    
        return InitialValueProblem(
            FT.(dimensions)...,
            initial_conditions,
            boundary_conditions,
        )
    end

end


#####
##### Initial conditions
#####

resting(x, y, z) = 0

struct InitialConditions{U, V, T, E}
    u::U
    v::V
    θ::T
    η::E
end


"""
    InitialConditions(; u=resting, v=resting, θ=resting, η=resting)

Stores initial conditions for each prognostic variable provided as functions of `x, y, z`.

Example
=======

# A Gaussian surface perturbation
a = 0.1 # m, amplitude
L = 1e5 # m, horizontal scale of the perturbation

ηᵢ(x, y, z) = a * exp(-(x^2 + y^2) / 2L^2)

ics = InitialConditions(η=ηᵢ)
"""
InitialConditions(; u = resting, v = resting, θ = resting, η = resting) =
    InitialConditions(u, v, θ, η)

"""
    ocean_init_state!(::HydrostaticBoussinesqModel, ic::InitialCondition, state, aux, coords, time)

Initialize the state variables `u = (u, v)` (a vector), `θ`, and `η`. Mutates `state`.

This function is called by `init_state_prognostic!(::HydrostaticBoussinesqModel, ...)`.
"""
function ocean_init_state!(
    ::HydrostaticBoussinesqModel,
    ivp::InitialValueProblem,
    state,
    aux,
    coords,
    time,
)

    ics = ivp.initial_conditions
    x, y, z = coords

    state.u = @SVector [ics.u(x, y, z), ics.v(x, y, z)]
    state.θ = ics.θ(x, y, z)
    state.η = ics.η(x, y, z)

    return nothing
end
