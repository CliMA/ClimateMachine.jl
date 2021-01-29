export CplSolver

"""
    CplSolver( ;component_list, cdt, t0 = 0 )

A time stepping like object for advancing a coupled system made up of a pre-defined
set of named components specified in `component_list`. Each component is a
balance law, discretization and timestepper collection that can be stepped
forward by an amount `cdt` the coupling timestep. Individual components may
take several internal timesteps to advance by an amount `cdt` and may use
callbacks to accumulate boundary state information into export fields
for use by the CplSolver.

Components are registered with callbacks that export their boundary state
for use by other components as imported bondary conditions. The CplSolver
abstraction controls
 1. the outer time stepping sequencing of components
 2. actions mapping exports from one or more components to imports of
    other components

We also pass in a pre- and port- function hook that is invoked before 
and after each component runs. This may work better than a callback at the end 
of each time step. Both are here for now, while we try out designs.

Some thinking out loud notes for me -

For now components need to include slightly wasteful "shadow" variables for 
accumulating boundary flux terms they compute across RK stages and across
timesteps. The first pass of this uses a shadow vairable in source! that duplicates
some boundary flux code used in a diffusive boundary condition. The shadown variable 
is a full 3d array because of wat current infrastructure works. This can be tidied up later once
design is settled. One way to tody (pending any support for mixed sizes in core vars)
would be to call out to a function that then accesses a pre-defined MPI state array
of correct size and using A.el etc.... to determine whether we are at interface.
We could probably also do this in bc code.
"""
mutable struct CplSolver{CL, FT} <: AbstractODESolver
    "Named list of pre-defined components"
    component_list::CL
    "Coupling timestep"
    dt::FT
    "Start time - initializes or tries to restart"
    t0::FT
    "Current time"
    t::FT
    "elapsed number of steps"
    steps::Int
end

function CplSolver(;component_list=component_list, coupling_dt=coupling_dt, t0=t0)
    return CplSolver(component_list, coupling_dt, t0, t0 , 0 )
end

function dostep!(Qtop,
                 csolver::CplSolver,
                 param,
                 time::Real)

    println("Start coupled cycle")

    for cpl_component in csolver.component_list

         # Atmos
         # - retrieve atmos import boundary state/flux from coupler
         # -  Step atmos ( solver.component_list[atmos_comp] )


         # print(cpl_component)
         cpl_pre_step=cpl_component[:pre_step]
         component=cpl_component[:component_model]
         cpl_post_step=cpl_component[:post_step]

         # pre_step fetching imports goes here
         cpl_pre_step(nothing)
         solve!(component.state,
                component.stepper;
                numberofsteps=component.nsteps)
         # post step pushing exports goes here
         cpl_post_step(nothing)

    end
    return nothing
end
