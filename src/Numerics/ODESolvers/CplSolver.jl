export CplSolver

"""
    CplSolver( ;component_list, cdt, t0 = 0 )

A stepping object for advancing a coupled system made up of a pre-defined 
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
"""
mutable struct CplSolver{CL, CBL, FT} <: AbstractODESolver
    "Named list of pre-defined components"
    component_list::CL
    "Named list of pre-defined export state saving callbacks"
    callback_list::CBL
    "Coupling timestep"
    dt::FT
    "Start time - initializes or tries to restart"
    t0::FT
    "Current time"
    t::FT
    "elapsed number of steps"
    steps::Int
end

function CplSolver(;component_list=component_list, callback_list=callback_list, coupling_dt=coupling_dt, t0=t0)
    return CplSolver(component_list, callback_list, coupling_dt, t0, t0 , 0 )
end

function dostep!(Qtop,
                 csolver::CplSolver,
                 param,
                 time::Real)

         # Atmos
         # - retrieve atmos import boundary state/flux from coupler
         # -  Step atmos ( solver.component_list[atmos_comp] )
         solve!(csolver.component_list.atmosphere.state,
                csolver.component_list.atmosphere.stepper;
                numberofsteps=5,
                callbacks=csolver.callback_list.atmosphere)
         # - post atmos export boundary state/flux to coupler


         # Ocean
         # - retrieve ocean import boundary state/flux from coupler
         # -  Step ocean
         solve!(csolver.component_list.ocean.state,
                csolver.component_list.ocean.stepper;
                numberofsteps=1,
                callbacks=csolver.callback_list.ocean)
         # - post ocean export boundary state/flux to coupler
          

         # Land
         # - retrieve land import boundary state/flux from coupler
         # -  Step land
         solve!(csolver.component_list.land.state,
                csolver.component_list.land.stepper;
                numberofsteps=10,
                callbacks=csolver.callback_list.land)
         # - post land export boundary state/flux to coupler

         return nothing
end
