"""
    Coupling

Primitive coupling module sufficient for initial atmos-ocean-land coupled simulation.
"""
module Coupling

using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.ODESolvers

import ClimateMachine.Ocean.Domains: DiscontinuousSpectralElementGrid

struct CplTestModel{G,E,S,TS}
         grid::G
         equations::E
         state::S
         stepper::TS
end

"""
    CplTestModel(;domain=nothing,BL_module=nothing)

Builds an instance of the coupling experimentation test model. Each
returned model instance is independent and can have its own grid, 
balance law, time stepper and other attributes.  For now the code 
keeps some of these things the same for initial testing.
"""
function CplTestModel(;
                      domain=nothing,
                      BL_module=nothing,
                     )

        FT = eltype(domain)

        # Instantiate the spatial grid for this component.
        # Use ocean scripting interface convenience wrapper for now.
        grid = nothing
        grid = DiscontinuousSpectralElementGrid(
         domain;
        )

        # Instantiate the equations for this component.
        # Assume that these can be held in a top-level balance law 
        # that exports prop_defaults() and LAW{}. prop_defaults() is
        # a named tuple that can be used to configure internal functions.
        # LAW{} is an alias for the specific type used for the overall
        # balance law.
        equations=nothing
        bl_prop=BL_module.prop_defaults()
        equations=bl_prop.LAW{FT}(;bl_prop=bl_prop)

        # Create a discretization that is the union of the spatial
        # grid and the equations, plus some numerical flux settings.
        # Settings should be a set of configurable parameters passsed in.
        discretization=nothing
        discretization = DGModel(equations,
                                 grid,
                                 RusanovNumericalFlux(),
                                 CentralNumericalFluxSecondOrder(),
                                 CentralNumericalFluxGradient()
                                )


        # Invoke the spatial ODE initialization functions
        state=nothing
        state=init_ode_state(discretization, FT(0); init_on_cpu = true)

        # Create a timestepper of the sort needed for this component.
        # Hard coded here - but can be configurable.
        stepper=nothing
        stepper=LSRK54CarpenterKennedy(
                                  discretization,
                                  state,
                                  dt=1.,
                                  t0=0.)

        # Return a CplTestModel entity that holds all the information
        # for a component that can be driver from a coupled stepping layer.
        return CplTestModel(
                             grid,
                             equations,
                             state,
                             stepper,
                            )
end

end
