struct CplTestModel{G,D,S,TS}
    grid::G
    discretization::D
    state::S
    stepper::TS
    nsteps::Int
end

"""
    CplTestModel(;domain,
                  equations,
                  nsteps,
                  btags=((0,0),(0,0),(1,2) ),
                  dt=1.,
                  NFSecondOrder=CentralNumericalFluxSecondOrder())

Builds an instance of a coupler test model.  This is a toy model
used for testing and designing coupling machinery. In a full-blown coupled
experiment this model would be replaced by a full compnent model.

      -  `domain` the spatial mesh used by this model. 
      -  `equations` the Balance Law used by this model. 

Each returned model instance is independent and can have its own grid,
balance law, time stepper and other attributes.  For now the code
keeps some of these things the same for initial testing, including
component timestepper and initial time (both of which need tweaking
to use for real).

"""
function CplTestModel(;
                      domain,
                      equations,
                      nsteps,
                      btags=( (0,0), (0,0), (1,2) ),
                      dt=1.,
                      NFSecondOrder=CentralNumericalFluxSecondOrder(),
                     )

        FT = eltype(domain)

        ###
        ### Instantiate the spatial grid for this component.
        ### Use ocean scripting interface convenience wrapper for now.
        ###
        grid = nothing
        grid = DiscontinuousSpectralElementGrid(
         domain;
         boundary_tags = btags
        )

        ###
        ### Create a discretization that is the union of the spatial
        ### grid and the equations, plus some numerical flux settings.
        ###
        discretization = DGModel(equations,
                                 grid,
                                 RusanovNumericalFlux(),
                                 NFSecondOrder,
                                 # CentralNumericalFluxSecondOrder(),
                                 CentralNumericalFluxGradient()
                                )

 
        ###
        ### Invoke the spatial ODE initialization functions
        ###
        state=init_ode_state(discretization, FT(0); init_on_cpu = true)

        ###
        ### Create a timestepper of the sort needed for this component.
        ### Hard coded here - but can be configurable.
        ###
        stepper=LSRK54CarpenterKennedy(
                                  discretization,
                                  state,
                                  dt=dt,
                                  t0=0.)

        ###
        ### Return a CplTestModel entity that holds all the information
        ### for a component that can be driver from a coupled stepping layer.
        ###
        return CplTestModel(
                             grid,
                             discretization,
                             state,
                             stepper,
                             nsteps,
                            )
end
