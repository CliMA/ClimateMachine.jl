# How to configure a coupled system - UNDER DEVELOPMENT
  
Within CLiMA we are building a minimal coupling library that can interface with DG BalanceLaw based models
and potentially with models that do not use the Balance Law machinery.

The ClimateMachine.jl Coupler consists of three main pieces

1. A time-stepper [CplSolver.j](src/Numerics/ODESolvers/CplSolver.jl). This is a custom 
   time-stepper that is designed to step forward a list of component models, for example 
   Atmosphere, Ocean and Land components, following some specified sequence.

2. A module [Coupling](src/Coupling/Coupling.jl). This is a module that defines a distinct type
   [CplState](src/Coupling/CplState.jl) that is used to hold state that passes between 
   components during coupling. For example the CplState type can be used to hold Ocean component
   SST values that an ocean model exports. These SST values can then be imported by the 
   Atmospheric component for use as boundary conditions. The [Coupling](src/Coupling/Coupling.jl) module
   also defines the API through which components can export fields to and import fields from 
   the coupler. 
   
3. A set of [example experiments](experiments/CouplingDesignTests/). These use the Coupler to 
   carry out simulations involving multiple components.
   
The time-stepper and module pieces that make up the Coupler are designed to be generic. 
They can support several different coupling scenarios depending on needs of the CLiMA project.

The example experiments show how to use the Coupler for specific combinations of models and specific
problems. The initial example experiments being developed aim to illustrate 



