
"""
    Parameters

Data structure containing the spatial and temporal parameter structures

!!! note
We may want to update this with Abstract types for the space and time parameters
once we upgrade the project to Julia 1.1 since `Base.@kwdefs` will support
parametric structs and structs with supertypes.

"""
# abstract type AbstractSpaceParameter end # Should we have this?
# abstract type AbstractTimeParameter end # Should we have this?
struct Parameters{SpaceMethod, TimeMethod}
  "Parameters structure for the spacial discretization"
  space
  "Parameters structure for the temporal discretization"
  time
end
function Parameters(spacemethod, spaceargs, timemethod, timeargs)
  space = createparameters(Val(spacemethod); spaceargs...)
  time = createparameters(Val(timemethod); timeargs...)
  Parameters{Val(spacemethod), Val(timemethod)}(space, time)
end

"""
    Configuration

Data structure containing the spatial and temporal configuration structures

"""
abstract type AbstractSpaceConfiguration end
abstract type AbstractTimeConfiguration end
struct Configuration{S <: AbstractSpaceConfiguration,
                     T <: AbstractTimeConfiguration}
  "Configuration structure for the spacial discretization"
  space::S
  "Configuration structure for the temporal discretization"
  time::T
end
function Configuration(params::Parameters, mpicomm)
  space = createconfiguration(params.space, mpicomm)
  time = createconfiguration(params.time, mpicomm, (space, params.space))
  Configuration(space, time)
end

"""
    State

Data structure containing the spatial and temporal state structures

"""
abstract type AbstractSpaceState end
abstract type AbstractTimeState end
struct State{S <: AbstractSpaceState,
             T <: AbstractTimeState}
  "State structure for the spacial discretization"
  space::S
  "State structure for the temporal discretization"
  time::T
end
function State(config::Configuration, params::Parameters)
  space = createstate(config.space, params.space)
  time = createstate(config.time, params.time,
                     (space, config.space, params.space))
  State(space, time)
end

# Some convenience function wrappers
L2solutionnorm(state::State, config::Configuration,
               params::Parameters, x...; kw...) =
L2solutionnorm(state.space, config.space, params.space, x...; kw...)

# TODO: Document these more fully and specify what exactly should be implemented
# Function to be implemented by space discretization
# REQUIRED (called by library)
createparameters() = error("no implementation")
createconfiguration() = error("no implementation")
createstate() = error("no implementation")
gettime() = error("no implementation")
settime!() = error("no implementation")
getQ() = error("no implementation")
similarQ() = error("no implementation")
getmesh() = error("no implementation")
rhs!() = error("no implementation")

# OPTIONAL (called by user)
initspacestate!() = error("no implementation")
estimatedt() = error("no implementation")
writevtk() = error("no implementation")
L2solutionnorm() = error("no implementation")


# Function to be implemented by time discretization
# REQUIRED (called by library)
run!() = error("no implementation")
# createparameters() = error("no implementation")
# createconfiguration() = error("no implementation")
# createstate() = error("no implementation")

# OPTIONAL (called by user)
inittimestate!() = error("no implementation")
printtimeinfo() = error("no implementation")

