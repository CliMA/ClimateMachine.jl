using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.LowStorageRungeKuttaMethod
using LinearAlgebra
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.ODESolvers

MPI.Initialized() || MPI.Init()
mpicomm = MPI.COMM_WORLD

# set up domain
topl = StackedBrickTopology(
    mpicomm,
    (0:10, 0:10, 0:3);
    periodicity = (false, false, false),
    boundary = ((1, 1), (1, 2), (1, 2)),
)

@show MPI.Comm_rank(mpicomm) length(topl.realelems)
