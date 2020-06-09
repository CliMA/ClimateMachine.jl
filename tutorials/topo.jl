using MPI
using Logging
using LinearAlgebra
using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MPIStateArrays

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
