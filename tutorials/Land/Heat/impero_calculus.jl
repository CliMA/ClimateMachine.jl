using ClimateMachine
using ClimateMachine.MPIStateArrays

ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
dim = 2
FT = Float64
Ne = 2
periodicity = ntuple(j -> false, dim)
brickrange = (
    ntuple(
        j -> range(FT(-1); length = Ne + 1,
            stop = 1),
        dim - 1,
    )...,
    range(FT(-5); length = 5Ne + 1, stop = 5),
)

topl = StackedBrickTopology(
                            mpicomm,
                            brickrange;
                            periodicity = periodicity,
                            boundary = (
                                ntuple(j -> (1, 2), dim - 1)...,
                                (3, 4),
                            )
)
N = 1
grid = DiscontinuousSpectralElementGrid(
    topl,
    FloatType = FT,
    DeviceArray = ArrayType,
    polynomialorder = N,
)