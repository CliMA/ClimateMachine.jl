using ClimateMachine
using ClimateMachine.MPIStateArrays
using ClimateMachine.Diagnostics

ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
dim = 3
FT = Float64
Nh = 3
Nv = 3
# Ω =  Circle(0,2pi) × Circle(0,2π)
periodicity = ntuple(j -> false, dim)
# ImperoGrid(Ω, numberofelements = (Ne,Ne), poloynomialorder = (Np1, Np2), floatype=, ...)
brickrange = (
    ntuple(
        j -> range(FT(-1); length = Nh + 1,
            stop = 1),
        dim - 1,
    )...,
    range(FT(-5); length = Nv + 1, stop = 5),
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
Np = N+1
grid = DiscontinuousSpectralElementGrid(
    topl,
    FloatType = FT,
    DeviceArray = ArrayType,
    polynomialorder = N,
)
numelem = length(grid.topology.realelems)

Q = MPIStateArray{Float64}(mpicomm, ArrayType, Np^dim, 3, numelem)
d1 = Diagnostics.VectorGradient(grid, Q, 1)


# ∇ = Operator(nothing, GradientMetaData("name"))
# curl()

#=
_ρ, _ρu, _ρv, _ρw = ind[1], ind[2], ind[3], ind[4]
Nq = N + 1
nrealelem = length(grid.topology.realelems)

Q = init_ode_state(dg, )

d1 = Diagnostics.VectorGradient(grid, Q, _ρu)
d2 = Diagnostics.VectorGradient(grid, Q, _ρv)
d3 = Diagnostics.VectorGradient(grid, Q, _ρw)
vgrad = Diagnostics.VectorGradients(d1, d2, d3)
vort = Diagnostics.Vorticity(dg, vgrad)
=#