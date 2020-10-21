using Test
using ClimateMachine
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MPIStateArrays
using ClimateMachine.Abstractions
import ClimateMachine.Abstractions: DiscontinuousSpectralElementGrid

using Impero, Printf, MPI

#import ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid

ClimateMachine.init()

const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD

Ω = Circle(0,2π) × Circle(0,2π) # Ω = S¹(0,2π) × Interval(-1,1) × Interval(-2,2), Ω = Earth()
# error messages
Abstractions.DiscontinuousSpectralElementGrid(Ω)
Abstractions.DiscontinuousSpectralElementGrid(Ω, elements = (10,10,10))
Abstractions.DiscontinuousSpectralElementGrid(Ω, elements = (10,10,10), polynomialorder = (3,3,3))
Abstractions.DiscontinuousSpectralElementGrid(Ω, elements = (10,10), polynomialorder = (3,4))
Abstractions.DiscontinuousSpectralElementGrid(Ω×Ω, elements = (10,10,10,10), polynomialorder = (3,3,3,3))
# functional 2D
grid = Abstractions.DiscontinuousSpectralElementGrid(Ω, elements = (10,10), polynomialorder = (4,4), array = ArrayType)
# functional 3D
Ω = Circle(0,2π) × Circle(0,2π) × Interval(-1,1)
grid = Abstractions.DiscontinuousSpectralElementGrid(Ω, elements = (2,2,2), polynomialorder = (4,4,4), array = ArrayType)
