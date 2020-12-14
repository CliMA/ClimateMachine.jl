using Test
using ClimateMachine
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MPIStateArrays
using ClimateMachine.Abstractions
import ClimateMachine.Abstractions: DiscontinuousSpectralElementGrid
using Impero, Printf, MPI, LinearAlgebra, Statistics, GaussQuadrature
include(pwd() * "/sandbox/test_utils.jl")
include(pwd() * "/sandbox/abstract_kernels.jl")

ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD
const FT = Float64
Ω = Circle(-1,1) × Circle(-1,1) × Circle(-1,1)
dims = ndims(Ω)

ClimateMachine.gpu_allowscalar(true)

if 2 == ndims(Ω)
    grid = DiscontinuousSpectralElementGrid(Ω, elements = (1,1), polynomialorder = (4,4), array = ArrayType)
else
    grid = DiscontinuousSpectralElementGrid(Ω, elements = (10,3,3), polynomialorder = (4,4,4), array = ArrayType)
end

x, y, z = coordinates(grid)
nrealelem = size(x)[2] # fix this later to only depend on grid intrinsically
ijksize = prod(polynomialorders(grid) .+ 1)
device = array_device(x)
dim = ndims(Ω)
N = round(Int, size(x)[1]^(1/dim)) - 1
dependencies = nothing

# Initialize State s
Q  = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 1)
∇Q = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
exact_∇Q = copy(∇Q)
##
event = launch_volume_gradient!(grid, ∇Q, Q, nrealelem, device)
wait(event)

## Test Block 1: Volume Test
@. Q[:,:, 1] = sin(π*x)
@. exact_∇Q[:,:, 1] =  π*cos(π*x)
@. exact_∇Q[:,:, 2] =  0.0 
@. exact_∇Q[:,:, 3] =  0.0

event = launch_volume_gradient!(grid, ∇Q, Q, nrealelem, device)
wait(event)
tol = eps(1e5) 
L∞(x) = maximum(abs.(x))
println(L∞(∇Q - exact_∇Q))
@testset "Gradient Test" begin
    @test L∞(∇Q - exact_∇Q) < tol
end
