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
include(pwd() * "/sandbox/gradient_test_utils.jl")

ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD
const FT = Float64
Ω = Circle(-1,1) × Circle(-1,1) × Circle(-1,1)
dims = ndims(Ω)

ClimateMachine.gpu_allowscalar(true)

polynomialorder = (3,3,3)
elements = (1,1,1)
nrealelem = prod(elements)
grid = DiscontinuousSpectralElementGrid(Ω, elements = elements, polynomialorder = polynomialorder, array = ArrayType)
x, y, z = coordinates(grid)
device = array_device(x)

# Define Arrays
# Divergence Arrays
ijksize = prod(polynomialorder .+ 1)
Φ = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem , 3)
v_flux_divergence = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem , 1)
s_flux_divergence = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem , 1)
# Gradient Arrays
Q  = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 1)
v_∇Q = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
s_∇Q = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
##
# Test 1: Check for same numerical computation
a = 1
b = 1
c = 1
ClimateMachine.gpu_allowscalar(true)
@. Φ[:,:,1] = a * sin(π*x)
@. Φ[:,:,2] = b * sin(π*y)
@. Φ[:,:,3] = c * cos(π*z)
@. Q = a * sin(π*x) + b * sin(π*y) + c * cos(π*z)

# Divergence
event = launch_volume_divergence!(grid, v_flux_divergence, Φ, nrealelem, device)
wait(event)
event = launch_interface_divergence!(grid, s_flux_divergence, Φ, device)
wait(event)

# Gradient
event = launch_volume_gradient!(grid, v_∇Q, Q, nrealelem, device)
wait(event)
event = launch_interface_gradient!(grid, s_∇Q, Q, device)
wait(event)

tol = eps(1e5)
L∞(x) = maximum(abs.(x))
@testset "Divergence = Trace(Gradient)" begin
    total_divergence = -v_flux_divergence + s_flux_divergence
    total_trace_gradient = sum(v_∇Q - s_∇Q, dims=3)
    @test L∞(total_divergence - total_trace_gradient) < tol
end
