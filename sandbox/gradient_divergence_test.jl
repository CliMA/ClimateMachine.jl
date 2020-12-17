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
elements = (2,1,3)
nrealelem = prod(elements)
grid = DiscontinuousSpectralElementGrid(Ω, elements = elements, polynomialorder = polynomialorder, array = ArrayType)
println(nrealelem - length(grid.interiorelems))
x, y, z = coordinates(grid)
device = array_device(x)
##

# Define Arrays
# Divergence Arrays
ijksize = prod(polynomialorder .+ 1)
Φ = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem , 3)
v_flux_divergence = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem , 1)
s_flux_divergence = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem , 1)
# Gradient Arrays
Q    = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 1)
v_∇Q = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
s_∇Q = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
##
# Test 1: Continuous function
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
@testset "Continuous: Divergence = Trace(Gradient)" begin
    total_divergence = -v_flux_divergence + s_flux_divergence
    total_trace_gradient = sum(v_∇Q - s_∇Q, dims=3)
    @test L∞(total_divergence - total_trace_gradient) < tol
end

##
# Test 2: Discontinuous function
a = 1
b = 0
c = 0
ClimateMachine.gpu_allowscalar(true)
# Heaviside Step Function Check
@. Φ[:,1,1] = 0.0 
@. Φ[:,2,1] = 1.0 
@. Φ[:,:,2] = 0.0
@. Φ[:,:,3] = 0.0 
@. Q = Φ[:,:,1]

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
@testset "Discontinuous: Divergence = Trace(Gradient)" begin
    total_divergence = -v_flux_divergence + s_flux_divergence
    total_gradient =  v_∇Q + s_∇Q
    total_trace_gradient = sum(total_gradient, dims=3)
    @test L∞(v_∇Q) < tol # since piecewise constant
    @test L∞(total_divergence - total_trace_gradient) < tol
end
##
abstract type AbstractCalculus end

struct Div{T} <: AbstractCalculus
    grid::T
end

function grad(Q::MPIStateArray, _1)
    Q_x1 = view(Q, (:,:,_1))

    event = launch_volume_gradient!(grid, v_∇Q, Q, nrealelem, device)
    wait(event)
    event = launch_interface_gradient!(grid, s_∇Q, Q, device)
    wait(event)
end

function grad(Q::MPIStateArray, _1, _2, _3)
    Q_x1 = view(Q, (:,:,_1))
    Q_x2 = view(Q, (:,:,_2))
    Q_x3 = view(Q, (:,:,_3))

    v_∇Q_x1 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    v_∇Q_x2 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    v_∇Q_x3 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)

    s_∇Q_x1 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    s_∇Q_x2 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    s_∇Q_x3 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)


    event_x1 = launch_volume_gradient!(grid, v_∇Q, Q_x1)
    event_x2 = launch_volume_gradient!(grid, v_∇Q, Q_x2)
    event_x3 = launch_volume_gradient!(grid, v_∇Q, Q_x3)

    wait(event_x1)
    wait(event_x2)
    wait(event_x3)

    event_x1 = launch_interface_gradient!(grid, s_∇Q, Q_x1)
    event_x2 = launch_interface_gradient!(grid, s_∇Q, Q_x2)
    event_x3 = launch_interface_gradient!(grid, s_∇Q, Q_x3)
    
    wait(event_x1)
    wait(event_x2)
    wait(event_x3)

    ∇Q_x1 = v_∇Q_x1 + s_∇Q_x1
    ∇Q_x2 = v_∇Q_x2 + s_∇Q_x2
    ∇Q_x3 = v_∇Q_x3 + s_∇Q_x3

    return (; ∇Q_x1, ∇Q_x2, ∇Q_x3)
end

function div(Q::MPIStateArray, _1, _2, _3)
    ∇Q_x1, ∇Q_x2, ∇Q_x3 = grad(Q, _1, _2, _3)


end

function curl(Q::MPIStateArray, _1, _2, _3)
    Q_x1 = view(Q, (:,:,_1))
    Q_x2 = view(Q, (:,:,_2))
    Q_x3 = view(Q, (:,:,_3))

    v_∇Q_x1 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    v_∇Q_x2 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    v_∇Q_x3 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)

    s_∇Q_x1 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    s_∇Q_x2 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    s_∇Q_x3 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)


    event_x1 = launch_volume_gradient!(grid, v_∇Q, Q_x1)
    event_x2 = launch_volume_gradient!(grid, v_∇Q, Q_x2)
    event_x3 = launch_volume_gradient!(grid, v_∇Q, Q_x3)

    wait(event_x1)
    wait(event_x2)
    wait(event_x3)

    event_x1 = launch_interface_gradient!(grid, s_∇Q, Q_x1)
    event_x2 = launch_interface_gradient!(grid, s_∇Q, Q_x2)
    event_x3 = launch_interface_gradient!(grid, s_∇Q, Q_x3)
    
    wait(event_x1)
    wait(event_x2)
    wait(event_x3)

    ∇Q_x1 = v_∇Q_x1 + s_∇Q_x1
    ∇Q_x2 = v_∇Q_x2 + s_∇Q_x2
    ∇Q_x3 = v_∇Q_x3 + s_∇Q_x3

    ω = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)

    ω[:,:,1] = ∇Q_x3[:,:,2] - ∇Q_x2[:,:,3]
    ω[:,:,2] = ∇Q_x1[:,:,3] - ∇Q_x3[:,:,1]
    ω[:,:,3] = ∇Q_x2[:,:,1] - ∇Q_x1[:,:,2]

end

