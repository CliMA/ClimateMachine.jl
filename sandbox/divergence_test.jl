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


# Define Grid: might want to loop over element sizes and polynomial orders
grid = DiscontinuousSpectralElementGrid(Ω, elements = (10,10,10), polynomialorder = (4,4,4), array = ArrayType)

x, y, z = coordinates(grid)
device = array_device(x)
dim = ndims(Ω)
N = round(Int, size(x)[1]^(1/dim)) - 1
dependencies = nothing

# Initialize State s
F1 = MPIStateArray{FT}(mpicomm, ArrayType, size(x)[1], size(x)[2], 1)
F2 = MPIStateArray{FT}(mpicomm, ArrayType, size(x)[1], size(x)[2], 1)
F3 = MPIStateArray{FT}(mpicomm, ArrayType, size(x)[1], size(x)[2], 1)
flux = MPIStateArray{FT}(mpicomm, ArrayType, size(x)[1], size(x)[2] , 3)

v_flux_divergence = F1 .* 0
s_flux_divergence = F1 .* 0 

allindices = collect(1:length(s_flux_divergence))
interior   = setdiff(1:length(s_flux_divergence), grid.vmap⁻)

## Test Block 1: Incompressible Flow Field
# Impero should make this nicer
@. F1 = z
@. F2 = x 
@. F3 = y + x

flux[:,:,1:1] .= F1
flux[:,:,2:2] .= F2
flux[:,:,3:3] .= F3
analytic_flux_divergence = @. 0 * x

event = launch_volume_divergence!(grid, v_flux_divergence, flux)
wait(event)
event = launch_interface_divergence!(grid, s_flux_divergence, flux)
wait(event)

tol = eps(1e6) 
L∞(x) = maximum(abs.(x))
# The divergence operation is : -v_flux_divergence + s_flux_divergence (note the sign convention)  
@testset "Incompressible Flow Field" begin
    computed_divergence = -v_flux_divergence + s_flux_divergence
    @test L∞(computed_divergence - analytic_flux_divergence ) < tol
    @test L∞(v_flux_divergence - s_flux_divergence) < tol
    # test to see if only the exterior terms were affected
    @test L∞(s_flux_divergence[interior]) < eps(1.0)
end

## Test Block 2: Compressible Flow Field
@. F1 =  sin(π*x)
@. F2 =  sin(π*z) 
@. F3 =  cos(π*y)
ClimateMachine.gpu_allowscalar(true)
flux[:,:,1:1] .= F1
flux[:,:,2:2] .= F2
flux[:,:,3:3] .= F3
analytic_flux_divergence = @. π*cos(π*x)

event = launch_volume_divergence!(grid, v_flux_divergence, flux)
wait(event)
event = launch_interface_divergence!(grid, s_flux_divergence, flux)
wait(event)

tol = 1e-3 # for this test should be a function of polynomial order / element size
# The divergence operation is : -v_flux_divergence + s_flux_divergence (note the sign convention)  
@testset "Compressible Flow Field" begin
    computed_divergence = -v_flux_divergence + s_flux_divergence
    @test L∞(computed_divergence - analytic_flux_divergence ) < tol
    # test to see if only the exterior terms were affected
    @test L∞(s_flux_divergence[interior]) < eps(1.0)
end

## Test Block 3: Compressible Flow Field
a = 1
b = 1 
c = 1
@. F1 =  a * sin(π*x)
@. F2 =  b * sin(π*y) 
@. F3 =  c * cos(π*z)
ClimateMachine.gpu_allowscalar(true)
flux[:,:,1:1] .= F1
flux[:,:,2:2] .= F2
flux[:,:,3:3] .= F3
analytic_flux_divergence = @. a * π*cos(π*x) + b*π*cos(π*y) - c*π*sin(π*z)

event = launch_volume_divergence!(grid, v_flux_divergence, flux)
wait(event)
event = launch_interface_divergence!(grid, s_flux_divergence, flux)
wait(event)

tol = 1e-3 # for this test should be a function of polynomial order / element size
# The divergence operation is : -v_flux_divergence + s_flux_divergence (note the sign convention)  
@testset "Compressible Flow Field" begin
    computed_divergence = -v_flux_divergence + s_flux_divergence
    @test L∞(computed_divergence - analytic_flux_divergence ) < tol
    # test to see if only the exterior terms were affected
    @test L∞(s_flux_divergence[interior]) < eps(1.0)
end
