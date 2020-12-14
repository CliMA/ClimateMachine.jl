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
Ω = Circle(-1,1) × Circle(-1,1) #  × Circle(-1,1)
dims = ndims(Ω)

ClimateMachine.gpu_allowscalar(true)

# the grid data structure contains 37 members
# (6) Surface structures
# sgeo, and the 5 ids
# (17) Volume structures
# vgeo, and the 16 ids
# (2) Volume to Surface index
# vmap⁻, vmap⁺
# (1) Topology
# topology
# (3) Operators
# grid.D (differentiation matrix), grid.Imat (integration matrix?), grid.ω (quadrature weights)
# information (1)
# activedofs (active degrees of freedom)
# Domain Boundary Information (1)
# elemtobndy (element to boundary)
# MPI stuff (6)
# interiorelems, exteriorelems, nabrtovmaprecv, nabrtovmapsend, vmaprecv, vmapsendv 

# sgeo, the first index is for the ids
# the second index are the gauss-lobatto entries
# the third index is the face
# the last index is the element
# the ids are: (1, n1id), (2 n2id), (3 n3id) (4, sMid), (5, vMIid)
# grid.sgeo[1,:,:,:] are normals in the n1 direction
# norm(grid.vgeo[:, grid.MIid, :][grid.vmap⁻] - grid.sgeo[5,:,:,:]) is zero
# x[grid.vmap⁻[10]] is connected to x[grid.vmap⁺[10]]

# vgeo, first index is guass-lobatto points
# second index is ids
# third index is elements
# the are are 16 ids
# 1-9 are metric terms
# 10 is the mass matrix
# 11 is the inverse mass matrix
# 12 is MHid (horizontal mass matrix?)
# 13-15 are x1 x2 x3
# 16 is the vertical volume jacobian

# Define Grid: might want to loop over element sizes and polynomial orders
if 2 == ndims(Ω)
    grid = DiscontinuousSpectralElementGrid(Ω, elements = (1,1), polynomialorder = (4,4), array = ArrayType)
else
    grid = DiscontinuousSpectralElementGrid(Ω, elements = (10,3,3), polynomialorder = (4,4,4), array = ArrayType)
end
x, y, z = coordinates(grid)
nrealelem = size(x)[2]
device = array_device(x)
dim = ndims(Ω)
N = round(Int, size(x)[1]^(1/dim)) - 1
dependencies = nothing

# Initialize State s
F1 = MPIStateArray{FT}(mpicomm, ArrayType, size(x)[1], size(x)[2] , 1)
F2 = MPIStateArray{FT}(mpicomm, ArrayType, size(x)[1], size(x)[2] , 1)
F3 = MPIStateArray{FT}(mpicomm, ArrayType, size(x)[1], size(x)[2] , 1)
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

event = launch_volume_divergence!(grid, v_flux_divergence, flux, nrealelem, device)
wait(event)
event = launch_interface_divergence!(grid, s_flux_divergence, flux, device)
wait(event)

tol = eps(1e5) 
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

event = launch_volume_divergence!(grid, v_flux_divergence, flux, nrealelem, device)
wait(event)
event = launch_interface_divergence!(grid, s_flux_divergence, flux, device)
wait(event)

tol = 1e-3 # for this test should be a function of polynomial order / element size
# The divergence operation is : -v_flux_divergence + s_flux_divergence (note the sign convention)  
@testset "Compressible Flow Field" begin
    computed_divergence = -v_flux_divergence + s_flux_divergence
    @test L∞(computed_divergence - analytic_flux_divergence ) < tol
    # test to see if only the exterior terms were affected
    @test L∞(s_flux_divergence[interior]) < eps(1.0)
end
