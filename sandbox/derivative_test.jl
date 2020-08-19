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

# functional 2D
if 2 == ndims(Ω)
    grid = DiscontinuousSpectralElementGrid(Ω, elements = (1,1), polynomialorder = (4,4), array = ArrayType)
else
    grid = DiscontinuousSpectralElementGrid(Ω, elements = (3,3,3), polynomialorder = (4,4,4), array = ArrayType)
end
x, y, z = coordinates(grid)

##
grid.sgeo[grid.n1id, :, :, :]
grid.sgeo[grid.n2id, :, :, :]
grid.sgeo[grid.n3id, :, :, :]
M = grid.vgeo[:, grid.Mid, :, :]
Mi = grid.vgeo[:, grid.MIid, :, :]

#
F1 = MPIStateArray{FT}(mpicomm, ArrayType, size(x)[1], size(x)[2] , 1)
F2 = MPIStateArray{FT}(mpicomm, ArrayType, size(x)[1], size(x)[2] , 1)
F3 = MPIStateArray{FT}(mpicomm, ArrayType, size(x)[1], size(x)[2] , 1)
F1 .= @. x * 0 + z
F2 .= @. y * 0 + x 
F3 .= @. z * 0 + y + x

ClimateMachine.gpu_allowscalar(true)

flux = MPIStateArray{FT}(mpicomm, ArrayType, size(x)[1], size(x)[2] , 3)
flux[:,:,1:1] .= F1
flux[:,:,2:2] .= F2
flux[:,:,3:3] .= F3


nrealelem = size(x)[2]
analytic_flux_divergence = @. 0 * x
v_flux_divergence = F1 .* 0
s_flux_divergence = F1 .* 0
dim = ndims(Ω)
N = round(Int, size(x)[1]^(1/dim)) - 1
dependencies = nothing
device = array_device(x)

event = launch_volume_divergence!(grid, v_flux_divergence, flux, N, nrealelem, dim, device)
wait(event)
##
event = launch_interface_divergence!(grid, s_flux_divergence, flux, N, dim, device)
wait(event)
##
# v_flux_divergece + s_flux_divergence = 0  for continuous incomprssible fluxes
allindices = collect(1:length(s_flux_divergence))
interior   = setdiff(1:length(s_flux_divergence), grid.vmap⁻)
tol = eps(1e5) 

#ClimateMachine.gpu_allowscalar(false)

# The divergence operation is : -v_flux_divergence + s_flux_divergence (note the sign convention)
# multiplying by the mass matrix makes it an integral check
L∞(x) = maximum(abs.(x))   
@testset "Code Runs" begin
    computed_divergence = -v_flux_divergence + s_flux_divergence
    @test L∞(computed_divergence - analytic_flux_divergence )< tol
    @test L∞(v_flux_divergence - s_flux_divergence) < tol
    # test to see if only the exterior terms were affected
    @test L∞(s_flux_divergence[interior]) < eps(1.0)
end
#=
##
## Figuring out what the different things mean:
if dim == 1
    Np = (N + 1)
    Nfp = 1
    nface = 2
elseif dim == 2
    Np = (N + 1) * (N + 1)
    Nfp = (N + 1)
    nface = 4
elseif dim == 3
    Np = (N + 1) * (N + 1) * (N + 1)
    Nfp = (N + 1) * (N + 1)
    nface = 6
end
# no 19
for f in [1,2,3,4]
    for e in [1]
        for n in collect(1:(N+1)^(dims-1))
        normal_vector = SVector(
            grid.sgeo[grid.n1id, n, f, e],
            grid.sgeo[grid.n2id, n, f, e],
            grid.sgeo[grid.n3id, n, f, e],
        )
        # Get surface mass, volume mass inverse
        sM, vMI = grid.sgeo[grid.sMid, n, f, e], grid.sgeo[_vMI, n, f, e]
        id⁻, id⁺ = grid.vmap⁻[n, f, e], grid.vmap⁺[n, f, e]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1
        e⁻ = ((id⁻ - 1) ÷ Np) + 1
        # not sure I understand vid⁻ and vid⁺
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1
        local_flux  = normal_vector[1] * (flux[vid⁻, e⁻, 1] + flux[vid⁺, e⁺, 1])/2
        local_flux += normal_vector[2] * (flux[vid⁻, e⁻, 2] + flux[vid⁺, e⁺, 2])/2
        local_flux += normal_vector[3] * (flux[vid⁻, e⁻, 3] + flux[vid⁺, e⁺, 3])/2
        println("----------------------")
        println("For face = ", f, ", element e = ", e, ", gl point = ", n)
        println("The (x,y,z) coordinates are ")
        println((x[id⁻], y[id⁻], z[id⁻]))
        println("The normal vector is ")
        println(normal_vector)
        println("vmap⁻=", id⁻)
        println("vid⁻=", vid⁻)
        println("local flux=", local_flux)
        println("----------------------")
        x[vid⁻, e]
        normal_vector
        end
    end
end
=#
#=
##
v_flux_divergence[grid.vmap⁻]
s_flux_divergence[grid.vmap⁻]
scatter(x[interior], y[interior])
##
Φ = reshape(flux, (N+1, N+1, size(flux)[2], size(flux)[3]))
divΦ = reshape(v_flux_divergence, (N+1, N+1, size(x)[2]))
∮divΦ = reshape(s_flux_divergence, (N+1, N+1, size(x)[2]))
## grid.sgeo[grid.sMid, :, :, :]
Φ = reshape(flux, (N+1, N+1, N+1, size(flux)[2], size(flux)[3]))
divΦ = reshape(v_flux_divergence, (N+1, N+1, N+1, size(x)[2]))
∮divΦ = reshape(s_flux_divergence, (N+1, N+1, N+1, size(x)[2]))

##
Mr = reshape(M, (N+1,N+1,N+1))
Mir = reshape(Mi, (N+1,N+1,N+1))
i = 1
j = 1 
# in the single element case
for i in 1:N+1
    for j in 1:N+1
        println("------------------")
        println("for i = ", i , " and j = ", j)
        println("The local flux 3 should be ")
        println(Mr[i,j,:] .* Φ[i,j,:,1,3])
        println("The value should be")
        println(Mir[i,j,:] .* (grid.D[1]' * ( Mr[i,j,:] .* Φ[i,j,:,1,3] )))
        println("The volume term is ")
        println(divΦ[i,j,:])
        println("The surface term is ")
        println(∮divΦ[i,j,:])
        println("------------------")
    end
end
## more details
for i in 1:N+1
    for j in 1:N+1
        for k in 1:N+1
            println("------------------")
            println("for i = ", i , " and j = ", j, " and k = ", k)
            println("The local flux 3 should be ")
            println(Mr[i,j,k] .* Φ[i,j,k,1,3])
            println("The value should be")
            tmp = Mir[i,j,k] .* (grid.D[1]' * ( Mr[i,j,:] .* Φ[i,j,:,1,3] ))
            println(tmp[k])
            println("The volume term is ")
            println(divΦ[i,j,k])
            println("The surface term is ")
            println(∮divΦ[i,j,k])
            println("------------------")
        end
    end
end

##
# this is another way to check the answer
Φ = reshape(flux, (N+1, N+1, N+1, size(flux)[2], size(flux)[3]))
∂ˣΦ = zeros(N+1, N+1, N+1, size(flux)[2])
metricterm = grid.vgeo[:, 1, :, :]
for j in 1:(N+1)
    for k in 1:(N+1)
        for e in 1:size(flux)[2]
            ∂ˣΦ[:, j, k, e] = grid.D[1] * Φ[:, j, k, e, 1] * metricterm[1]
        end
    end
end
check = reshape(∂ˣΦ, ((N+1)^dim, size(flux)[2]))
check - analytic_flux_divergence
##
# The check
v_flux_divergence - s_flux_divergence

=#