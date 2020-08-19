using Test
using ClimateMachine
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MPIStateArrays
using ClimateMachine.Abstractions
import ClimateMachine.Abstractions: DiscontinuousSpectralElementGrid
using Impero, Printf, MPI, LinearAlgebra, Statistics,Plots
include(pwd() * "/sandbox/test_utils.jl")
ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD
Ω = Circle(0,1) × Circle(0,1)

# functional 2D
n = 4
grid = DiscontinuousSpectralElementGrid(Ω, elements = (5,5), polynomialorder = (n,n), array = ArrayType)
x, y, z = coordinates(grid)
M = view(grid.vgeo, :, grid.Mid, :)    # mass matrix
xC, yC, zC = cell_centers(grid)
volumes = sum(M, dims = 1)[:]
check_x = cell_average(x, M = M)
norm(xC - check_x)
power = 10
tmp = cell_average(x.^power, M = M)
sum(tmp .* volumes) - 1/(power+1)

##
# sgeo, the first index is for the ids
# the second index are the entries
# the third index is the face
# the last index is the element
grid.sgeo[grid.sMid, :, :, :]
surface = sum(grid.sgeo[grid.sMid,:,:,:] .* x[grid.vmap⁻], dims = 1)
##
face = 1
element = 1
approx = surface[1, face, element]
# face 1 and 2 are constant x
# face 3 and 4 are constant y
# here we are integrating the function x
if face > 2
    exact = 0.5*(x[grid.vmap⁻[n+1, face, element]]^2 - x[grid.vmap⁻[1, face, element]]^2)
else
    exact = x[grid.vmap⁻[n+1, face, element]] * (y[grid.vmap⁻[n+1, face, element]] - y[grid.vmap⁻[1, face, element]])
end
println(norm(approx - exact))
gr(size=(300,300))
p1 = scatter(x[:], y[:], label = false, color = :black)
scatter!(x[grid.vmap⁻[1:n+1,face,element]], y[grid.vmap⁻[1:n+1,face,element]], color = :red, label = false)
