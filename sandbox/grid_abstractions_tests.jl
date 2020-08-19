using Test
using ClimateMachine
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MPIStateArrays
using ClimateMachine.Abstractions
import ClimateMachine.Abstractions: DiscontinuousSpectralElementGrid
using Impero, Printf, MPI, LinearAlgebra, Statistics
include(pwd() * "/sandbox/test_utils.jl")
ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD
Ω = Circle(0,1) × Circle(0,1)
n = 8
# functional 2D
grid = DiscontinuousSpectralElementGrid(Ω, elements = (2,3), polynomialorder = (n,n), array = ArrayType)
x = view(grid.vgeo, :, grid.x1id, :)   # x-direction	
y = view(grid.vgeo, :, grid.x2id, :)   # y-direction	
z = view(grid.vgeo, :, grid.x3id, :)   # z-direction

M = view(grid.vgeo, :, grid.Mid, :)    # mass matrix

ω1 = reshape(grid.ω[1], (length(grid.ω[1]), 1, 1))
ω2 = reshape(grid.ω[2], (1,length(grid.ω[2]), 1))
jacobian = M ./ reshape((ω1 .* ω2), (length(grid.ω[1]) * length(grid.ω[2]), 1))

##
using Plots
theme(:default)
gr(size=(300,300))
p1 = scatter(x[:], y[:], label = false, color = :black)
for i in 1:length(grid.vmap⁻)
    if div(i-1,n+1)%4 == 0
        color = :red
    elseif div(i-1,n+1)%4 == 1
        color = :orange 
    elseif div(i-1,n+1)%4 == 2
        color = :purple 
    else 
        color = :green
    end
    scatter!([x[grid.vmap⁻[i]]], [y[grid.vmap⁻[i]]], color = color, label = false)
    display(p1)
    sleep(0.05)
end
##
for i in 1:size(x)[2]
    scatter(x[:], y[:], label = false, color = :black)
    p1 = scatter!(
            x[:,i],
            y[:,i], 
            label = false, 
            color = :red,
            title = "Element " * string(i))
    display(p1)
    sleep(0.5)
end

