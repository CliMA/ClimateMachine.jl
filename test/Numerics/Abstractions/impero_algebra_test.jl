using Test, MPI, Impero, Plots, GraphRecipes, LinearAlgebra
import Impero: compute

using ClimateMachine
using ClimateMachine.MPIStateArrays

ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD

Q = MPIStateArray{Float64}(mpicomm, ArrayType, 4, 6, 8)

compute(Q::MPIStateArray) = Q.realdata
@wrapper q=Q
compute(2*q)
compute(2*q) - 2 * Q

@testset "Impero Algebra" begin
    @test norm(compute(2*q) - 2 * Q) == 0.0
end