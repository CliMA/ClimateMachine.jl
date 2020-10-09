using Test, MPI, Impero
import Impero: compute
include(pwd() * "/tutorials/Land/Heat/wrapper_functions.jl")
include(pwd() * "/src/Diagnostics/diagnostic_fields.jl")

using ClimateMachine
using ClimateMachine.MPIStateArrays

ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD

Q = MPIStateArray{Float64}(mpicomm, ArrayType, 4, 6, 8)

compute(Q::MPIStateArray) = Q.realdata

@wrapper q=Q

# compute(q::Field{S,T}) where {S <: Number, T} = q.data
compute(2*q)


##
