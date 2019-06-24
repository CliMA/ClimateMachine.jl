using Test, MPI
using LinearAlgebra

using CLIMA
using CLIMA.MPIStateArrays

MPI.Initialized() || MPI.Init()
Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())

const mpicomm = MPI.COMM_WORLD

mpisize = MPI.Comm_size(mpicomm)
localsize = (4, 6, 8)

A = Array{Float32}(reshape(1:prod(localsize), localsize))
globalA = vcat([A for _ in 1:mpisize])

QA = MPIStateArray{Tuple{localsize[1:2]...}, Float32, Array}(mpicomm, localsize[3])
QA .= A

@test isapprox(norm(QA), norm(globalA))

B = Array{Float32}(reshape(reverse(1:prod(localsize)), localsize))
globalB = vcat([B for _ in 1:mpisize])

QB = similar(QA)
QB .= B

@test isapprox(euclidean_distance(QA, QB), norm(globalA .- globalB))

isinteractive() || MPI.Finalize()
nothing
