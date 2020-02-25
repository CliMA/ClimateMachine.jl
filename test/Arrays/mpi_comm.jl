using Test
using MPI
using CLIMA
using CLIMA.MPIStateArrays
using CLIMA.Mesh.BrickMesh
using Pkg

CLIMA.init()
const ArrayType = CLIMA.array_type()
const comm = MPI.COMM_WORLD


function main()

  crank = MPI.Comm_rank(comm)
  csize = MPI.Comm_size(comm)


  @assert csize == 3

  if crank == 0
    numreal = 4
    numghost = 3

    nabrtorank = [1, 2]

    sendelems = [1, 2, 3, 4, 1, 4]
    nabrtorecv = [1:2, 3:3]
    nabrtosend = [1:4, 5:6]

    vmaprecv = [37, 38, 39, 40, 42, 43, 44, 45, 46, 49, 52, 53, 54,
                57, 60, 61, 62, 63]
    vmapsend = [3, 6, 9, 10, 11, 12, 19, 22, 25, 34, 35, 36,
                1, 2, 3, 28, 31, 34]

    nabrtovmaprecv = [1:13, 14:18]
    nabrtovmapsend = [1:12, 13:18]

    expectedghostdata = [1001, 1002, 1003, 1004, 1006, 1007, 1008, 1009,
                         1010, 1013, 1016, 1017, 1018,
                         2003, 2006, 2007, 2008, 2009]
  elseif crank == 1
    numreal = 2
    numghost = 4

    nabrtorank = [0]

    sendelems = [1, 2]
    nabrtorecv = [1:4]
    nabrtosend = [1:2]

    vmaprecv = [21, 24, 27, 28, 29, 30, 37, 40, 43, 52, 53, 54]
    vmapsend = [1, 2, 3, 4, 6, 7, 8, 9, 10, 13, 16, 17, 18]

    nabrtovmaprecv = [1:length(vmaprecv)]
    nabrtovmapsend = [1:length(vmapsend)]

    expectedghostdata = [ 3,  6,  9,
                         10, 11, 12,
                         19, 22, 25,
                         34, 35, 36]
  elseif crank == 2
    numreal = 1
    numghost = 2

    nabrtorank = [0]

    sendelems = [1]
    nabrtorecv = [1:2]
    nabrtosend = [1:1]

    vmaprecv = [10, 11, 12, 19, 22, 25]
    vmapsend = [3, 6, 7, 8, 9]

    nabrtovmaprecv = [1:length(vmaprecv)]
    nabrtovmapsend = [1:length(vmapsend)]

    expectedghostdata = [ 1,  2,  3,
                         28, 31, 34]
  end

  numelem = numreal+numghost

  realelems = 1:numreal
  ghostelems = numreal .+ (1:numghost)

  weights = Array{Int64}(undef, (0, 0, 0))

  A = MPIStateArray{Int64}(comm, ArrayType, 9, 2, numelem, realelems, ghostelems,
                           ArrayType(vmaprecv), ArrayType(vmapsend), nabrtorank,
                           nabrtovmaprecv, nabrtovmapsend, ArrayType(weights),
                           555)

  Q = Array(A.data)
  Q .= -1
  shift = 100
  Q[:, 1, realelems] .= reshape((crank * 1000)          .+ (1:9*numreal), 9, numreal)
  Q[:, 2, realelems] .= reshape((crank * 1000) .+ shift .+ (1:9*numreal), 9, numreal)
  copyto!(A.data, Q)

  MPIStateArrays.start_ghost_exchange!(A)
  MPIStateArrays.finish_ghost_exchange!(A)

  Q = Array(A.data)
  @test all(         expectedghostdata .== Q[:, 1, :][:][vmaprecv])
  @test all(shift .+ expectedghostdata .== Q[:, 2, :][:][vmaprecv])

end

main()
