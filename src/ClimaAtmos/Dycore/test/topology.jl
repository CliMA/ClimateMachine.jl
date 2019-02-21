
@testset "BrickTopology tests" begin
  using CLIMAAtmosDycore.Topologies

  let
    comm = MPI.COMM_SELF


    elemrange = (0:10,)
    periodicity=(true,)

    topology = BrickTopology(comm, elemrange, periodicity=periodicity)

    nelem = length(elemrange[1])-1

    for e = 1:nelem
      @test topology.elemtocoord[:,:, e] == [e-1  e]
    end

    @test topology.elemtoelem == [nelem collect(1:(nelem-1))'; collect(2:nelem)' 1]
    @test topology.elemtoface == repeat(2:-1:1, outer=(1,nelem))

    @test topology.elemtoordr == ones(Int, size(topology.elemtoordr))
    @test topology.elemtobndy == zeros(Int, size(topology.elemtoordr))

    @test topology.elems == 1:nelem
    @test topology.realelems == 1:nelem
    @test topology.ghostelems == nelem.+(1:0)

    @test length(topology.sendelems) == 0

    @test topology.nabrtorank == Int[]
    @test topology.nabrtorecv == UnitRange{Int}[]
    @test topology.nabrtosend == UnitRange{Int}[]
  end

  let
    comm = MPI.COMM_SELF
    topology = BrickTopology(comm, (0:4,5:9), periodicity=(false,true))

    nelem = 16

    @test topology.elemtocoord[:,:, 1] == [0 1 0 1; 5 5 6 6]
    @test topology.elemtocoord[:,:, 2] == [1 2 1 2; 5 5 6 6]
    @test topology.elemtocoord[:,:, 3] == [1 2 1 2; 6 6 7 7]
    @test topology.elemtocoord[:,:, 4] == [0 1 0 1; 6 6 7 7]
    @test topology.elemtocoord[:,:, 5] == [0 1 0 1; 7 7 8 8]
    @test topology.elemtocoord[:,:, 6] == [0 1 0 1; 8 8 9 9]
    @test topology.elemtocoord[:,:, 7] == [1 2 1 2; 8 8 9 9]
    @test topology.elemtocoord[:,:, 8] == [1 2 1 2; 7 7 8 8]
    @test topology.elemtocoord[:,:, 9] == [2 3 2 3; 7 7 8 8]
    @test topology.elemtocoord[:,:,10] == [2 3 2 3; 8 8 9 9]
    @test topology.elemtocoord[:,:,11] == [3 4 3 4; 8 8 9 9]
    @test topology.elemtocoord[:,:,12] == [3 4 3 4; 7 7 8 8]
    @test topology.elemtocoord[:,:,13] == [3 4 3 4; 6 6 7 7]
    @test topology.elemtocoord[:,:,14] == [2 3 2 3; 6 6 7 7]
    @test topology.elemtocoord[:,:,15] == [2 3 2 3; 5 5 6 6]
    @test topology.elemtocoord[:,:,16] == [3 4 3 4; 5 5 6 6]

    @test topology.elemtoelem ==
      [1   1   4  4  5  6   6  5   8   7  10   9  14   3   2  15
       2  15  14  3  8  7  10  9  12  11  11  12  13  13  16  16
       6   7   2  1  4  5   8  3  14   9  12  13  16  15  10  11
       4   3   8  5  6  1   2  7  10  15  16  11  12   9  14  13]

    @test topology.elemtoface ==
      [1  2  2  1  1  1  2  2  2  2  2  2  2  2  2  2
       1  1  1  1  1  1  1  1  1  1  2  2  2  1  1  2
       4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4
       3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3]

    @test topology.elemtoordr == ones(Int, size(topology.elemtoordr))

    @test topology.elems == 1:nelem
    @test topology.realelems == 1:nelem
    @test topology.ghostelems == nelem.+(1:0)

    @test length(topology.sendelems) == 0

    @test topology.nabrtorank == Int[]
    @test topology.nabrtorecv == UnitRange{Int}[]
    @test topology.nabrtosend == UnitRange{Int}[]
  end
end
