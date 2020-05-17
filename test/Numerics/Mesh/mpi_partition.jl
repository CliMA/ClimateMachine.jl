using Test
using MPI
using ClimateMachine.Mesh.BrickMesh

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    @assert csize == 3

    (etv, etc, etb, fc) = brickmesh(
        (0:4, 5:9),
        (false, true),
        boundary = ((1, 2), (3, 4), (5, 6)),
        part = crank + 1,
        numparts = csize,
    )
    (etv, etc, etb, fc) = partition(comm, etv, etc, etb, fc)

    if crank == 0
        etv_expect = [
            1 2 7 6 11
            2 3 8 7 12
            6 7 12 11 16
            7 8 13 12 17
        ]
        @test etv == etv_expect
        @test etc[:, :, 1] == [0 1 0 1; 5 5 6 6]
        @test etc[:, :, 2] == [1 2 1 2; 5 5 6 6]
        @test etc[:, :, 3] == [1 2 1 2; 6 6 7 7]
        @test etc[:, :, 4] == [0 1 0 1; 6 6 7 7]
        @test etc[:, :, 5] == [0 1 0 1; 7 7 8 8]
        etb_expect = [
            1 0 0 1 1
            0 0 0 0 0
            0 0 0 0 0
            0 0 0 0 0
        ]
        @test etb == etb_expect
        fc_expect = Array{Int64, 1}[]
        @test fc == fc_expect
    elseif crank == 1
        etv_expect = [
            16 17 12 13 18
            17 18 13 14 19
            21 22 17 18 23
            22 23 18 19 24
        ]
        @test etv == etv_expect
        @test etc[:, :, 1] == [0 1 0 1; 8 8 9 9]
        @test etc[:, :, 2] == [1 2 1 2; 8 8 9 9]
        @test etc[:, :, 3] == [1 2 1 2; 7 7 8 8]
        @test etc[:, :, 4] == [2 3 2 3; 7 7 8 8]
        @test etc[:, :, 5] == [2 3 2 3; 8 8 9 9]
        etb_expect = [
            1 0 0 0 0
            0 0 0 0 0
            0 0 0 0 0
            0 0 0 0 0
        ]
        @test etb == etb_expect
        fc_expect = Array{Int64, 1}[[1, 4, 1, 2], [2, 4, 2, 3], [5, 4, 3, 4]]
        @test fc == fc_expect
    elseif crank == 2
        etv_expect = [
            19 14 9 8 3 4
            20 15 10 9 4 5
            24 19 14 13 8 9
            25 20 15 14 9 10
        ]
        @test etv == etv_expect
        @test etc[:, :, 1] == [3 4 3 4; 8 8 9 9]
        @test etc[:, :, 2] == [3 4 3 4; 7 7 8 8]
        @test etc[:, :, 3] == [3 4 3 4; 6 6 7 7]
        @test etc[:, :, 4] == [2 3 2 3; 6 6 7 7]
        @test etc[:, :, 5] == [2 3 2 3; 5 5 6 6]
        @test etc[:, :, 6] == [3 4 3 4; 5 5 6 6]
        etb_expect = [
            0 0 0 0 0 0
            2 2 2 0 0 2
            0 0 0 0 0 0
            0 0 0 0 0 0
        ]
        @test etb == etb_expect
        fc_expect = Array{Int64, 1}[[1, 4, 4, 5]]
        @test fc == fc_expect
    end
end

main()
