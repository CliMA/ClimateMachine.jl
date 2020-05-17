using Test
using MPI
using ClimateMachine.Mesh.Topologies

function main()
    MPI.Init()

    comm = MPI.COMM_WORLD
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    @assert csize == 2

    topology = StackedBrickTopology(
        comm,
        (1:4, 5:8, 9:12),
        periodicity = (false, true, false),
        boundary = ((1, 2), (3, 4), (5, 6)),
    )

    elems = topology.elems
    realelems = topology.realelems
    ghostelems = topology.ghostelems
    sendelems = topology.sendelems
    elemtocoord = topology.elemtocoord
    elemtoelem = topology.elemtoelem
    elemtoface = topology.elemtoface
    elemtoordr = topology.elemtoordr
    elemtobndy = topology.elemtobndy
    nabrtorank = topology.nabrtorank
    nabrtorecv = topology.nabrtorecv
    nabrtosend = topology.nabrtosend

    globalelemtoface = [
        1 1 1 2 2 2 2 2 2 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2
        4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
        3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
        5 6 6 5 6 6 5 6 6 5 6 6 5 6 6 5 6 6 5 6 6 5 6 6 5 6 6
        5 5 6 5 5 6 5 5 6 5 5 6 5 5 6 5 5 6 5 5 6 5 5 6 5 5 6
    ]

    globalelemtoordr = ones(Int, size(globalelemtoface))

    globalelemtocoord = Array{Int}(undef, 3, 8, 27)

    globalelemtocoord[:, :, 1] =
        [1 2 1 2 1 2 1 2; 5 5 6 6 5 5 6 6; 9 9 9 9 10 10 10 10]
    globalelemtocoord[:, :, 2] =
        [1 2 1 2 1 2 1 2; 5 5 6 6 5 5 6 6; 10 10 10 10 11 11 11 11]
    globalelemtocoord[:, :, 3] =
        [1 2 1 2 1 2 1 2; 5 5 6 6 5 5 6 6; 11 11 11 11 12 12 12 12]
    globalelemtocoord[:, :, 4] =
        [2 3 2 3 2 3 2 3; 5 5 6 6 5 5 6 6; 9 9 9 9 10 10 10 10]
    globalelemtocoord[:, :, 5] =
        [2 3 2 3 2 3 2 3; 5 5 6 6 5 5 6 6; 10 10 10 10 11 11 11 11]
    globalelemtocoord[:, :, 6] =
        [2 3 2 3 2 3 2 3; 5 5 6 6 5 5 6 6; 11 11 11 11 12 12 12 12]
    globalelemtocoord[:, :, 7] =
        [2 3 2 3 2 3 2 3; 6 6 7 7 6 6 7 7; 9 9 9 9 10 10 10 10]
    globalelemtocoord[:, :, 8] =
        [2 3 2 3 2 3 2 3; 6 6 7 7 6 6 7 7; 10 10 10 10 11 11 11 11]
    globalelemtocoord[:, :, 9] =
        [2 3 2 3 2 3 2 3; 6 6 7 7 6 6 7 7; 11 11 11 11 12 12 12 12]
    globalelemtocoord[:, :, 10] =
        [1 2 1 2 1 2 1 2; 6 6 7 7 6 6 7 7; 9 9 9 9 10 10 10 10]
    globalelemtocoord[:, :, 11] =
        [1 2 1 2 1 2 1 2; 6 6 7 7 6 6 7 7; 10 10 10 10 11 11 11 11]
    globalelemtocoord[:, :, 12] =
        [1 2 1 2 1 2 1 2; 6 6 7 7 6 6 7 7; 11 11 11 11 12 12 12 12]
    globalelemtocoord[:, :, 13] =
        [1 2 1 2 1 2 1 2; 7 7 8 8 7 7 8 8; 9 9 9 9 10 10 10 10]
    globalelemtocoord[:, :, 14] =
        [1 2 1 2 1 2 1 2; 7 7 8 8 7 7 8 8; 10 10 10 10 11 11 11 11]
    globalelemtocoord[:, :, 15] =
        [1 2 1 2 1 2 1 2; 7 7 8 8 7 7 8 8; 11 11 11 11 12 12 12 12]
    globalelemtocoord[:, :, 16] =
        [2 3 2 3 2 3 2 3; 7 7 8 8 7 7 8 8; 9 9 9 9 10 10 10 10]
    globalelemtocoord[:, :, 17] =
        [2 3 2 3 2 3 2 3; 7 7 8 8 7 7 8 8; 10 10 10 10 11 11 11 11]
    globalelemtocoord[:, :, 18] =
        [2 3 2 3 2 3 2 3; 7 7 8 8 7 7 8 8; 11 11 11 11 12 12 12 12]
    globalelemtocoord[:, :, 19] =
        [3 4 3 4 3 4 3 4; 7 7 8 8 7 7 8 8; 9 9 9 9 10 10 10 10]
    globalelemtocoord[:, :, 20] =
        [3 4 3 4 3 4 3 4; 7 7 8 8 7 7 8 8; 10 10 10 10 11 11 11 11]
    globalelemtocoord[:, :, 21] =
        [3 4 3 4 3 4 3 4; 7 7 8 8 7 7 8 8; 11 11 11 11 12 12 12 12]
    globalelemtocoord[:, :, 22] =
        [3 4 3 4 3 4 3 4; 6 6 7 7 6 6 7 7; 9 9 9 9 10 10 10 10]
    globalelemtocoord[:, :, 23] =
        [3 4 3 4 3 4 3 4; 6 6 7 7 6 6 7 7; 10 10 10 10 11 11 11 11]
    globalelemtocoord[:, :, 24] =
        [3 4 3 4 3 4 3 4; 6 6 7 7 6 6 7 7; 11 11 11 11 12 12 12 12]
    globalelemtocoord[:, :, 25] =
        [3 4 3 4 3 4 3 4; 5 5 6 6 5 5 6 6; 9 9 9 9 10 10 10 10]
    globalelemtocoord[:, :, 26] =
        [3 4 3 4 3 4 3 4; 5 5 6 6 5 5 6 6; 10 10 10 10 11 11 11 11]
    globalelemtocoord[:, :, 27] =
        [3 4 3 4 3 4 3 4; 5 5 6 6 5 5 6 6; 11 11 11 11 12 12 12 12]


    globalelemtobndy = [
        1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        5 0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0
        0 0 6 0 0 6 0 0 6 0 0 6 0 0 6 0 0 6 0 0 6 0 0 6 0 0 6
    ]

    if crank == 0
        nrealelem = 12
        globalelems = [
            1,
            2,
            3, # 1
            4,
            5,
            6, # 2
            7,
            8,
            9, # 3
            10,
            11,
            12, # 4
            13,
            14,
            15, # 5
            16,
            17,
            18, # 6
            22,
            23,
            24, # 8
            25,
            26,
            27,
        ] # 9

        elemtoelem_expect = [
            1 2 3 1 2 3 10 11 12 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
            4 5 6 22 23 24 19 20 21 7 8 9 13 14 15 16 17 18 19 20 21 22 23 24
            13 14 15 16 17 18 4 5 6 1 2 3 13 14 15 16 17 18 19 20 21 22 23 24
            10 11 12 7 8 9 16 17 18 13 14 15 13 14 15 16 17 18 19 20 21 22 23 24
            1 1 2 4 4 5 7 7 8 10 10 11 13 14 15 16 17 18 19 20 21 22 23 24
            2 3 3 5 6 6 8 9 9 11 12 12 13 14 15 16 17 18 19 20 21 22 23 24
        ]

        nabrtorank_expect = [1]
        nabrtorecv_expect = UnitRange{Int}[1:12]
        nabrtosend_expect = UnitRange{Int}[1:12]
    elseif crank == 1
        nrealelem = 15

        globalelems = [
            13,
            14,
            15, # 5
            16,
            17,
            18, # 6
            19,
            20,
            21, # 7
            22,
            23,
            24, # 8
            25,
            26,
            27, # 9
            1,
            2,
            3, # 1
            4,
            5,
            6, # 2
            7,
            8,
            9, # 3
            10,
            11,
            12,
        ] # 4

        elemtoelem_expect = [
            1 2 3 1 2 3 4 5 6 22 23 24 19 20 21 16 17 18 19 20 21 22 23 24 25 26 27
            4 5 6 7 8 9 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
            25 26 27 22 23 24 10 11 12 13 14 15 7 8 9 16 17 18 19 20 21 22 23 24 25 26 27
            16 17 18 19 20 21 13 14 15 7 8 9 10 11 12 16 17 18 19 20 21 22 23 24 25 26 27
            1 1 2 4 4 5 7 7 8 10 10 11 13 13 14 16 17 18 19 20 21 22 23 24 25 26 27
            2 3 3 5 6 6 8 9 9 11 12 12 14 15 15 16 17 18 19 20 21 22 23 24 25 26 27
        ]

        nabrtorank_expect = [0]
        nabrtorecv_expect = UnitRange{Int}[1:12]
        nabrtosend_expect = UnitRange{Int}[1:12]
    end

    @test elems == 1:length(globalelems)
    @test realelems == 1:nrealelem
    @test ghostelems == (nrealelem + 1):length(globalelems)

    @test elemtocoord == globalelemtocoord[:, :, globalelems]
    @test elemtoface[:, realelems] ==
          globalelemtoface[:, globalelems[realelems]]
    @test elemtoelem == elemtoelem_expect
    @test elemtobndy == globalelemtobndy[:, globalelems]
    @test elemtoordr == ones(eltype(elemtoordr), size(elemtoordr))
    @test nabrtorank == nabrtorank_expect
    @test nabrtorecv == nabrtorecv_expect
    @test nabrtosend == nabrtosend_expect

    @test collect(realelems) ==
          sort(union(topology.exteriorelems, topology.interiorelems))
    @test unique(sort(sendelems)) == topology.exteriorelems
    @test length(intersect(topology.exteriorelems, topology.interiorelems)) == 0
end

main()
