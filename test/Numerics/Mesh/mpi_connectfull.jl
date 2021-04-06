using Test
using MPI
using ClimateMachine.Mesh.Topologies

function test_connectmeshfull()
    MPI.Init()
    comm = MPI.COMM_WORLD
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    @assert csize == 3

    topology = BrickTopology(
        comm,
        (0:4, 5:9);
        boundary = ((1, 2), (3, 4)),
        periodicity = (false, true),
        connectivity = :full,
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
        1 2 2 1 1 1 2 2 2 2 2 2 2 2 2 2
        1 1 1 1 1 1 1 1 1 1 2 2 2 1 1 2
        4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
        3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
    ]

    globalelemtoordr = ones(Int, size(globalelemtoface))

    globalelemtocoord = Array{Int}(undef, 2, 4, 16)
    globalelemtocoord[:, :, 1] = [0 1 0 1; 5 5 6 6]
    globalelemtocoord[:, :, 2] = [1 2 1 2; 5 5 6 6]
    globalelemtocoord[:, :, 3] = [1 2 1 2; 6 6 7 7]
    globalelemtocoord[:, :, 4] = [0 1 0 1; 6 6 7 7]
    globalelemtocoord[:, :, 5] = [0 1 0 1; 7 7 8 8]
    globalelemtocoord[:, :, 6] = [0 1 0 1; 8 8 9 9]
    globalelemtocoord[:, :, 7] = [1 2 1 2; 8 8 9 9]
    globalelemtocoord[:, :, 8] = [1 2 1 2; 7 7 8 8]
    globalelemtocoord[:, :, 9] = [2 3 2 3; 7 7 8 8]
    globalelemtocoord[:, :, 10] = [2 3 2 3; 8 8 9 9]
    globalelemtocoord[:, :, 11] = [3 4 3 4; 8 8 9 9]
    globalelemtocoord[:, :, 12] = [3 4 3 4; 7 7 8 8]
    globalelemtocoord[:, :, 13] = [3 4 3 4; 6 6 7 7]
    globalelemtocoord[:, :, 14] = [2 3 2 3; 6 6 7 7]
    globalelemtocoord[:, :, 15] = [2 3 2 3; 5 5 6 6]
    globalelemtocoord[:, :, 16] = [3 4 3 4; 5 5 6 6]

    globalelemtobndy = [
        1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 2 2 2 0 0 2
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    ]

    if crank == 0
        nrealelem = 5
        globalelems = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15]
        elemtoelem_expect = [
            1 1 4 2 3 4 6 5 8 7 3 2
            2 12 11 3 8 7 10 9 9 10 11 12
            6 7 2 1 4 5 8 3 11 9 12 10
            4 3 8 5 6 1 2 7 10 12 9 11
        ]

        elemtoface_expect = [
            1 2 2 1 1 1 2 2 2 2 2 2
            1 1 1 1 1 1 1 1 2 2 2 2
            4 4 4 4 4 4 4 4 4 4 4 4
            3 3 3 3 3 3 3 3 3 3 3 3
        ]

        nabrtorank_expect = [1, 2]
        nabrtorecv_expect = UnitRange{Int}[1:5, 6:7]
        nabrtosend_expect = UnitRange{Int}[1:5, 6:7]

    elseif crank == 1
        nrealelem = 5
        globalelems = [6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16]
        elemtoelem_expect = [
            1 1 10 3 2 2 6 9 3 4 5 4 14 8 7 15
            2 5 4 12 11 7 15 14 8 3 1 2 3 13 16 4
            10 3 8 14 4 1 2 7 6 9 12 13 16 15 5 11
            6 7 2 5 15 9 8 3 10 1 16 11 12 4 14 13
        ]
        elemtoface_expect = [
            1 2 2 2 2 1 2 2 1 1 2 2 2 2 2 2
            1 1 1 1 1 1 1 1 1 1 2 2 2 1 1 2
            4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
            3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
        ]

        nabrtorank_expect = [0, 2]
        nabrtorecv_expect = UnitRange{Int}[1:5, 6:11]
        nabrtosend_expect = UnitRange{Int}[1:5, 6:9]

    elseif crank == 2
        nrealelem = 6
        globalelems = [11, 12, 13, 14, 15, 16, 2, 3, 7, 8, 9, 10]
        elemtoelem_expect = [
            12 11 4 8 7 5 7 8 9 10 10 9
            1 2 3 3 6 4 5 4 12 11 2 1
            2 3 6 5 12 1 9 7 10 8 4 11
            6 1 2 11 4 3 8 10 7 9 12 5
        ]
        elemtoface_expect = [
            2 2 2 2 2 2 1 1 1 1 2 2
            2 2 2 1 1 2 1 1 1 1 1 1
            4 4 4 4 4 4 4 4 4 4 4 4
            3 3 3 3 3 3 3 3 3 3 3 3
        ]

        nabrtorank_expect = [0, 1]
        nabrtorecv_expect = UnitRange{Int}[1:2, 3:6]
        nabrtosend_expect = UnitRange{Int}[1:2, 3:8]
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

test_connectmeshfull()
