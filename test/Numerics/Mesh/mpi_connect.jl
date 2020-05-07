using Test
using MPI
using ClimateMachine.Mesh.Topologies

function main()
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
        globalelems = [1, 2, 3, 4, 5, 6, 7, 8, 14, 15]
        elemtoelem_expect = [
            1 1 4 4 5 6 7 8 9 10
            2 10 9 3 8 6 7 8 9 10
            6 7 2 1 4 6 7 8 9 10
            4 3 8 5 6 6 7 8 9 10
        ]
        nabrtorank_expect = [1, 2]
        nabrtorecv_expect = UnitRange{Int}[1:3, 4:5]
        nabrtosend_expect = UnitRange{Int}[1:4, 5:6]
    elseif crank == 1
        nrealelem = 5
        globalelems = [6, 7, 8, 9, 10, 1, 2, 3, 5, 11, 12, 14, 15]
        elemtoelem_expect = [
            1 1 9 3 2 6 7 8 9 10 11 12 13
            2 5 4 11 10 6 7 8 9 10 11 12 13
            9 3 8 12 4 6 7 8 9 10 11 12 13
            6 7 2 5 13 6 7 8 9 10 11 12 13
        ]
        nabrtorank_expect = [0, 2]
        nabrtorecv_expect = UnitRange{Int}[1:4, 5:8]
        nabrtosend_expect = UnitRange{Int}[1:3, 4:5]
    elseif crank == 2
        nrealelem = 6
        globalelems = [11, 12, 13, 14, 15, 16, 2, 3, 9, 10]
        elemtoelem_expect = [
            10 9 4 8 7 5 7 8 9 10
            1 2 3 3 6 6 7 8 9 10
            2 3 6 5 10 1 7 8 9 10
            6 1 2 9 4 3 7 8 9 10
        ]
        nabrtorank_expect = [0, 1]
        nabrtorecv_expect = UnitRange{Int}[1:2, 3:4]
        nabrtosend_expect = UnitRange{Int}[1:2, 3:6]
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
