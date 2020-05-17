using Test
using MPI
using ClimateMachine.Mesh.Topologies

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    @assert csize == 3

    topology = StackedBrickTopology(
        comm,
        (2:5, 4:6),
        periodicity = (false, true),
        boundary = ((1, 2), (3, 4)),
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
        1 1 2 2 2 2
        1 1 1 1 2 2
        4 4 4 4 4 4
        3 3 3 3 3 3
    ]

    globalelemtoordr = ones(Int, size(globalelemtoface))

    globalelemtocoord = Array{Int}(undef, 2, 4, 6)
    globalelemtocoord[:, :, 1] = [2 3 2 3; 4 4 5 5]
    globalelemtocoord[:, :, 2] = [2 3 2 3; 5 5 6 6]
    globalelemtocoord[:, :, 3] = [3 4 3 4; 4 4 5 5]
    globalelemtocoord[:, :, 4] = [3 4 3 4; 5 5 6 6]
    globalelemtocoord[:, :, 5] = [4 5 4 5; 4 4 5 5]
    globalelemtocoord[:, :, 6] = [4 5 4 5; 5 5 6 6]


    globalelemtobndy = [
        1 1 0 0 0 0
        0 0 0 0 2 2
        0 0 0 0 0 0
        0 0 0 0 0 0
    ]

    if crank == 0
        nrealelem = 2
        globalelems = [1, 2, 3, 4]
        elemtoelem_expect = [
            1 2 3 4
            3 4 3 4
            2 1 3 4
            2 1 3 4
        ]
        nabrtorank_expect = [1]
        nabrtorecv_expect = UnitRange{Int}[1:2]
        nabrtosend_expect = UnitRange{Int}[1:2]
    elseif crank == 1
        nrealelem = 2
        globalelems = [3, 4, 1, 2, 5, 6]
        elemtoelem_expect = [
            3 4 3 4 5 6
            5 6 3 4 5 6
            2 1 3 4 5 6
            2 1 3 4 5 6
        ]
        nabrtorank_expect = [0, 2]
        nabrtorecv_expect = UnitRange{Int}[1:2, 3:4]
        nabrtosend_expect = UnitRange{Int}[1:2, 3:4]
    elseif crank == 2
        nrealelem = 2
        globalelems = [5, 6, 3, 4]
        elemtoelem_expect = [
            3 4 3 4
            1 2 3 4
            2 1 3 4
            2 1 3 4
        ]
        nabrtorank_expect = [1]
        nabrtorecv_expect = UnitRange{Int}[1:2]
        nabrtosend_expect = UnitRange{Int}[1:2]
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
