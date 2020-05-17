using Test
using MPI
using ClimateMachine.Mesh.Topologies

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    @assert csize == 5

    topology = BrickTopology(
        comm,
        (0:10,);
        boundary = ((1, 2),),
        periodicity = (true,),
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

    globalelemtoelem = [
        10 1 2 3 4 5 6 7 8 9
        2 3 4 5 6 7 8 9 10 1
    ]

    globalelemtoface = [
        2 2 2 2 2 2 2 2 2 2
        1 1 1 1 1 1 1 1 1 1
    ]

    globalelemtoordr = ones(Int, size(globalelemtoface))
    globalelemtobndy = zeros(Int, size(globalelemtoface))

    globalelemtocoord = Array{Int}(undef, 1, 2, 10)
    globalelemtocoord[:, :, 1] = [0 1]
    globalelemtocoord[:, :, 2] = [1 2]
    globalelemtocoord[:, :, 3] = [2 3]
    globalelemtocoord[:, :, 4] = [3 4]
    globalelemtocoord[:, :, 5] = [4 5]
    globalelemtocoord[:, :, 6] = [5 6]
    globalelemtocoord[:, :, 7] = [6 7]
    globalelemtocoord[:, :, 8] = [7 8]
    globalelemtocoord[:, :, 9] = [8 9]
    globalelemtocoord[:, :, 10] = [9 10]

    @assert csize == 5
    nrealelem = 2
    if crank == 0
        globalelems = [1, 2, 3, 10]
        elemtoelem_expect = [4 1 3 4; 2 3 3 4]
        nabrtorank_expect = [1, 4]
    elseif crank == 1
        globalelems = [3, 4, 2, 5]
        elemtoelem_expect = [3 1 3 4; 2 4 3 4]
        nabrtorank_expect = [0, 2]
    elseif crank == 2
        globalelems = [5, 6, 4, 7]
        elemtoelem_expect = [3 1 3 4; 2 4 3 4]
        nabrtorank_expect = [1, 3]
    elseif crank == 3
        globalelems = [7, 8, 6, 9]
        elemtoelem_expect = [3 1 3 4; 2 4 3 4]
        nabrtorank_expect = [2, 4]
    elseif crank == 4
        globalelems = [9, 10, 1, 8]
        elemtoelem_expect = [4 1 3 4; 2 3 3 4]
        nabrtorank_expect = [0, 3]
    end
    nabrtorecv_expect = UnitRange{Int}[1:1, 2:2]
    nabrtosend_expect = UnitRange{Int}[1:1, 2:2]

    @test elems == 1:length(globalelems)
    @test realelems == 1:nrealelem
    @test ghostelems == (nrealelem + 1):length(globalelems)
    @test elemtocoord == globalelemtocoord[:, :, globalelems]
    @test elemtoface[:, realelems] ==
          globalelemtoface[:, globalelems[realelems]]
    @test elemtoelem == elemtoelem_expect
    @test elemtobndy == globalelemtobndy[:, globalelems]
    @test elemtoordr == globalelemtoordr[:, globalelems]
    @test nabrtorank == nabrtorank_expect
    @test nabrtorecv == nabrtorecv_expect
    @test nabrtosend == nabrtosend_expect

    @test collect(realelems) ==
          sort(union(topology.exteriorelems, topology.interiorelems))
    @test unique(sort(sendelems)) == topology.exteriorelems
    @test length(intersect(topology.exteriorelems, topology.interiorelems)) == 0
end

main()
