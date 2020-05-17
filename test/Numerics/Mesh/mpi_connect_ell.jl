using Test
using MPI
using ClimateMachine.Mesh.Topologies

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    @assert csize == 2

    FT = Float64
    Nx = 3
    Ny = 2
    x = range(FT(0); length = Nx + 1, stop = 1)
    y = range(FT(0); length = Ny + 1, stop = 1)

    topology = BrickTopology(
        comm,
        (x, y);
        boundary = ((1, 2), (3, 4)),
        periodicity = (true, true),
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
        2 2 2 2 2 2
        1 1 1 1 1 1
        4 4 4 4 4 4
        3 3 3 3 3 3
    ]

    globalelemtoordr = ones(Int, size(globalelemtoface))
    globalelemtobndy = zeros(Int, size(globalelemtoface))

    if crank == 0
        nrealelem = 3
        globalelems = [1, 2, 3, 4, 5, 6]
        elemtoelem_expect = [
            6 4 2 4 5 6
            5 3 4 4 5 6
            2 1 5 4 5 6
            2 1 5 4 5 6
        ]
        nabrtorank_expect = [1]
        nabrtorecv_expect = UnitRange{Int}[1:3]
        nabrtosend_expect = UnitRange{Int}[1:3]
    elseif crank == 1
        nrealelem = 3
        globalelems = [4, 5, 6, 1, 2, 3]
        elemtoelem_expect = [
            6 4 2 4 5 6
            5 3 4 4 5 6
            3 6 1 4 5 6
            3 6 1 4 5 6
        ]
        nabrtorank_expect = [0]
        nabrtorecv_expect = UnitRange{Int}[1:3]
        nabrtosend_expect = UnitRange{Int}[1:3]
    end

    @test elems == 1:length(globalelems)
    @test realelems == 1:nrealelem
    @test ghostelems == (nrealelem + 1):length(globalelems)
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
