using Test
using ClimateMachine.Mesh.Topologies
using Combinatorics, MPI

MPI.Initialized() || MPI.Init()

@testset "cubedshellwarp tests" begin
    import ClimateMachine.Mesh.Topologies: cubedshellwarp

    @testset "check radius" begin
        @test hypot(cubedshellwarp(3.0, -2.2, 1.3)...) ≈ 3.0 rtol = eps()
        @test hypot(cubedshellwarp(-3.0, -2.2, 1.3)...) ≈ 3.0 rtol = eps()
        @test hypot(cubedshellwarp(1.1, -2.2, 3.0)...) ≈ 3.0 rtol = eps()
        @test hypot(cubedshellwarp(1.1, -2.2, -3.0)...) ≈ 3.0 rtol = eps()
        @test hypot(cubedshellwarp(1.1, 3.0, 0.0)...) ≈ 3.0 rtol = eps()
        @test hypot(cubedshellwarp(1.1, -3.0, 0.0)...) ≈ 3.0 rtol = eps()
    end

    @testset "check sign" begin
        @test sign.(cubedshellwarp(3.0, -2.2, 1.3)) == sign.((3.0, -2.2, 1.3))
        @test sign.(cubedshellwarp(-3.0, -2.2, 1.3)) == sign.((-3.0, -2.2, 1.3))
        @test sign.(cubedshellwarp(1.1, -2.2, 3.0)) == sign.((1.1, -2.2, 3.0))
        @test sign.(cubedshellwarp(1.1, -2.2, -3.0)) == sign.((1.1, -2.2, -3.0))
        @test sign.(cubedshellwarp(1.1, 3.0, 0.0)) == sign.((1.1, 3.0, 0.0))
        @test sign.(cubedshellwarp(1.1, -3.0, 0.0)) == sign.((1.1, -3.0, 0.0))
    end

    @testset "check continuity" begin
        for (u, v) in zip(
            permutations([3.0, 2.999999999, 1.3]),
            permutations([2.999999999, 3.0, 1.3]),
        )
            @test all(cubedshellwarp(u...) .≈ cubedshellwarp(v...))
        end
        for (u, v) in zip(
            permutations([3.0, -2.999999999, 1.3]),
            permutations([2.999999999, -3.0, 1.3]),
        )
            @test all(cubedshellwarp(u...) .≈ cubedshellwarp(v...))
        end
        for (u, v) in zip(
            permutations([-3.0, 2.999999999, 1.3]),
            permutations([-2.999999999, 3.0, 1.3]),
        )
            @test all(cubedshellwarp(u...) .≈ cubedshellwarp(v...))
        end
        for (u, v) in zip(
            permutations([-3.0, -2.999999999, 1.3]),
            permutations([-2.999999999, -3.0, 1.3]),
        )
            @test all(cubedshellwarp(u...) .≈ cubedshellwarp(v...))
        end
    end
end

@testset "cubedshellunwarp" begin
    import ClimateMachine.Mesh.Topologies: cubedshellwarp, cubedshellunwarp

    for u in permutations([3.0, 2.999999999, 1.3])
        @test all(cubedshellunwarp(cubedshellwarp(u...)...) .≈ u)
    end
    for u in permutations([3.0, -2.999999999, 1.3])
        @test all(cubedshellunwarp(cubedshellwarp(u...)...) .≈ u)
    end
    for u in permutations([-3.0, 2.999999999, 1.3])
        @test all(cubedshellunwarp(cubedshellwarp(u...)...) .≈ u)
    end
    for u in permutations([-3.0, -2.999999999, 1.3])
        @test all(cubedshellunwarp(cubedshellwarp(u...)...) .≈ u)
    end
end

@testset "grid1d" begin
    g = grid1d(0, 10, nelem = 10)
    @test eltype(g) == Float64
    @test length(g) == 11
    @test g[1] == 0
    @test g[end] == 10

    g = grid1d(10f0, 20f0, elemsize = 0.1)
    @test eltype(g) == Float32
    @test length(g) == 101
    @test g[1] == 10
    @test g[end] == 20

    g = grid1d(10f0, 20f0, InteriorStretching(0), elemsize = 0.1)
    @test eltype(g) == Float32
    @test length(g) == 101
    @test g[1] == 10
    @test g[end] == 20

    g = grid1d(10f0, 20f0, SingleExponentialStretching(2.5f0), elemsize = 0.1)
    @test eltype(g) == Float32
    @test length(g) == 101
    @test g[1] == 10
    @test g[end] == 20

    g = grid1d(
        10f0,
        20f0,
        DoubleSidedSingleExponentialStretching(2.5f0),
        elemsize = 0.1,
    )
    @test eltype(g) == Float32
    @test length(g) == 101
    @test g[1] == 10
    @test g[end] == 20
    @info g
    @test g ≈ Float32[
        10.0,
        10.022638,
        10.04585,
        10.069649,
        10.094049,
        10.119069,
        10.144721,
        10.171023,
        10.19799,
        10.225641,
        10.253991,
        10.283059,
        10.312863,
        10.343421,
        10.374753,
        10.406878,
        10.439816,
        10.473589,
        10.508216,
        10.543719,
        10.580122,
        10.617446,
        10.655715,
        10.694953,
        10.735184,
        10.776433,
        10.818727,
        10.862091,
        10.906553,
        10.952141,
        10.998882,
        11.046807,
        11.095946,
        11.146328,
        11.197986,
        11.250951,
        11.305257,
        11.360938,
        11.418029,
        11.476564,
        11.536582,
        11.598119,
        11.661214,
        11.725906,
        11.792235,
        11.860244,
        11.929975,
        12.001471,
        12.074776,
        12.149938,
        17.772999,
        17.850063,
        17.925224,
        17.99853,
        18.070026,
        18.139755,
        18.207764,
        18.274096,
        18.338787,
        18.401882,
        18.463417,
        18.523436,
        18.58197,
        18.639061,
        18.694742,
        18.749048,
        18.802013,
        18.853672,
        18.904055,
        18.953192,
        19.001118,
        19.04786,
        19.093447,
        19.137909,
        19.181273,
        19.223568,
        19.264816,
        19.305046,
        19.344284,
        19.382553,
        19.419878,
        19.45628,
        19.491783,
        19.526411,
        19.560184,
        19.59312,
        19.625248,
        19.656578,
        19.687138,
        19.716942,
        19.74601,
        19.77436,
        19.80201,
        19.828976,
        19.855278,
        19.880932,
        19.90595,
        19.930351,
        19.954151,
        19.977362,
        20.0,
    ]
end

@testset "BrickTopology tests" begin

    let
        comm = MPI.COMM_SELF


        elemrange = (0:10,)
        periodicity = (true,)

        topology = BrickTopology(comm, elemrange, periodicity = periodicity)

        nelem = length(elemrange[1]) - 1

        for e in 1:nelem
            @test topology.elemtocoord[:, :, e] == [e - 1 e]
        end

        @test topology.elemtoelem ==
              [nelem collect(1:(nelem - 1))'; collect(2:nelem)' 1]
        @test topology.elemtoface == repeat(2:-1:1, outer = (1, nelem))

        @test topology.elemtoordr == ones(Int, size(topology.elemtoordr))
        @test topology.elemtobndy == zeros(Int, size(topology.elemtoordr))

        @test topology.elems == 1:nelem
        @test topology.realelems == 1:nelem
        @test topology.ghostelems == nelem .+ (1:0)

        @test length(topology.sendelems) == 0
        @test length(topology.exteriorelems) == 0
        @test collect(topology.realelems) == topology.interiorelems

        @test topology.nabrtorank == Int[]
        @test topology.nabrtorecv == UnitRange{Int}[]
        @test topology.nabrtosend == UnitRange{Int}[]
    end

    let
        comm = MPI.COMM_SELF
        topology = BrickTopology(comm, (0:4, 5:9), periodicity = (false, true))

        nelem = 16

        @test topology.elemtocoord[:, :, 1] == [0 1 0 1; 5 5 6 6]
        @test topology.elemtocoord[:, :, 2] == [1 2 1 2; 5 5 6 6]
        @test topology.elemtocoord[:, :, 3] == [1 2 1 2; 6 6 7 7]
        @test topology.elemtocoord[:, :, 4] == [0 1 0 1; 6 6 7 7]
        @test topology.elemtocoord[:, :, 5] == [0 1 0 1; 7 7 8 8]
        @test topology.elemtocoord[:, :, 6] == [0 1 0 1; 8 8 9 9]
        @test topology.elemtocoord[:, :, 7] == [1 2 1 2; 8 8 9 9]
        @test topology.elemtocoord[:, :, 8] == [1 2 1 2; 7 7 8 8]
        @test topology.elemtocoord[:, :, 9] == [2 3 2 3; 7 7 8 8]
        @test topology.elemtocoord[:, :, 10] == [2 3 2 3; 8 8 9 9]
        @test topology.elemtocoord[:, :, 11] == [3 4 3 4; 8 8 9 9]
        @test topology.elemtocoord[:, :, 12] == [3 4 3 4; 7 7 8 8]
        @test topology.elemtocoord[:, :, 13] == [3 4 3 4; 6 6 7 7]
        @test topology.elemtocoord[:, :, 14] == [2 3 2 3; 6 6 7 7]
        @test topology.elemtocoord[:, :, 15] == [2 3 2 3; 5 5 6 6]
        @test topology.elemtocoord[:, :, 16] == [3 4 3 4; 5 5 6 6]

        @test topology.elemtoelem == [
            1 1 4 4 5 6 6 5 8 7 10 9 14 3 2 15
            2 15 14 3 8 7 10 9 12 11 11 12 13 13 16 16
            6 7 2 1 4 5 8 3 14 9 12 13 16 15 10 11
            4 3 8 5 6 1 2 7 10 15 16 11 12 9 14 13
        ]

        @test topology.elemtoface == [
            1 2 2 1 1 1 2 2 2 2 2 2 2 2 2 2
            1 1 1 1 1 1 1 1 1 1 2 2 2 1 1 2
            4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
            3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
        ]

        @test topology.elemtoordr == ones(Int, size(topology.elemtoordr))

        @test topology.elems == 1:nelem
        @test topology.realelems == 1:nelem
        @test topology.ghostelems == nelem .+ (1:0)

        @test length(topology.sendelems) == 0
        @test length(topology.exteriorelems) == 0
        @test collect(topology.realelems) == topology.interiorelems

        @test topology.nabrtorank == Int[]
        @test topology.nabrtorecv == UnitRange{Int}[]
        @test topology.nabrtosend == UnitRange{Int}[]
    end

    let
        comm = MPI.COMM_SELF
        for px in (true, false)
            topology = BrickTopology(comm, (0:10,), periodicity = (px,))
            @test Topologies.hasboundary(topology) == !px
        end
        for py in (true, false), px in (true, false)
            topology = BrickTopology(comm, (0:10, 0:3), periodicity = (px, py))
            @test Topologies.hasboundary(topology) == !(px && py)
        end
        for pz in (true, false), py in (true, false), px in (true, false)
            topology = BrickTopology(
                comm,
                (0:10, 0:3, -1:3),
                periodicity = (px, py, pz),
            )
            @test Topologies.hasboundary(topology) == !(px && py && pz)
        end
    end
end

@testset "StackedBrickTopology tests" begin
    let
        comm = MPI.COMM_SELF
        topology = StackedBrickTopology(
            comm,
            (2:5, 4:6),
            periodicity = (false, true),
            boundary = ((1, 2), (3, 4)),
        )

        nelem = 6

        @test topology.elemtocoord[:, :, 1] == [2 3 2 3; 4 4 5 5]
        @test topology.elemtocoord[:, :, 2] == [2 3 2 3; 5 5 6 6]
        @test topology.elemtocoord[:, :, 3] == [3 4 3 4; 4 4 5 5]
        @test topology.elemtocoord[:, :, 4] == [3 4 3 4; 5 5 6 6]
        @test topology.elemtocoord[:, :, 5] == [4 5 4 5; 4 4 5 5]
        @test topology.elemtocoord[:, :, 6] == [4 5 4 5; 5 5 6 6]

        @test topology.elemtoelem == [
            1 2 1 2 3 4
            3 4 5 6 5 6
            2 1 4 3 6 5
            2 1 4 3 6 5
        ]

        @test topology.elemtoface == [
            1 1 2 2 2 2
            1 1 1 1 2 2
            4 4 4 4 4 4
            3 3 3 3 3 3
        ]

        @test topology.elemtoordr == ones(Int, size(topology.elemtoordr))

        @test topology.elemtobndy == [
            1 1 0 0 0 0
            0 0 0 0 2 2
            0 0 0 0 0 0
            0 0 0 0 0 0
        ]

        @test topology.elems == 1:nelem
        @test topology.realelems == 1:nelem
        @test topology.ghostelems == nelem .+ (1:0)

        @test length(topology.sendelems) == 0
        @test length(topology.exteriorelems) == 0
        @test collect(topology.realelems) == topology.interiorelems

        @test topology.nabrtorank == Int[]
        @test topology.nabrtorecv == UnitRange{Int}[]
        @test topology.nabrtosend == UnitRange{Int}[]
    end
    let
        comm = MPI.COMM_SELF
        for py in (true, false), px in (true, false)
            topology =
                StackedBrickTopology(comm, (0:10, 0:3), periodicity = (px, py))
            @test Topologies.hasboundary(topology) == !(px && py)
        end
        for pz in (true, false), py in (true, false), px in (true, false)
            topology = StackedBrickTopology(
                comm,
                (0:10, 0:3, -1:3),
                periodicity = (px, py, pz),
            )
            @test Topologies.hasboundary(topology) == !(px && py && pz)
        end
    end
end

@testset "StackedCubedSphereTopology tests" begin
    topology = StackedCubedSphereTopology(MPI.COMM_SELF, 3, 1.0:3.0)
    @test Topologies.hasboundary(topology)
end

@testset "CubedShellTopology tests" begin
    topology = CubedShellTopology(MPI.COMM_SELF, 3, Float64)
    @test !Topologies.hasboundary(topology)
end
