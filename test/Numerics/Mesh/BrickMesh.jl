using ClimateMachine.Mesh.BrickMesh
using Test
using MPI

MPI.Initialized() || MPI.Init()

@testset "Linear Parition" begin
    @test BrickMesh.linearpartition(1, 1, 1) == 1:1
    @test BrickMesh.linearpartition(20, 1, 1) == 1:20
    @test BrickMesh.linearpartition(10, 1, 2) == 1:5
    @test BrickMesh.linearpartition(10, 2, 2) == 6:10
end

@testset "Hilbert Code" begin
    @test BrickMesh.hilbertcode([0, 0], bits = 1) == [0, 0]
    @test BrickMesh.hilbertcode([0, 1], bits = 1) == [0, 1]
    @test BrickMesh.hilbertcode([1, 1], bits = 1) == [1, 0]
    @test BrickMesh.hilbertcode([1, 0], bits = 1) == [1, 1]
    @test BrickMesh.hilbertcode([0, 0], bits = 2) == [0, 0]
    @test BrickMesh.hilbertcode([1, 0], bits = 2) == [0, 1]
    @test BrickMesh.hilbertcode([1, 1], bits = 2) == [0, 2]
    @test BrickMesh.hilbertcode([0, 1], bits = 2) == [0, 3]
    @test BrickMesh.hilbertcode([0, 2], bits = 2) == [1, 0]
    @test BrickMesh.hilbertcode([0, 3], bits = 2) == [1, 1]
    @test BrickMesh.hilbertcode([1, 3], bits = 2) == [1, 2]
    @test BrickMesh.hilbertcode([1, 2], bits = 2) == [1, 3]
    @test BrickMesh.hilbertcode([2, 2], bits = 2) == [2, 0]
    @test BrickMesh.hilbertcode([2, 3], bits = 2) == [2, 1]
    @test BrickMesh.hilbertcode([3, 3], bits = 2) == [2, 2]
    @test BrickMesh.hilbertcode([3, 2], bits = 2) == [2, 3]
    @test BrickMesh.hilbertcode([3, 1], bits = 2) == [3, 0]
    @test BrickMesh.hilbertcode([2, 1], bits = 2) == [3, 1]
    @test BrickMesh.hilbertcode([2, 0], bits = 2) == [3, 2]
    @test BrickMesh.hilbertcode([3, 0], bits = 2) == [3, 3]

    @test BrickMesh.hilbertcode(UInt64.([14, 3, 4])) ==
          UInt64.([0x0, 0x0, 0xe25])
end

@testset "Mesh to Hilbert Code" begin
    let
        etc = Array{Float64}(undef, 2, 4, 6)
        etc[:, :, 1] = [2.0 3.0 2.0 3.0; 4.0 4.0 5.0 5.0]
        etc[:, :, 2] = [3.0 4.0 3.0 4.0; 4.0 4.0 5.0 5.0]
        etc[:, :, 3] = [4.0 5.0 4.0 5.0; 4.0 4.0 5.0 5.0]
        etc[:, :, 4] = [2.0 3.0 2.0 3.0; 5.0 5.0 6.0 6.0]
        etc[:, :, 5] = [3.0 4.0 3.0 4.0; 5.0 5.0 6.0 6.0]
        etc[:, :, 6] = [4.0 5.0 4.0 5.0; 5.0 5.0 6.0 6.0]

        code_exect = UInt64[
            0x0000000000000000 0x1555555555555555 0xffffffffffffffff 0x5555555555555555 0x6aaaaaaaaaaaaaaa 0xaaaaaaaaaaaaaaaa
            0x0000000000000000 0x5555555555555555 0xffffffffffffffff 0x5555555555555555 0xaaaaaaaaaaaaaaaa 0xaaaaaaaaaaaaaaaa
        ]

        code = centroidtocode(MPI.COMM_SELF, etc)

        @test code == code_exect
    end

    let
        nelem = 1
        d = 2

        etc = Array{Float64}(undef, d, d^2, nelem)
        etc[:, :, 1] = [2.0 3.0 2.0 3.0; 4.0 4.0 5.0 5.0]
        code = centroidtocode(MPI.COMM_SELF, etc)

        @test code == zeros(eltype(code), d, nelem)
    end
end

@testset "Vertex Ordering" begin
    @test ((1,), 1) == BrickMesh.vertsortandorder(1)

    @test ((1, 2), 1) == BrickMesh.vertsortandorder(1, 2)
    @test ((1, 2), 2) == BrickMesh.vertsortandorder(2, 1)

    @test ((1, 2, 3), 1) == BrickMesh.vertsortandorder(1, 2, 3)
    @test ((1, 2, 3), 2) == BrickMesh.vertsortandorder(3, 1, 2)
    @test ((1, 2, 3), 3) == BrickMesh.vertsortandorder(2, 3, 1)
    @test ((1, 2, 3), 4) == BrickMesh.vertsortandorder(2, 1, 3)
    @test ((1, 2, 3), 5) == BrickMesh.vertsortandorder(3, 2, 1)
    @test ((1, 2, 3), 6) == BrickMesh.vertsortandorder(1, 3, 2)

    @test_throws ErrorException BrickMesh.vertsortandorder(2, 1, 1)

    @test ((1, 2, 3, 4), 1) == BrickMesh.vertsortandorder(1, 2, 3, 4)
    @test ((1, 2, 3, 4), 2) == BrickMesh.vertsortandorder(1, 3, 2, 4)
    @test ((1, 2, 3, 4), 3) == BrickMesh.vertsortandorder(2, 1, 3, 4)
    @test ((1, 2, 3, 4), 4) == BrickMesh.vertsortandorder(2, 4, 1, 3)
    @test ((1, 2, 3, 4), 5) == BrickMesh.vertsortandorder(3, 1, 4, 2)
    @test ((1, 2, 3, 4), 6) == BrickMesh.vertsortandorder(3, 4, 1, 2)
    @test ((1, 2, 3, 4), 7) == BrickMesh.vertsortandorder(4, 2, 3, 1)
    @test ((1, 2, 3, 4), 8) == BrickMesh.vertsortandorder(4, 3, 2, 1)

    @test_throws ErrorException BrickMesh.vertsortandorder(1, 3, 3, 1)
end

@testset "Mesh" begin
    let
        (etv, etc, etb, fc) = brickmesh((4:7,), (false,))
        etv_expect = [
            1 2 3
            2 3 4
        ]
        etb_expect = [
            1 0 0
            0 0 1
        ]
        fc_expect = Array{Int64, 1}[]

        @test etv == etv_expect
        @test etb == etb_expect
        @test fc == fc_expect
        @test etc[:, :, 1] == [4 5]
        @test etc[:, :, 2] == [5 6]
        @test etc[:, :, 3] == [6 7]
    end

    let
        (etv, etc, etb, fc) = brickmesh((4:7,), (true,))
        etv_expect = [
            1 2 3
            2 3 4
        ]
        etb_expect = [
            0 0 0
            0 0 0
        ]
        fc_expect = Array{Int64, 1}[[3, 2, 1]]

        @test etv == etv_expect
        @test etb == etb_expect
        @test fc == fc_expect
        @test etc[:, :, 1] == [4 5]
        @test etc[:, :, 2] == [5 6]
        @test etc[:, :, 3] == [6 7]
    end

    let
        (etv, etc, etb, fc) = brickmesh((2:5, 4:6), (false, true))

        etv_expect = [
            1 2 5 6
            2 3 6 7
            3 4 7 8
            5 6 9 10
            6 7 10 11
            7 8 11 12
        ]'
        etb_expect = [
            1 0 0 1 0 0
            0 0 1 0 0 1
            0 0 0 0 0 0
            0 0 0 0 0 0
        ]
        fc_expect = Array{Int64, 1}[[4, 4, 1, 2], [5, 4, 2, 3], [6, 4, 3, 4]]

        @test etv == etv_expect
        @test etb == etb_expect
        @test fc == fc_expect
        @test etc[:, :, 1] == [
            2 3 2 3
            4 4 5 5
        ]
        @test etc[:, :, 5] == [
            3 4 3 4
            5 5 6 6
        ]
    end

    let
        (etv, etc, etb, fc) =
            brickmesh((-1:2:1, -1:2:1, -1:1:1), (true, true, true))
        etv_expect = [
            1 5
            2 6
            3 7
            4 8
            5 9
            6 10
            7 11
            8 12
        ]
        etb_expect = zeros(Int64, 6, 2)

        fc_expect = Array{Int64, 1}[
            [1, 2, 1, 3, 5, 7],
            [1, 4, 1, 2, 5, 6],
            [2, 2, 5, 7, 9, 11],
            [2, 4, 5, 6, 9, 10],
            [2, 6, 1, 2, 3, 4],
        ]

        @test etv == etv_expect
        @test etb == etb_expect
        @test fc == fc_expect

        @test etc[:, :, 1] == [
            -1 1 -1 1 -1 1 -1 1
            -1 -1 1 1 -1 -1 1 1
            -1 -1 -1 -1 0 0 0 0
        ]

        @test etc[:, :, 2] == [
            -1 1 -1 1 -1 1 -1 1
            -1 -1 1 1 -1 -1 1 1
            0 0 0 0 1 1 1 1
        ]
    end

    let
        (etv, etc, etb, fc) = brickmesh(
            (-1:1, -1:1, -1:1),
            (false, false, false),
            boundary = ((11, 12), (13, 14), (15, 16)),
        )

        @test etb == [
            11 0 11 0 11 0 11 0
            0 12 0 12 0 12 0 12
            13 13 0 0 13 13 0 0
            0 0 14 14 0 0 14 14
            15 15 15 15 0 0 0 0
            0 0 0 0 16 16 16 16
        ]
    end

    let
        x = (1:1000,)
        p = (false,)
        b = ((1, 2),)

        (etv, etc, etb, fc) = brickmesh(x, p, boundary = b)

        n = 50
        (etv_parts, etc_parts, etb_parts, fc_parts) =
            brickmesh(x, p, boundary = b, part = 1, numparts = n)
        for j in 2:n
            (etv_j, etc_j, etb_j, fc_j) =
                brickmesh(x, p, boundary = b, part = j, numparts = n)
            etv_parts = cat(etv_parts, etv_j; dims = 2)
            etc_parts = cat(etc_parts, etc_j; dims = 3)
            etb_parts = cat(etb_parts, etb_j; dims = 2)
        end

        @test etv == etv_parts
        @test etc == etc_parts
        @test etb == etb_parts
    end


    let
        x = (-1:2:10, -1:1:1, -4:1:1)
        p = (true, false, true)
        b = ((1, 2), (3, 4), (5, 6))

        (etv, etc, etb, fc) = brickmesh(x, p, boundary = b)

        n = 50
        (etv_parts, etc_parts, etb_parts, fc_parts) =
            brickmesh(x, p, boundary = b, part = 1, numparts = n)
        for j in 2:n
            (etv_j, etc_j, etb_j, fc_j) =
                brickmesh(x, p, boundary = b, part = j, numparts = n)
            etv_parts = cat(etv_parts, etv_j; dims = 2)
            etc_parts = cat(etc_parts, etc_j; dims = 3)
            etb_parts = cat(etb_parts, etb_j; dims = 2)
        end

        @test etv == etv_parts
        @test etc == etc_parts
        @test etb == etb_parts
    end
end

@testset "Connect" begin
    let
        comm = MPI.COMM_SELF

        mesh = connectmesh(
            comm,
            partition(comm, brickmesh((0:10,), (true,))...)[1:4]...,
        )

        nelem = 10

        @test mesh[:elemtocoord][:, :, 1] == [0 1]
        @test mesh[:elemtocoord][:, :, 2] == [1 2]
        @test mesh[:elemtocoord][:, :, 3] == [2 3]
        @test mesh[:elemtocoord][:, :, 4] == [3 4]
        @test mesh[:elemtocoord][:, :, 5] == [4 5]
        @test mesh[:elemtocoord][:, :, 6] == [5 6]
        @test mesh[:elemtocoord][:, :, 7] == [6 7]
        @test mesh[:elemtocoord][:, :, 8] == [7 8]
        @test mesh[:elemtocoord][:, :, 9] == [8 9]
        @test mesh[:elemtocoord][:, :, 10] == [9 10]

        @test mesh[:elemtoelem] == [
            10 1 2 3 4 5 6 7 8 9
            2 3 4 5 6 7 8 9 10 1
        ]

        @test mesh[:elemtoface] == [
            2 2 2 2 2 2 2 2 2 2
            1 1 1 1 1 1 1 1 1 1
        ]

        @test mesh[:elemtoordr] == ones(Int, size(mesh[:elemtoordr]))
        @test mesh[:elemtobndy] == zeros(Int, size(mesh[:elemtoordr]))

        @test mesh[:elems] == 1:nelem
        @test mesh[:realelems] == 1:nelem
        @test mesh[:ghostelems] == nelem .+ (1:0)

        @test length(mesh[:sendelems]) == 0

        @test mesh[:nabrtorank] == Int[]
        @test mesh[:nabrtorecv] == UnitRange{Int}[]
        @test mesh[:nabrtosend] == UnitRange{Int}[]
    end

    let
        comm = MPI.COMM_SELF
        mesh = connectmesh(
            comm,
            partition(comm, brickmesh((0:4, 5:9), (false, true))...)[1:4]...,
        )

        nelem = 16

        @test mesh[:elemtocoord][:, :, 1] == [0 1 0 1; 5 5 6 6]
        @test mesh[:elemtocoord][:, :, 2] == [1 2 1 2; 5 5 6 6]
        @test mesh[:elemtocoord][:, :, 3] == [1 2 1 2; 6 6 7 7]
        @test mesh[:elemtocoord][:, :, 4] == [0 1 0 1; 6 6 7 7]
        @test mesh[:elemtocoord][:, :, 5] == [0 1 0 1; 7 7 8 8]
        @test mesh[:elemtocoord][:, :, 6] == [0 1 0 1; 8 8 9 9]
        @test mesh[:elemtocoord][:, :, 7] == [1 2 1 2; 8 8 9 9]
        @test mesh[:elemtocoord][:, :, 8] == [1 2 1 2; 7 7 8 8]
        @test mesh[:elemtocoord][:, :, 9] == [2 3 2 3; 7 7 8 8]
        @test mesh[:elemtocoord][:, :, 10] == [2 3 2 3; 8 8 9 9]
        @test mesh[:elemtocoord][:, :, 11] == [3 4 3 4; 8 8 9 9]
        @test mesh[:elemtocoord][:, :, 12] == [3 4 3 4; 7 7 8 8]
        @test mesh[:elemtocoord][:, :, 13] == [3 4 3 4; 6 6 7 7]
        @test mesh[:elemtocoord][:, :, 14] == [2 3 2 3; 6 6 7 7]
        @test mesh[:elemtocoord][:, :, 15] == [2 3 2 3; 5 5 6 6]
        @test mesh[:elemtocoord][:, :, 16] == [3 4 3 4; 5 5 6 6]

        @test mesh[:elemtoelem] == [
            1 1 4 4 5 6 6 5 8 7 10 9 14 3 2 15
            2 15 14 3 8 7 10 9 12 11 11 12 13 13 16 16
            6 7 2 1 4 5 8 3 14 9 12 13 16 15 10 11
            4 3 8 5 6 1 2 7 10 15 16 11 12 9 14 13
        ]

        @test mesh[:elemtoface] == [
            1 2 2 1 1 1 2 2 2 2 2 2 2 2 2 2
            1 1 1 1 1 1 1 1 1 1 2 2 2 1 1 2
            4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
            3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
        ]

        @test mesh[:elemtoordr] == ones(Int, size(mesh[:elemtoordr]))

        @test mesh[:elems] == 1:nelem
        @test mesh[:realelems] == 1:nelem
        @test mesh[:ghostelems] == nelem .+ (1:0)

        @test length(mesh[:sendelems]) == 0

        @test mesh[:nabrtorank] == Int[]
        @test mesh[:nabrtorecv] == UnitRange{Int}[]
        @test mesh[:nabrtosend] == UnitRange{Int}[]
    end
end

@testset "Mappings" begin
    let
        comm = MPI.COMM_SELF
        x = (0:4,)
        mesh =
            connectmesh(comm, partition(comm, brickmesh(x, (true,))...)[1:4]...)

        N = 3
        d = length(x)
        nelem = prod(length.(x) .- 1)
        nface = 2d
        Nfp = (N + 1)^(d - 1)

        vmap⁻, vmap⁺ =
            mappings(N, mesh[:elemtoelem], mesh[:elemtoface], mesh[:elemtoordr])

        @test vmap⁻ == reshape([1, 4, 5, 8, 9, 12, 13, 16], Nfp, nface, nelem)
        @test vmap⁺ == reshape([16, 5, 4, 9, 8, 13, 12, 1], Nfp, nface, nelem)
    end

    let
        comm = MPI.COMM_SELF
        x = (-1:1, 0:1)
        p = (false, true)
        mesh = connectmesh(comm, partition(comm, brickmesh(x, p)...)[1:4]...)

        N = 2
        d = length(x)
        nelem = prod(length.(x) .- 1)
        nface = 2d
        Nfp = (N + 1)^(d - 1)

        vmap⁻, vmap⁺ =
            mappings(N, mesh[:elemtoelem], mesh[:elemtoface], mesh[:elemtoordr])

        @test vmap⁻ == reshape(
            [
                1,
                4,
                7,  # f=1 e=1
                3,
                6,
                9,  # f=2 e=1
                1,
                2,
                3,  # f=3 e=1
                7,
                8,
                9,  # f=4 e=1
                10,
                13,
                16,  # f=1 e=2
                12,
                15,
                18,  # f=2 e=2
                10,
                11,
                12,  # f=3 e=2
                16,
                17,
                18,
            ], # f=4 e=2
            Nfp,
            nface,
            nelem,
        )

        @test vmap⁺ == reshape(
            [
                1,
                4,
                7,  # f=1 e=1
                10,
                13,
                16,  # f=1 e=2
                7,
                8,
                9,  # f=4 e=1
                1,
                2,
                3,  # f=3 e=1
                3,
                6,
                9,  # f=2 e=1
                12,
                15,
                18,  # f=2 e=2
                16,
                17,
                18,  # f=4 e=2
                10,
                11,
                12,
            ], # f=3 e=2
            Nfp,
            nface,
            nelem,
        )
    end

    let
        comm = MPI.COMM_SELF
        x = (0:1, 0:1, -1:1)
        p = (false, true, false)
        mesh = connectmesh(comm, partition(comm, brickmesh(x, p)...)[1:4]...)

        N = 2
        d = length(x)
        nelem = prod(length.(x) .- 1)
        nface = 2d
        Np = (N + 1)^d
        Nfp = (N + 1)^(d - 1)

        vmap⁻, vmap⁺ =
            mappings(N, mesh[:elemtoelem], mesh[:elemtoface], mesh[:elemtoordr])

        fmask = [
            1 3 1 7 1 19
            4 6 2 8 2 20
            7 9 3 9 3 21
            10 12 10 16 4 22
            13 15 11 17 5 23
            16 18 12 18 6 24
            19 21 19 25 7 25
            22 24 20 26 8 26
            25 27 21 27 9 27
        ]


        @test vmap⁻ == reshape([fmask[:]; fmask[:] .+ Np], Nfp, nface, nelem)

        @test vmap⁺ == reshape(
            [
                fmask[:, 1]
                fmask[:, 2]
                fmask[:, 4]
                fmask[:, 3]
                fmask[:, 5]
                fmask[:, 5] .+ Np
                fmask[:, 1] .+ Np
                fmask[:, 2] .+ Np
                fmask[:, 4] .+ Np
                fmask[:, 3] .+ Np
                fmask[:, 6]
                fmask[:, 6] .+ Np
            ],
            Nfp,
            nface,
            nelem,
        )
    end
end

@testset "Get Partition" begin
    let
        Nelem = 150
        (so, ss, rs) = BrickMesh.getpartition(MPI.COMM_SELF, Nelem:-1:1)
        @test so == Nelem:-1:1
        @test ss == [1, Nelem + 1]
        @test rs == [1, Nelem + 1]
    end

    let
        Nelem = 111
        code = [ones(1, Nelem); collect(Nelem:-1:1)']
        (so, ss, rs) = BrickMesh.getpartition(MPI.COMM_SELF, Nelem:-1:1)
        @test so == Nelem:-1:1
        @test ss == [1, Nelem + 1]
        @test rs == [1, Nelem + 1]
    end
end

@testset "Partition" begin
    (etv, etc, etb, fc) =
        brickmesh((-1:2:1, -1:2:1, -2:1:2), (true, true, true))
    (netv, netc, netb, nfc) = partition(MPI.COMM_SELF, etv, etc, etb, fc)[1:4]
    @test etv == netv
    @test etc == netc
    @test etb == netb
    @test fc == nfc
end

@testset "Comm Mappings" begin
    let
        N = 1
        d = 2
        nface = 2d

        commelems = [1, 2, 5]
        commfaces = BitArray(undef, nface, length(commelems))
        commfaces .= false
        nabrtocomm = [1:2, 3:3]

        vmapC, nabrtovmapC = commmapping(N, commelems, commfaces, nabrtocomm)

        @test vmapC == Int[]
        @test nabrtovmapC == UnitRange{Int64}[1:0, 1:0]
    end

    let
        N = 1
        d = 2
        nface = 2d

        commelems = [1, 2, 5]
        commfaces = BitArray([
            false false false
            false true false
            false true false
            false false true
        ])
        nabrtocomm = [1:2, 3:3]

        vmapC, nabrtovmapC = commmapping(N, commelems, commfaces, nabrtocomm)

        @test vmapC == [5, 6, 8, 19, 20]
        @test nabrtovmapC == UnitRange{Int64}[1:3, 4:5]
    end

    let
        N = 2
        d = 2
        nface = 2d

        commelems = [2, 4, 5]
        commfaces = BitArray([
            true true false
            false false false
            false true true
            false false true
        ])
        nabrtocomm = [1:1, 2:3]

        vmapC, nabrtovmapC = commmapping(N, commelems, commfaces, nabrtocomm)

        @test vmapC == [10, 13, 16, 28, 29, 30, 31, 34, 37, 38, 39, 43, 44, 45]
        @test nabrtovmapC == UnitRange{Int64}[1:3, 4:14]
    end

    let
        N = 2
        d = 3
        nface = 2d

        commelems = [3, 4, 7, 9]
        commfaces = BitArray([
            true true true false
            false true false false
            false true false false
            false true false false
            false true true true
            false true false true
        ])
        nabrtocomm = [1:1, 2:4]

        vmapC, nabrtovmapC = commmapping(N, commelems, commfaces, nabrtocomm)

        @test vmapC == [
            55,
            58,
            61,
            64,
            67,
            70,
            73,
            76,
            79,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            175,
            178,
            181,
            184,
            187,
            217,
            218,
            219,
            220,
            221,
            222,
            223,
            224,
            225,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
        ]
        @test nabrtovmapC == UnitRange{Int64}[1:9, 10:68]
    end

end
