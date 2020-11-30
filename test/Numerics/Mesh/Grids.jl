using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies: BrickTopology
using Test
using MPI

MPI.Initialized() || MPI.Init()

@testset "2-D Mass matrix" begin
    topology = BrickTopology(MPI.COMM_WORLD, ntuple(_ -> (-1, 1), 2);)

    @testset for N in ((2, 2), (2, 3), (3, 2))
        Nq = N .+ 1
        Np = prod(Nq)
        Nfp = div.(Np, Nq)

        grid = DiscontinuousSpectralElementGrid(
            topology,
            FloatType = Float64,
            DeviceArray = Array,
            polynomialorder = N,
        )

        @views begin
            ω = grid.ω
            M = grid.vgeo[:, Grids._M, 1]
            MH = grid.vgeo[:, Grids._MH, 1]
            sM = grid.sgeo[Grids._sM, :, :, 1]

            M = reshape(M, Nq[1], Nq[2])
            @test M ≈ [ω[1][i] * ω[2][j] for i in 1:Nq[1], j in 1:Nq[2]]

            MH = reshape(MH, Nq[1], Nq[2])
            @test MH ≈ [ω[1][i] for i in 1:Nq[1], j in 1:Nq[2]]

            sM12 = sM[1:Nfp[1], 1:2]
            @test sM12[:, 1] ≈ [ω[2][j] for j in 1:Nq[2]]
            @test sM12[:, 2] ≈ [ω[2][j] for j in 1:Nq[2]]

            sM34 = sM[1:Nfp[2], 3:4]
            @test sM34[:, 1] ≈ [ω[1][j] for j in 1:Nq[1]]
            @test sM34[:, 2] ≈ [ω[1][j] for j in 1:Nq[1]]
        end
    end
end

@testset "3-D Mass matrix" begin
    topology = BrickTopology(MPI.COMM_WORLD, ntuple(_ -> (-1, 1), 3);)

    @testset for N in ((2, 2, 2), (2, 3, 4), (4, 3, 2), (2, 4, 3))
        Nq = N .+ 1
        Np = prod(Nq)
        Nfp = div.(Np, Nq)

        grid = DiscontinuousSpectralElementGrid(
            topology,
            FloatType = Float64,
            DeviceArray = Array,
            polynomialorder = N,
        )

        @views begin
            ω = grid.ω
            M = grid.vgeo[:, Grids._M, 1]
            MH = grid.vgeo[:, Grids._MH, 1]
            sM = grid.sgeo[Grids._sM, :, :, 1]

            M = reshape(M, Nq[1], Nq[2], Nq[3])
            @test M ≈ [
                ω[1][i] * ω[2][j] * ω[3][k]
                for i in 1:Nq[1], j in 1:Nq[2], k in 1:Nq[3]
            ]

            MH = reshape(MH, Nq[1], Nq[2], Nq[3])
            @test MH ≈ [
                ω[1][i] * ω[2][j] for i in 1:Nq[1], j in 1:Nq[2], k in 1:Nq[3]
            ]

            sM12 = reshape(sM[1:Nfp[1], 1:2], Nq[2], Nq[3], 2)
            @test sM12[:, :, 1] ≈
                  [ω[2][j] * ω[3][k] for j in 1:Nq[2], k in 1:Nq[3]]
            @test sM12[:, :, 2] ≈
                  [ω[2][j] * ω[3][k] for j in 1:Nq[2], k in 1:Nq[3]]

            sM34 = reshape(sM[1:Nfp[2], 3:4], Nq[1], Nq[3], 2)
            @test sM34[:, :, 1] ≈
                  [ω[1][i] * ω[3][k] for i in 1:Nq[1], k in 1:Nq[3]]
            @test sM34[:, :, 2] ≈
                  [ω[1][i] * ω[3][k] for i in 1:Nq[1], k in 1:Nq[3]]

            sM56 = reshape(sM[1:Nfp[3], 5:6], Nq[1], Nq[2], 2)
            @test sM56[:, :, 1] ≈
                  [ω[1][i] * ω[2][j] for i in 1:Nq[1], j in 1:Nq[2]]
            @test sM56[:, :, 2] ≈
                  [ω[1][i] * ω[2][j] for i in 1:Nq[1], j in 1:Nq[2]]
        end
    end
end
