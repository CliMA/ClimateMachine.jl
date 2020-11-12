using ClimateMachine.Mesh.Elements
using ClimateMachine.Mesh.Metrics
using Test
using Random: MersenneTwister

const VGEO2D = (x1 = 1, x2 = 2, J = 3, ξ1x1 = 4, ξ2x1 = 5, ξ1x2 = 6, ξ2x2 = 7)
const SGEO2D = (sJ = 1, n1 = 2, n2 = 3)

const VGEO3D = (
    x1 = 1,
    x2 = 2,
    x3 = 3,
    J = 4,
    ξ1x1 = 5,
    ξ2x1 = 6,
    ξ3x1 = 7,
    ξ1x2 = 8,
    ξ2x2 = 9,
    ξ3x2 = 10,
    ξ1x3 = 11,
    ξ2x3 = 12,
    ξ3x3 = 13,
)
const SGEO3D = (sJ = 1, n1 = 2, n2 = 3, n3 = 4)

@testset "1-D Metric terms" begin
    for FT in (Float32, Float64)
        #{{{
        let
            N = (4,)
            Nq = N .+ 1
            Np = prod(Nq)

            dim = length(N)
            nface = 2dim

            # Create element operators for each polynomial order
            ξω = ntuple(j -> Elements.lglpoints(FT, N[j]), dim)
            ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)

            D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

            dim = 1
            e2c = Array{FT, 3}(undef, 1, 2, 2)
            e2c[:, :, 1] = [-1 0]
            e2c[:, :, 2] = [0 10]
            nelem = size(e2c, 3)

            x1 = Array{FT, 2}(undef, Nq[1], nelem)
            Metrics.creategrid!(x1, e2c, ξ[1])
            @test x1[:, 1] ≈ (ξ[1] .- 1) / 2
            @test x1[:, 2] ≈ 5 * (ξ[1] .+ 1)

            metric = Metrics.computemetric(x1, D...)
            @test metric.J[:, 1] ≈ ones(FT, Nq) / 2
            @test metric.J[:, 2] ≈ 5 * ones(FT, Nq)
            @test metric.ξ1x1[:, 1] ≈ 2 * ones(FT, Nq)
            @test metric.ξ1x1[:, 2] ≈ ones(FT, Nq) / 5
            @test metric.n1[1, 1, :] ≈ -ones(FT, nelem)
            @test metric.n1[1, 2, :] ≈ ones(FT, nelem)
            @test metric.sJ[1, 1, :] ≈ ones(FT, nelem)
            @test metric.sJ[1, 2, :] ≈ ones(FT, nelem)
        end
        #}}}
    end
end

@testset "2-D Metric terms" begin
    # for FT in (Float32, Float64)
    for FT in (Float32, Float64), N in ((4, 4), (4, 6), (6, 4))
        Nq = N .+ 1
        Np = prod(Nq)
        Nfp = div.(Np, Nq)

        dim = length(N)
        nface = 2dim

        # Create element operators for each polynomial order
        ξω = ntuple(j -> Elements.lglpoints(FT, N[j]), dim)
        ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)
        D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

        # linear and rotation test
        #{{{
        let
            e2c = Array{FT, 3}(undef, 2, 4, 4)
            e2c[:, :, 1] = [
                0 2 0 2
                0 0 2 2
            ]
            e2c[:, :, 2] = [
                2 2 0 0
                0 2 0 2
            ]
            e2c[:, :, 3] = [
                2 0 2 0
                2 2 0 0
            ]
            e2c[:, :, 4] = [
                0 0 2 2
                2 0 2 0
            ]
            nelem = size(e2c, 3)

            x_exact = Array{FT, 3}(undef, Nq..., 4)
            x_exact[:, :, 1] .= 1 .+ ξ[1]
            x_exact[:, :, 2] .= 1 .- ξ[2]'
            x_exact[:, :, 3] .= 1 .- ξ[1]
            x_exact[:, :, 4] .= 1 .+ ξ[2]'

            y_exact = Array{FT, 3}(undef, Nq..., 4)
            y_exact[:, :, 1] .= 1 .+ ξ[2]'
            y_exact[:, :, 2] .= 1 .+ ξ[1]
            y_exact[:, :, 3] .= 1 .- ξ[2]'
            y_exact[:, :, 4] .= 1 .- ξ[1]

            J_exact = ones(FT, Nq..., 4)

            ξ1x1_exact = zeros(FT, Nq..., 4)
            ξ1x1_exact[:, :, 1] .= 1
            ξ1x1_exact[:, :, 3] .= -1

            ξ1x2_exact = zeros(FT, Nq..., 4)
            ξ1x2_exact[:, :, 2] .= 1
            ξ1x2_exact[:, :, 4] .= -1

            ξ2x1_exact = zeros(FT, Nq..., 4)
            ξ2x1_exact[:, :, 2] .= -1
            ξ2x1_exact[:, :, 4] .= 1

            ξ2x2_exact = zeros(FT, Nq..., 4)
            ξ2x2_exact[:, :, 1] .= 1
            ξ2x2_exact[:, :, 3] .= -1

            sJ_exact = fill(FT(NaN), maximum(Nfp), nface, nelem)
            sJ_exact[1:Nfp[1], 1, :] .= 1
            sJ_exact[1:Nfp[1], 2, :] .= 1
            sJ_exact[1:Nfp[2], 3, :] .= 1
            sJ_exact[1:Nfp[2], 4, :] .= 1

            nx_exact = fill(FT(NaN), maximum(Nfp), nface, nelem)
            nx_exact[1:Nfp[1], 1:2, :] .= 0
            nx_exact[1:Nfp[2], 3:4, :] .= 0

            nx_exact[1:Nfp[1], 1, 1] .= -1
            nx_exact[1:Nfp[1], 2, 1] .= 1
            nx_exact[1:Nfp[2], 3, 2] .= 1
            nx_exact[1:Nfp[2], 4, 2] .= -1
            nx_exact[1:Nfp[1], 1, 3] .= 1
            nx_exact[1:Nfp[1], 2, 3] .= -1
            nx_exact[1:Nfp[2], 3, 4] .= -1
            nx_exact[1:Nfp[2], 4, 4] .= 1

            ny_exact = fill(FT(NaN), maximum(Nfp), nface, nelem)
            ny_exact[1:Nfp[1], 1:2, :] .= 0
            ny_exact[1:Nfp[2], 3:4, :] .= 0

            ny_exact[1:Nfp[2], 3, 1] .= -1
            ny_exact[1:Nfp[2], 4, 1] .= 1
            ny_exact[1:Nfp[1], 1, 2] .= -1
            ny_exact[1:Nfp[1], 2, 2] .= 1
            ny_exact[1:Nfp[2], 3, 3] .= 1
            ny_exact[1:Nfp[2], 4, 3] .= -1
            ny_exact[1:Nfp[1], 1, 4] .= 1
            ny_exact[1:Nfp[1], 2, 4] .= -1

            vgeo = Array{FT, 4}(undef, Nq..., length(VGEO2D), nelem)
            sgeo =
                Array{FT, 4}(undef, maximum(Nfp), nface, length(SGEO2D), nelem)
            Metrics.creategrid!(
                ntuple(j -> (@view vgeo[:, :, j, :]), dim)...,
                e2c,
                ξ...,
            )
            Metrics.computemetric!(
                ntuple(j -> (@view vgeo[:, :, j, :]), length(VGEO2D))...,
                ntuple(j -> (@view sgeo[:, :, j, :]), length(SGEO2D))...,
                D...,
            )

            @test (@view vgeo[:, :, VGEO2D.x1, :]) ≈ x_exact
            @test (@view vgeo[:, :, VGEO2D.x2, :]) ≈ y_exact
            @test (@view vgeo[:, :, VGEO2D.J, :]) ≈ J_exact
            @test (@view vgeo[:, :, VGEO2D.ξ1x1, :]) ≈ ξ1x1_exact
            @test (@view vgeo[:, :, VGEO2D.ξ1x2, :]) ≈ ξ1x2_exact
            @test (@view vgeo[:, :, VGEO2D.ξ2x1, :]) ≈ ξ2x1_exact
            @test (@view vgeo[:, :, VGEO2D.ξ2x2, :]) ≈ ξ2x2_exact
            msk = isfinite.(sJ_exact)
            @test sgeo[:, :, SGEO2D.sJ, :][msk] ≈ sJ_exact[msk]
            @test sgeo[:, :, SGEO2D.n1, :][msk] ≈ nx_exact[msk]
            @test sgeo[:, :, SGEO2D.n2, :][msk] ≈ ny_exact[msk]

            nothing
        end
        #}}}

        # Polynomial 2-D test
        #{{{
        let
            f(ξ1, ξ2) = (
                9 .* ξ1 - (1 .+ ξ1) .* ξ2 .^ 2 +
                (ξ1 .- 1) .^ 2 .* (1 .- ξ2 .^ 2 .+ ξ2 .^ 3),
                10 .* ξ2 .+ ξ1 .^ 4 .* (1 .- ξ2) .+ ξ1 .^ 2 .* ξ2 .* (1 .+ ξ2),
            )
            fx1ξ1(ξ1, ξ2) =
                7 .+ ξ2 .^ 2 .- 2 .* ξ2 .^ 3 .+
                2 .* ξ1 .* (1 .- ξ2 .^ 2 .+ ξ2 .^ 3)
            fx1ξ2(ξ1, ξ2) =
                -2 .* (1 .+ ξ1) .* ξ2 .+
                (-1 .+ ξ1) .^ 2 .* ξ2 .* (-2 .+ 3 .* ξ2)
            fx2ξ1(ξ1, ξ2) =
                -4 .* ξ1 .^ 3 .* (-1 .+ ξ2) .+ 2 .* ξ1 .* ξ2 .* (1 .+ ξ2)
            fx2ξ2(ξ1, ξ2) = 10 .- ξ1 .^ 4 .+ ξ1 .^ 2 .* (1 .+ 2 .* ξ2)

            e2c = Array{FT, 3}(undef, 2, 4, 1)
            e2c[:, :, 1] = [-1 1 -1 1; -1 -1 1 1]
            nelem = size(e2c, 3)

            vgeo = Array{FT, 4}(undef, Nq..., length(VGEO2D), nelem)
            sgeo =
                Array{FT, 4}(undef, maximum(Nfp), nface, length(SGEO2D), nelem)

            Metrics.creategrid!(
                ntuple(j -> (@view vgeo[:, :, j, :]), dim)...,
                e2c,
                ξ...,
            )
            x1 = @view vgeo[:, :, VGEO2D.x1, :]
            x2 = @view vgeo[:, :, VGEO2D.x2, :]

            (x1ξ1, x1ξ2, x2ξ1, x2ξ2) =
                (fx1ξ1(x1, x2), fx1ξ2(x1, x2), fx2ξ1(x1, x2), fx2ξ2(x1, x2))
            J = x1ξ1 .* x2ξ2 - x1ξ2 .* x2ξ1
            foreach(j -> (x1[j], x2[j]) = f(x1[j], x2[j]), 1:length(x1))

            Metrics.computemetric!(
                ntuple(j -> (@view vgeo[:, :, j, :]), length(VGEO2D))...,
                ntuple(j -> (@view sgeo[:, :, j, :]), length(SGEO2D))...,
                D...,
            )
            @test J ≈ (@view vgeo[:, :, VGEO2D.J, :])
            @test (@view vgeo[:, :, VGEO2D.ξ1x1, :]) ≈ x2ξ2 ./ J
            @test (@view vgeo[:, :, VGEO2D.ξ2x1, :]) ≈ -x2ξ1 ./ J
            @test (@view vgeo[:, :, VGEO2D.ξ1x2, :]) ≈ -x1ξ2 ./ J
            @test (@view vgeo[:, :, VGEO2D.ξ2x2, :]) ≈ x1ξ1 ./ J

            # check the normals?
            n1 = @view sgeo[:, :, SGEO2D.n1, :]
            n2 = @view sgeo[:, :, SGEO2D.n2, :]
            sJ = @view sgeo[:, :, SGEO2D.sJ, :]
            @test all(hypot.(n1[1:Nfp[1], 1:2, :], n2[1:Nfp[1], 1:2, :]) .≈ 1)
            @test all(hypot.(n1[1:Nfp[2], 3:4, :], n2[1:Nfp[2], 3:4, :]) .≈ 1)
            @test sJ[1:Nfp[1], 1, :] .* n1[1:Nfp[1], 1, :] ≈ -x2ξ2[1, :, :]
            @test sJ[1:Nfp[1], 1, :] .* n2[1:Nfp[1], 1, :] ≈ x1ξ2[1, :, :]
            @test sJ[1:Nfp[1], 2, :] .* n1[1:Nfp[1], 2, :] ≈ x2ξ2[Nq[1], :, :]
            @test sJ[1:Nfp[1], 2, :] .* n2[1:Nfp[1], 2, :] ≈ -x1ξ2[Nq[1], :, :]
            @test sJ[1:Nfp[2], 3, :] .* n1[1:Nfp[2], 3, :] ≈ x2ξ1[:, 1, :]
            @test sJ[1:Nfp[2], 3, :] .* n2[1:Nfp[2], 3, :] ≈ -x1ξ1[:, 1, :]
            @test sJ[1:Nfp[2], 4, :] .* n1[1:Nfp[2], 4, :] ≈ -x2ξ1[:, Nq[2], :]
            @test sJ[1:Nfp[2], 4, :] .* n2[1:Nfp[2], 4, :] ≈ x1ξ1[:, Nq[2], :]
        end
        #}}}

        # Constant preserving test
        #{{{
        let
            rng = MersenneTwister(777)
            f(ξ1, ξ2) = (
                ξ1 + (ξ1 * ξ2 * rand(rng) + rand(rng)) / 10,
                ξ2 + (ξ1 * ξ2 * rand(rng) + rand(rng)) / 10,
            )

            e2c = Array{FT, 3}(undef, 2, 4, 1)
            e2c[:, :, 1] = [-1 1 -1 1; -1 -1 1 1]
            nelem = size(e2c, 3)

            vgeo = Array{FT, 4}(undef, Nq..., length(VGEO2D), nelem)
            sgeo =
                Array{FT, 4}(undef, maximum(Nfp), nface, length(SGEO2D), nelem)

            Metrics.creategrid!(
                ntuple(j -> (@view vgeo[:, :, j, :]), dim)...,
                e2c,
                ξ...,
            )
            x1 = @view vgeo[:, :, VGEO2D.x1, :]
            x2 = @view vgeo[:, :, VGEO2D.x2, :]

            foreach(j -> (x1[j], x2[j]) = f(x1[j], x2[j]), 1:length(x1))

            Metrics.computemetric!(
                ntuple(j -> (@view vgeo[:, :, j, :]), length(VGEO2D))...,
                ntuple(j -> (@view sgeo[:, :, j, :]), length(SGEO2D))...,
                D...,
            )

            (Cx1, Cx2) = (zeros(FT, Nq...), zeros(FT, Nq...))

            J = @view vgeo[:, :, VGEO2D.J, :]
            ξ1x1 = @view vgeo[:, :, VGEO2D.ξ1x1, :]
            ξ1x2 = @view vgeo[:, :, VGEO2D.ξ1x2, :]
            ξ2x1 = @view vgeo[:, :, VGEO2D.ξ2x1, :]
            ξ2x2 = @view vgeo[:, :, VGEO2D.ξ2x2, :]

            e = 1
            for n in 1:Nq[2]
                Cx1[:, n] += D[1] * (J[:, n, e] .* ξ1x1[:, n, e])
                Cx2[:, n] += D[1] * (J[:, n, e] .* ξ1x2[:, n, e])
            end
            for n in 1:Nq[1]
                Cx1[n, :] += D[2] * (J[n, :, e] .* ξ2x1[n, :, e])
                Cx2[n, :] += D[2] * (J[n, :, e] .* ξ2x2[n, :, e])
            end
            @test maximum(abs.(Cx1)) ≤ 100 * eps(FT)
            @test maximum(abs.(Cx2)) ≤ 100 * eps(FT)
        end
        #}}}
    end
end

@testset "3-D Metric terms" begin
    # linear test
    #{{{
    for FT in (Float32, Float64), N in ((2, 2, 2), (2, 3, 4), (4, 3, 2))
        Nq = N .+ 1
        Np = prod(Nq)
        Nfp = div.(Np, Nq)

        dim = length(N)
        nface = 2dim

        # Create element operators for each polynomial order
        ξω = ntuple(j -> Elements.lglpoints(FT, N[j]), dim)
        ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)
        D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

        e2c = Array{FT, 3}(undef, dim, 8, 2)
        e2c[:, :, 1] = [
            0 2 0 2 0 2 0 2
            0 0 2 2 0 0 2 2
            0 0 0 0 2 2 2 2
        ]
        e2c[:, :, 2] = [
            2 2 0 0 2 2 0 0
            0 2 0 2 0 2 0 2
            0 0 0 0 2 2 2 2
        ]

        nelem = size(e2c, 3)

        x_exact = Array{FT, 4}(undef, Nq..., nelem)
        x_exact[:, :, :, 1] .= 1 .+ ξ[1]
        x_exact[:, :, :, 2] .= 1 .- ξ[2]'

        ξ1x1_exact = zeros(Int, Nq..., nelem)
        ξ1x1_exact[:, :, :, 1] .= 1

        ξ1x2_exact = zeros(Int, Nq..., nelem)
        ξ1x2_exact[:, :, :, 2] .= 1

        ξ2x1_exact = zeros(Int, Nq..., nelem)
        ξ2x1_exact[:, :, :, 2] .= -1

        ξ2x2_exact = zeros(Int, Nq..., nelem)
        ξ2x2_exact[:, :, :, 1] .= 1

        ξ3x3_exact = ones(Int, Nq..., nelem)

        y_exact = Array{FT, 4}(undef, Nq..., nelem)
        y_exact[:, :, :, 1] .= 1 .+ ξ[2]'
        y_exact[:, :, :, 2] .= 1 .+ ξ[1]

        z_exact = Array{FT, 4}(undef, Nq..., nelem)
        z_exact[:, :, :, :] .= reshape(1 .+ ξ[3], 1, 1, Nq[3])

        J_exact = ones(Int, Nq..., nelem)

        sJ_exact = ones(Int, maximum(Nfp), nface, nelem)

        nx_exact = zeros(Int, maximum(Nfp), nface, nelem)
        nx_exact[:, 1, 1] .= -1
        nx_exact[:, 2, 1] .= 1
        nx_exact[:, 3, 2] .= 1
        nx_exact[:, 4, 2] .= -1

        ny_exact = zeros(Int, maximum(Nfp), nface, nelem)
        ny_exact[:, 3, 1] .= -1
        ny_exact[:, 4, 1] .= 1
        ny_exact[:, 1, 2] .= -1
        ny_exact[:, 2, 2] .= 1

        nz_exact = zeros(Int, maximum(Nfp), nface, nelem)
        nz_exact[:, 5, 1:2] .= -1
        nz_exact[:, 6, 1:2] .= 1

        vgeo = Array{FT, 5}(undef, Nq..., length(VGEO3D), nelem)
        sgeo = Array{FT, 4}(undef, maximum(Nfp), nface, length(SGEO3D), nelem)
        Metrics.creategrid!(
            ntuple(j -> (@view vgeo[:, :, :, j, :]), dim)...,
            e2c,
            ξ...,
        )
        Metrics.computemetric!(
            ntuple(j -> (@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
            ntuple(j -> (@view sgeo[:, :, j, :]), length(SGEO3D))...,
            D...,
        )

        @test (@view vgeo[:, :, :, VGEO3D.x1, :]) ≈ x_exact
        @test (@view vgeo[:, :, :, VGEO3D.x2, :]) ≈ y_exact
        @test (@view vgeo[:, :, :, VGEO3D.x3, :]) ≈ z_exact
        @test (@view vgeo[:, :, :, VGEO3D.J, :]) ≈ J_exact
        @test (@view vgeo[:, :, :, VGEO3D.ξ1x1, :]) ≈ ξ1x1_exact
        @test (@view vgeo[:, :, :, VGEO3D.ξ1x2, :]) ≈ ξ1x2_exact
        @test maximum(abs.(@view vgeo[:, :, :, VGEO3D.ξ1x3, :])) ≤ 100 * eps(FT)
        @test (@view vgeo[:, :, :, VGEO3D.ξ2x1, :]) ≈ ξ2x1_exact
        @test (@view vgeo[:, :, :, VGEO3D.ξ2x2, :]) ≈ ξ2x2_exact
        @test maximum(abs.(@view vgeo[:, :, :, VGEO3D.ξ2x3, :])) ≤ 100 * eps(FT)
        @test maximum(abs.(@view vgeo[:, :, :, VGEO3D.ξ3x1, :])) ≤ 100 * eps(FT)
        @test maximum(abs.(@view vgeo[:, :, :, VGEO3D.ξ3x2, :])) ≤ 100 * eps(FT)
        @test (@view vgeo[:, :, :, VGEO3D.ξ3x3, :]) ≈ ξ3x3_exact
        for d in 1:dim
            for f in (2d - 1):(2d)
                @test isapprox(
                    (@view sgeo[1:Nfp[d], f, SGEO3D.sJ, :]),
                    sJ_exact[1:Nfp[d], f, :];
                    atol = √eps(FT),
                    rtol = √eps(FT),
                )
                @test isapprox(
                    (@view sgeo[1:Nfp[d], f, SGEO3D.n1, :]),
                    nx_exact[1:Nfp[d], f, :];
                    atol = √eps(FT),
                    rtol = √eps(FT),
                )
                @test isapprox(
                    (@view sgeo[1:Nfp[d], f, SGEO3D.n2, :]),
                    ny_exact[1:Nfp[d], f, :];
                    atol = √eps(FT),
                    rtol = √eps(FT),
                )
                @test isapprox(
                    (@view sgeo[1:Nfp[d], f, SGEO3D.n3, :]),
                    nz_exact[1:Nfp[d], f, :];
                    atol = √eps(FT),
                    rtol = √eps(FT),
                )
            end
        end
    end
    #}}}

    # Polynomial 3-D test
    #{{{
    for FT in (Float32, Float64), N in ((9, 9, 9), (9, 9, 10), (10, 9, 11))
        Nq = N .+ 1
        Np = prod(Nq)
        Nfp = div.(Np, Nq)

        dim = length(N)
        nface = 2dim

        # Create element operators for each polynomial order
        ξω = ntuple(j -> Elements.lglpoints(FT, N[j]), dim)
        ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)
        D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

        f(ξ1, ξ2, ξ3) = @.( (
            ξ2 + ξ1 * ξ3 - (ξ1^2 * ξ2^2 * ξ3^2) / 4,
            ξ3 - ((ξ1 * ξ2 * ξ3 + 1) / 2)^3 + 1,
            ξ1 + 2 * ((ξ1 + 1) / 2)^6 * ((ξ2 + 1) / 2)^6 * ((ξ3 + 1) / 2)^6,
        ))

        fx1ξ1(ξ1, ξ2, ξ3) = @.(ξ3 - (ξ1 * ξ2^2 * ξ3^2) / 2)
        fx1ξ2(ξ1, ξ2, ξ3) = @.(1 - (ξ1^2 * ξ2 * ξ3^2) / 2)
        fx1ξ3(ξ1, ξ2, ξ3) = @.(ξ1 - (ξ1^2 * ξ2^2 * ξ3) / 2)
        fx2ξ1(ξ1, ξ2, ξ3) = @.(-(3 * ξ2 * ξ3 * ((ξ1 * ξ2 * ξ3 + 1) / 2)^2) / 2)
        fx2ξ2(ξ1, ξ2, ξ3) = @.(-(3 * ξ1 * ξ3 * ((ξ1 * ξ2 * ξ3 + 1) / 2)^2) / 2)
        fx2ξ3(ξ1, ξ2, ξ3) =
            @.(1 - (3 * ξ1 * ξ2 * ((ξ1 * ξ2 * ξ3 + 1) / 2)^2) / 2)
        fx3ξ1(ξ1, ξ2, ξ3) =
            @.(6 * ((ξ1 + 1) / 2)^5 * ((ξ2 + 1) / 2)^6 * ((ξ3 + 1) / 2)^6 + 1)
        fx3ξ2(ξ1, ξ2, ξ3) =
            @.(6 * ((ξ1 + 1) / 2)^6 * ((ξ2 + 1) / 2)^5 * ((ξ3 + 1) / 2)^6)
        fx3ξ3(ξ1, ξ2, ξ3) =
            @.(6 * ((ξ1 + 1) / 2)^6 * ((ξ2 + 1) / 2)^6 * ((ξ3 + 1) / 2)^5)

        e2c = Array{FT, 3}(undef, dim, 8, 1)
        e2c[:, :, 1] = [
            -1 1 -1 1 -1 1 -1 1
            -1 -1 1 1 -1 -1 1 1
            -1 -1 -1 -1 1 1 1 1
        ]

        nelem = size(e2c, 3)

        vgeo = Array{FT, 5}(undef, Nq..., length(VGEO3D), nelem)
        sgeo = Array{FT, 4}(undef, maximum(Nfp), nface, length(SGEO3D), nelem)
        Metrics.creategrid!(
            ntuple(j -> (@view vgeo[:, :, :, j, :]), dim)...,
            e2c,
            ξ...,
        )
        x1 = @view vgeo[:, :, :, VGEO3D.x1, :]
        x2 = @view vgeo[:, :, :, VGEO3D.x2, :]
        x3 = @view vgeo[:, :, :, VGEO3D.x3, :]

        # Compute exact metrics
        (x1ξ1, x1ξ2, x1ξ3, x2ξ1, x2ξ2, x2ξ3, x3ξ1, x3ξ2, x3ξ3) = (
            fx1ξ1(x1, x2, x3),
            fx1ξ2(x1, x2, x3),
            fx1ξ3(x1, x2, x3),
            fx2ξ1(x1, x2, x3),
            fx2ξ2(x1, x2, x3),
            fx2ξ3(x1, x2, x3),
            fx3ξ1(x1, x2, x3),
            fx3ξ2(x1, x2, x3),
            fx3ξ3(x1, x2, x3),
        )
        J = (
            x1ξ1 .* (x2ξ2 .* x3ξ3 - x2ξ3 .* x3ξ2) +
            x2ξ1 .* (x3ξ2 .* x1ξ3 - x3ξ3 .* x1ξ2) +
            x3ξ1 .* (x1ξ2 .* x2ξ3 - x1ξ3 .* x2ξ2)
        )

        ξ1x1 = (x2ξ2 .* x3ξ3 - x2ξ3 .* x3ξ2) ./ J
        ξ1x2 = (x3ξ2 .* x1ξ3 - x3ξ3 .* x1ξ2) ./ J
        ξ1x3 = (x1ξ2 .* x2ξ3 - x1ξ3 .* x2ξ2) ./ J
        ξ2x1 = (x2ξ3 .* x3ξ1 - x2ξ1 .* x3ξ3) ./ J
        ξ2x2 = (x3ξ3 .* x1ξ1 - x3ξ1 .* x1ξ3) ./ J
        ξ2x3 = (x1ξ3 .* x2ξ1 - x1ξ1 .* x2ξ3) ./ J
        ξ3x1 = (x2ξ1 .* x3ξ2 - x2ξ2 .* x3ξ1) ./ J
        ξ3x2 = (x3ξ1 .* x1ξ2 - x3ξ2 .* x1ξ1) ./ J
        ξ3x3 = (x1ξ1 .* x2ξ2 - x1ξ2 .* x2ξ1) ./ J

        # Warp the mesh
        foreach(
            j -> (x1[j], x2[j], x3[j]) = f(x1[j], x2[j], x3[j]),
            1:length(x1),
        )

        Metrics.computemetric!(
            ntuple(j -> (@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
            ntuple(j -> (@view sgeo[:, :, j, :]), length(SGEO3D))...,
            D...,
        )
        sgeo = reshape(sgeo, maximum(Nfp), nface, length(SGEO3D), nelem)

        @test (@view vgeo[:, :, :, VGEO3D.J, :]) ≈ J
        @test (@view vgeo[:, :, :, VGEO3D.ξ1x1, :]) ≈ ξ1x1
        @test (@view vgeo[:, :, :, VGEO3D.ξ1x2, :]) ≈ ξ1x2
        @test (@view vgeo[:, :, :, VGEO3D.ξ1x3, :]) ≈ ξ1x3
        @test (@view vgeo[:, :, :, VGEO3D.ξ2x1, :]) ≈ ξ2x1
        @test (@view vgeo[:, :, :, VGEO3D.ξ2x2, :]) ≈ ξ2x2
        @test (@view vgeo[:, :, :, VGEO3D.ξ2x3, :]) ≈ ξ2x3
        @test (@view vgeo[:, :, :, VGEO3D.ξ3x1, :]) ≈ ξ3x1
        @test (@view vgeo[:, :, :, VGEO3D.ξ3x2, :]) ≈ ξ3x2
        @test (@view vgeo[:, :, :, VGEO3D.ξ3x3, :]) ≈ ξ3x3
        n1 = @view sgeo[:, :, SGEO3D.n1, :]
        n2 = @view sgeo[:, :, SGEO3D.n2, :]
        n3 = @view sgeo[:, :, SGEO3D.n3, :]
        sJ = @view sgeo[:, :, SGEO3D.sJ, :]
        for d in 1:dim
            for f in (2d - 1):(2d)
                @test all(
                    hypot.(
                        n1[1:Nfp[d], f, :],
                        n2[1:Nfp[d], f, :],
                        n3[1:Nfp[d], f, :],
                    ) .≈ 1,
                )
            end
        end
        d, f = 1, 1
        @test [
            (sJ[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (-J[1, :, :, :] .* ξ1x1[1, :, :, :])[:],
            (-J[1, :, :, :] .* ξ1x2[1, :, :, :])[:],
            (-J[1, :, :, :] .* ξ1x3[1, :, :, :])[:],
        ]
        d, f = 1, 2
        @test [
            (sJ[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (J[Nq[d], :, :, :] .* ξ1x1[Nq[d], :, :, :])[:],
            (J[Nq[d], :, :, :] .* ξ1x2[Nq[d], :, :, :])[:],
            (J[Nq[d], :, :, :] .* ξ1x3[Nq[d], :, :, :])[:],
        ]
        d, f = 2, 3
        @test [
            (sJ[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (-J[:, 1, :, :] .* ξ2x1[:, 1, :, :])[:],
            (-J[:, 1, :, :] .* ξ2x2[:, 1, :, :])[:],
            (-J[:, 1, :, :] .* ξ2x3[:, 1, :, :])[:],
        ]
        d, f = 2, 4
        @test [
            (sJ[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (J[:, Nq[d], :, :] .* ξ2x1[:, Nq[d], :, :])[:],
            (J[:, Nq[d], :, :] .* ξ2x2[:, Nq[d], :, :])[:],
            (J[:, Nq[d], :, :] .* ξ2x3[:, Nq[d], :, :])[:],
        ]
        d, f = 3, 5
        @test [
            (sJ[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (-J[:, :, 1, :] .* ξ3x1[:, :, 1, :])[:],
            (-J[:, :, 1, :] .* ξ3x2[:, :, 1, :])[:],
            (-J[:, :, 1, :] .* ξ3x3[:, :, 1, :])[:],
        ]
        d, f = 3, 6
        @test [
            (sJ[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sJ[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (J[:, :, Nq[d], :] .* ξ3x1[:, :, Nq[d], :])[:],
            (J[:, :, Nq[d], :] .* ξ3x2[:, :, Nq[d], :])[:],
            (J[:, :, Nq[d], :] .* ξ3x3[:, :, Nq[d], :])[:],
        ]
    end
    #}}}

    # Constant preserving test
    #{{{
    for FT in (Float32, Float64), N in ((5, 5, 5), (3, 4, 5), (4, 4, 5))
        Nq = N .+ 1
        Np = prod(Nq)
        Nfp = div.(Np, Nq)

        dim = length(N)
        nface = 2dim

        # Create element operators for each polynomial order
        ξω = ntuple(j -> Elements.lglpoints(FT, N[j]), dim)
        ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)
        D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

        f(ξ1, ξ2, ξ3) = @.( (
            ξ2 + ξ1 * ξ3 - (ξ1^2 * ξ2^2 * ξ3^2) / 4,
            ξ3 - ((ξ1 * ξ2 * ξ3 + 1) / 2)^3 + 1,
            ξ1 + ((ξ1 + 1) / 2)^6 * ((ξ2 + 1) / 2)^6 * ((ξ3 + 1) / 2)^6,
        ))

        e2c = Array{FT, 3}(undef, dim, 8, 1)
        e2c[:, :, 1] = [
            -1 1 -1 1 -1 1 -1 1
            -1 -1 1 1 -1 -1 1 1
            -1 -1 -1 -1 1 1 1 1
        ]

        nelem = size(e2c, 3)

        vgeo = Array{FT, 5}(undef, Nq..., length(VGEO3D), nelem)
        sgeo = Array{FT, 4}(undef, maximum(Nfp), nface, length(SGEO3D), nelem)
        Metrics.creategrid!(
            ntuple(j -> (@view vgeo[:, :, :, j, :]), dim)...,
            e2c,
            ξ...,
        )
        x1 = @view vgeo[:, :, :, VGEO3D.x1, :]
        x2 = @view vgeo[:, :, :, VGEO3D.x2, :]
        x3 = @view vgeo[:, :, :, VGEO3D.x3, :]

        foreach(
            j -> (x1[j], x2[j], x3[j]) = f(x1[j], x2[j], x3[j]),
            1:length(x1),
        )

        Metrics.computemetric!(
            ntuple(j -> (@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
            ntuple(j -> (@view sgeo[:, :, j, :]), length(SGEO3D))...,
            D...,
        )

        (Cx1, Cx2, Cx3) = (zeros(FT, Nq...), zeros(FT, Nq...), zeros(FT, Nq...))

        J = @view vgeo[:, :, :, VGEO3D.J, :]
        ξ1x1 = @view vgeo[:, :, :, VGEO3D.ξ1x1, :]
        ξ1x2 = @view vgeo[:, :, :, VGEO3D.ξ1x2, :]
        ξ1x3 = @view vgeo[:, :, :, VGEO3D.ξ1x3, :]
        ξ2x1 = @view vgeo[:, :, :, VGEO3D.ξ2x1, :]
        ξ2x2 = @view vgeo[:, :, :, VGEO3D.ξ2x2, :]
        ξ2x3 = @view vgeo[:, :, :, VGEO3D.ξ2x3, :]
        ξ3x1 = @view vgeo[:, :, :, VGEO3D.ξ3x1, :]
        ξ3x2 = @view vgeo[:, :, :, VGEO3D.ξ3x2, :]
        ξ3x3 = @view vgeo[:, :, :, VGEO3D.ξ3x3, :]

        e = 1
        for k in 1:Nq[3]
            for j in 1:Nq[2]
                Cx1[:, j, k] += D[1] * (J[:, j, k, e] .* ξ1x1[:, j, k, e])
                Cx2[:, j, k] += D[1] * (J[:, j, k, e] .* ξ1x2[:, j, k, e])
                Cx3[:, j, k] += D[1] * (J[:, j, k, e] .* ξ1x3[:, j, k, e])
            end
        end

        for k in 1:Nq[3]
            for i in 1:Nq[1]
                Cx1[i, :, k] += D[2] * (J[i, :, k, e] .* ξ2x1[i, :, k, e])
                Cx2[i, :, k] += D[2] * (J[i, :, k, e] .* ξ2x2[i, :, k, e])
                Cx3[i, :, k] += D[2] * (J[i, :, k, e] .* ξ2x3[i, :, k, e])
            end
        end


        for j in 1:Nq[2]
            for i in 1:Nq[1]
                Cx1[i, j, :] += D[3] * (J[i, j, :, e] .* ξ3x1[i, j, :, e])
                Cx2[i, j, :] += D[3] * (J[i, j, :, e] .* ξ3x2[i, j, :, e])
                Cx3[i, j, :] += D[3] * (J[i, j, :, e] .* ξ3x3[i, j, :, e])
            end
        end
        @test maximum(abs.(Cx1)) ≤ 300 * eps(FT)
        @test maximum(abs.(Cx2)) ≤ 300 * eps(FT)
        @test maximum(abs.(Cx3)) ≤ 300 * eps(FT)
    end
    #}}}
end
