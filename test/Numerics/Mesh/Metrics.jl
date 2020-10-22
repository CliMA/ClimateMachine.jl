using ClimateMachine.Mesh.Elements
using ClimateMachine.Mesh.Metrics
using Test


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
            N = 4

            ξ1, w = Elements.lglpoints(FT, N)
            D = Elements.spectralderivative(ξ1)
            Nq = N + 1

            dim = 1
            e2c = Array{FT, 3}(undef, 1, 2, 2)
            e2c[:, :, 1] = [-1 0]
            e2c[:, :, 2] = [0 10]
            nelem = size(e2c, 3)

            (x1,) = Metrics.creategrid1d(e2c, ξ1)
            @test x1[:, 1] ≈ (ξ1 .- 1) / 2
            @test x1[:, 2] ≈ 5 * (ξ1 .+ 1)

            metric = Metrics.computemetric(x1, D)
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
    for FT in (Float32, Float64)
        # linear and rotation test
        #{{{
        let
            N = 2

            ξ1, w = Elements.lglpoints(FT, N)
            D = Elements.spectralderivative(ξ1)
            Nq = N + 1

            dim = 2
            nface = 2dim
            e2c = Array{FT, 3}(undef, 2, 4, 4)
            e2c[:, :, 1] = [0 2 0 2; 0 0 2 2]
            e2c[:, :, 2] = [2 2 0 0; 0 2 0 2]
            e2c[:, :, 3] = [2 0 2 0; 2 2 0 0]
            e2c[:, :, 4] = [0 0 2 2; 2 0 2 0]
            nelem = size(e2c, 3)

            x_exact = Array{Int, 3}(undef, 3, 3, 4)
            x_exact[:, :, 1] = [0 0 0; 1 1 1; 2 2 2]
            x_exact[:, :, 2] = rotr90(x_exact[:, :, 1])
            x_exact[:, :, 3] = rotr90(x_exact[:, :, 2])
            x_exact[:, :, 4] = rotr90(x_exact[:, :, 3])

            y_exact = Array{Int, 3}(undef, 3, 3, 4)
            y_exact[:, :, 1] = [0 1 2; 0 1 2; 0 1 2]
            y_exact[:, :, 2] = rotr90(y_exact[:, :, 1])
            y_exact[:, :, 3] = rotr90(y_exact[:, :, 2])
            y_exact[:, :, 4] = rotr90(y_exact[:, :, 3])

            J_exact = ones(Int, 3, 3, 4)

            ξ1x1_exact = zeros(Int, 3, 3, 4)
            ξ1x1_exact[:, :, 1] .= 1
            ξ1x1_exact[:, :, 3] .= -1

            ξ1x2_exact = zeros(Int, 3, 3, 4)
            ξ1x2_exact[:, :, 2] .= 1
            ξ1x2_exact[:, :, 4] .= -1

            ξ2x1_exact = zeros(Int, 3, 3, 4)
            ξ2x1_exact[:, :, 2] .= -1
            ξ2x1_exact[:, :, 4] .= 1

            ξ2x2_exact = zeros(Int, 3, 3, 4)
            ξ2x2_exact[:, :, 1] .= 1
            ξ2x2_exact[:, :, 3] .= -1

            sJ_exact = ones(Int, Nq, nface, nelem)

            nx_exact = zeros(Int, Nq, nface, nelem)
            nx_exact[:, 1, 1] .= -1
            nx_exact[:, 2, 1] .= 1
            nx_exact[:, 3, 2] .= 1
            nx_exact[:, 4, 2] .= -1
            nx_exact[:, 1, 3] .= 1
            nx_exact[:, 2, 3] .= -1
            nx_exact[:, 3, 4] .= -1
            nx_exact[:, 4, 4] .= 1

            ny_exact = zeros(Int, Nq, nface, nelem)
            ny_exact[:, 3, 1] .= -1
            ny_exact[:, 4, 1] .= 1
            ny_exact[:, 1, 2] .= -1
            ny_exact[:, 2, 2] .= 1
            ny_exact[:, 3, 3] .= 1
            ny_exact[:, 4, 3] .= -1
            ny_exact[:, 1, 4] .= 1
            ny_exact[:, 2, 4] .= -1

            vgeo = Array{FT, 4}(undef, Nq, Nq, length(VGEO2D), nelem)
            sgeo = Array{FT, 4}(undef, Nq, nface, length(SGEO2D), nelem)
            Metrics.creategrid!(
                ntuple(j -> (@view vgeo[:, :, j, :]), dim)...,
                e2c,
                ξ1,
            )
            Metrics.computemetric!(
                ntuple(j -> (@view vgeo[:, :, j, :]), length(VGEO2D))...,
                ntuple(j -> (@view sgeo[:, :, j, :]), length(SGEO2D))...,
                D,
            )

            @test (@view vgeo[:, :, VGEO2D.x1, :]) ≈ x_exact
            @test (@view vgeo[:, :, VGEO2D.x2, :]) ≈ y_exact
            @test (@view vgeo[:, :, VGEO2D.J, :]) ≈ J_exact
            @test (@view vgeo[:, :, VGEO2D.ξ1x1, :]) ≈ ξ1x1_exact
            @test (@view vgeo[:, :, VGEO2D.ξ1x2, :]) ≈ ξ1x2_exact
            @test (@view vgeo[:, :, VGEO2D.ξ2x1, :]) ≈ ξ2x1_exact
            @test (@view vgeo[:, :, VGEO2D.ξ2x2, :]) ≈ ξ2x2_exact
            @test (@view sgeo[:, :, SGEO2D.sJ, :]) ≈ sJ_exact
            @test (@view sgeo[:, :, SGEO2D.n1, :]) ≈ nx_exact
            @test (@view sgeo[:, :, SGEO2D.n2, :]) ≈ ny_exact

            nothing
        end
        #}}}

        # Polynomial 2-D test
        #{{{
        let
            N = 4

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

            ξ1, w = Elements.lglpoints(FT, N)
            D = Elements.spectralderivative(ξ1)
            Nq = N + 1

            dim = 2
            nface = 2dim
            e2c = Array{FT, 3}(undef, 2, 4, 1)
            e2c[:, :, 1] = [-1 1 -1 1; -1 -1 1 1]
            nelem = size(e2c, 3)

            vgeo = Array{FT, 4}(undef, Nq, Nq, length(VGEO2D), nelem)
            sgeo = Array{FT, 4}(undef, Nq, nface, length(SGEO2D), nelem)

            Metrics.creategrid!(
                ntuple(j -> (@view vgeo[:, :, j, :]), dim)...,
                e2c,
                ξ1,
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
                D,
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
            @test hypot.(n1, n2) ≈ ones(FT, size(n1))
            @test sJ[:, 1, :] .* n1[:, 1, :] ≈ -x2ξ2[1, :, :]
            @test sJ[:, 1, :] .* n2[:, 1, :] ≈ x1ξ2[1, :, :]
            @test sJ[:, 2, :] .* n1[:, 2, :] ≈ x2ξ2[Nq, :, :]
            @test sJ[:, 2, :] .* n2[:, 2, :] ≈ -x1ξ2[Nq, :, :]
            @test sJ[:, 3, :] .* n1[:, 3, :] ≈ x2ξ1[:, 1, :]
            @test sJ[:, 3, :] .* n2[:, 3, :] ≈ -x1ξ1[:, 1, :]
            @test sJ[:, 4, :] .* n1[:, 4, :] ≈ -x2ξ1[:, Nq, :]
            @test sJ[:, 4, :] .* n2[:, 4, :] ≈ x1ξ1[:, Nq, :]
        end
        #}}}

        #{{{
        let
            N = 4

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

            ξ1, w = Elements.lglpoints(FT, N)
            D = Elements.spectralderivative(ξ1)
            Nq = N + 1

            dim = 2
            e2c = Array{FT, 3}(undef, 2, 4, 1)
            e2c[:, :, 1] = [-1 1 -1 1; -1 -1 1 1]
            nelem = size(e2c, 3)

            (x1, x2) = Metrics.creategrid2d(e2c, ξ1)

            (x1ξ1, x1ξ2, x2ξ1, x2ξ2) =
                (fx1ξ1(x1, x2), fx1ξ2(x1, x2), fx2ξ1(x1, x2), fx2ξ2(x1, x2))
            J = x1ξ1 .* x2ξ2 - x1ξ2 .* x2ξ1
            foreach(j -> (x1[j], x2[j]) = f(x1[j], x2[j]), 1:length(x1))

            metric = Metrics.computemetric(x1, x2, D)
            @test J ≈ metric.J
            @test metric.ξ1x1 ≈ x2ξ2 ./ J
            @test metric.ξ2x1 ≈ -x2ξ1 ./ J
            @test metric.ξ1x2 ≈ -x1ξ2 ./ J
            @test metric.ξ2x2 ≈ x1ξ1 ./ J

            # check the normals?
            n1 = metric.n1
            n2 = metric.n2
            sJ = metric.sJ
            @test hypot.(n1, n2) ≈ ones(FT, size(n1))
            @test sJ[:, 1, :] .* n1[:, 1, :] ≈ -x2ξ2[1, :, :]
            @test sJ[:, 1, :] .* n2[:, 1, :] ≈ x1ξ2[1, :, :]
            @test sJ[:, 2, :] .* n1[:, 2, :] ≈ x2ξ2[Nq, :, :]
            @test sJ[:, 2, :] .* n2[:, 2, :] ≈ -x1ξ2[Nq, :, :]
            @test sJ[:, 3, :] .* n1[:, 3, :] ≈ x2ξ1[:, 1, :]
            @test sJ[:, 3, :] .* n2[:, 3, :] ≈ -x1ξ1[:, 1, :]
            @test sJ[:, 4, :] .* n1[:, 4, :] ≈ -x2ξ1[:, Nq, :]
            @test sJ[:, 4, :] .* n2[:, 4, :] ≈ x1ξ1[:, Nq, :]
        end
        #}}}
    end

    # Constant preserving test
    #{{{
    let
        N = 4
        FT = Float64

        f(ξ1, ξ2) = (
            9 .* ξ1 - (1 .+ ξ1) .* ξ2 .^ 2 +
            (ξ1 .- 1) .^ 2 .* (1 .- ξ2 .^ 2 .+ ξ2 .^ 3),
            10 .* ξ2 .+ ξ1 .^ 4 .* (1 .- ξ2) .+ ξ1 .^ 2 .* ξ2 .* (1 .+ ξ2),
        )
        fx1ξ1(ξ1, ξ2) =
            7 .+ ξ2 .^ 2 .- 2 .* ξ2 .^ 3 .+ 2 .* ξ1 .* (1 .- ξ2 .^ 2 .+ ξ2 .^ 3)
        fx1ξ2(ξ1, ξ2) =
            -2 .* (1 .+ ξ1) .* ξ2 .+ (-1 .+ ξ1) .^ 2 .* ξ2 .* (-2 .+ 3 .* ξ2)
        fx2ξ1(ξ1, ξ2) =
            -4 .* ξ1 .^ 3 .* (-1 .+ ξ2) .+ 2 .* ξ1 .* ξ2 .* (1 .+ ξ2)
        fx2ξ2(ξ1, ξ2) = 10 .- ξ1 .^ 4 .+ ξ1 .^ 2 .* (1 .+ 2 .* ξ2)

        ξ1, w = Elements.lglpoints(FT, N)
        D = Elements.spectralderivative(ξ1)
        Nq = N + 1

        dim = 2
        nface = 2dim
        e2c = Array{FT, 3}(undef, 2, 4, 1)
        e2c[:, :, 1] = [-1 1 -1 1; -1 -1 1 1]
        nelem = size(e2c, 3)

        vgeo = Array{FT, 4}(undef, Nq, Nq, length(VGEO2D), nelem)
        sgeo = Array{FT, 4}(undef, Nq, nface, length(SGEO2D), nelem)

        Metrics.creategrid!(ntuple(j -> (@view vgeo[:, :, j, :]), dim)..., e2c, ξ1)
        x1 = @view vgeo[:, :, VGEO2D.x1, :]
        x2 = @view vgeo[:, :, VGEO2D.x2, :]

        foreach(j -> (x1[j], x2[j]) = f(x1[j], x2[j]), 1:length(x1))

        Metrics.computemetric!(
            ntuple(j -> (@view vgeo[:, :, j, :]), length(VGEO2D))...,
            ntuple(j -> (@view sgeo[:, :, j, :]), length(SGEO2D))...,
            D,
        )

        (Cx1, Cx2) = (zeros(FT, Nq, Nq), zeros(FT, Nq, Nq))

        J = @view vgeo[:, :, VGEO2D.J, :]
        ξ1x1 = @view vgeo[:, :, VGEO2D.ξ1x1, :]
        ξ1x2 = @view vgeo[:, :, VGEO2D.ξ1x2, :]
        ξ2x1 = @view vgeo[:, :, VGEO2D.ξ2x1, :]
        ξ2x2 = @view vgeo[:, :, VGEO2D.ξ2x2, :]

        e = 1
        for n in 1:Nq
            Cx1[:, n] += D * (J[:, n, e] .* ξ1x1[:, n, e])
            Cx1[n, :] += D * (J[n, :, e] .* ξ2x1[n, :, e])

            Cx2[:, n] += D * (J[:, n, e] .* ξ1x2[:, n, e])
            Cx2[n, :] += D * (J[n, :, e] .* ξ2x2[n, :, e])
        end
        @test maximum(abs.(Cx1)) ≤ 1000 * eps(FT)
        @test maximum(abs.(Cx2)) ≤ 1000 * eps(FT)
    end
    #}}}
end

@testset "3-D Metric terms" begin
    # linear test
    #{{{
    for FT in (Float32, Float64)
        let
            N = 2

            ξ1, w = Elements.lglpoints(FT, N)
            D = Elements.spectralderivative(ξ1)
            Nq = N + 1

            dim = 3
            nface = 2dim
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

            x_exact = Array{Int, 4}(undef, 3, 3, 3, nelem)
            x_exact[1, :, :, 1] .= 0
            x_exact[2, :, :, 1] .= 1
            x_exact[3, :, :, 1] .= 2
            x_exact[:, 1, :, 2] .= 2
            x_exact[:, 2, :, 2] .= 1
            x_exact[:, 3, :, 2] .= 0

            ξ1x1_exact = zeros(Int, 3, 3, 3, nelem)
            ξ1x1_exact[:, :, :, 1] .= 1

            ξ1x2_exact = zeros(Int, 3, 3, 3, nelem)
            ξ1x2_exact[:, :, :, 2] .= 1

            ξ2x1_exact = zeros(Int, 3, 3, 3, nelem)
            ξ2x1_exact[:, :, :, 2] .= -1

            ξ2x2_exact = zeros(Int, 3, 3, 3, nelem)
            ξ2x2_exact[:, :, :, 1] .= 1

            ξ3x3_exact = ones(Int, 3, 3, 3, nelem)

            y_exact = Array{Int, 4}(undef, 3, 3, 3, nelem)
            y_exact[:, 1, :, 1] .= 0
            y_exact[:, 2, :, 1] .= 1
            y_exact[:, 3, :, 1] .= 2
            y_exact[1, :, :, 2] .= 0
            y_exact[2, :, :, 2] .= 1
            y_exact[3, :, :, 2] .= 2

            z_exact = Array{Int, 4}(undef, 3, 3, 3, nelem)
            z_exact[:, :, 1, 1:2] .= 0
            z_exact[:, :, 2, 1:2] .= 1
            z_exact[:, :, 3, 1:2] .= 2

            J_exact = ones(Int, 3, 3, 3, nelem)

            sJ_exact = ones(Int, Nq, Nq, nface, nelem)

            nx_exact = zeros(Int, Nq, Nq, nface, nelem)
            nx_exact[:, :, 1, 1] .= -1
            nx_exact[:, :, 2, 1] .= 1
            nx_exact[:, :, 3, 2] .= 1
            nx_exact[:, :, 4, 2] .= -1

            ny_exact = zeros(Int, Nq, Nq, nface, nelem)
            ny_exact[:, :, 3, 1] .= -1
            ny_exact[:, :, 4, 1] .= 1
            ny_exact[:, :, 1, 2] .= -1
            ny_exact[:, :, 2, 2] .= 1

            nz_exact = zeros(Int, Nq, Nq, nface, nelem)
            nz_exact[:, :, 5, 1:2] .= -1
            nz_exact[:, :, 6, 1:2] .= 1

            vgeo = Array{FT, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
            sgeo = Array{FT, 4}(undef, Nq^2, nface, length(SGEO3D), nelem)
            Metrics.creategrid!(
                ntuple(j -> (@view vgeo[:, :, :, j, :]), dim)...,
                e2c,
                ξ1,
            )
            Metrics.computemetric!(
                ntuple(j -> (@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
                ntuple(j -> (@view sgeo[:, :, j, :]), length(SGEO3D))...,
                D,
            )
            sgeo = reshape(sgeo, Nq, Nq, nface, length(SGEO3D), nelem)

            @test (@view vgeo[:, :, :, VGEO3D.x1, :]) ≈ x_exact
            @test (@view vgeo[:, :, :, VGEO3D.x2, :]) ≈ y_exact
            @test (@view vgeo[:, :, :, VGEO3D.x3, :]) ≈ z_exact
            @test (@view vgeo[:, :, :, VGEO3D.J, :]) ≈ J_exact
            @test (@view vgeo[:, :, :, VGEO3D.ξ1x1, :]) ≈ ξ1x1_exact
            @test (@view vgeo[:, :, :, VGEO3D.ξ1x2, :]) ≈ ξ1x2_exact
            @test maximum(abs.(@view vgeo[
                :,
                :,
                :,
                VGEO3D.ξ1x3,
                :,
            ])) ≤ 10 * eps(FT)
            @test (@view vgeo[:, :, :, VGEO3D.ξ2x1, :]) ≈ ξ2x1_exact
            @test (@view vgeo[:, :, :, VGEO3D.ξ2x2, :]) ≈ ξ2x2_exact
            @test maximum(abs.(@view vgeo[
                :,
                :,
                :,
                VGEO3D.ξ2x3,
                :,
            ])) ≤ 10 * eps(FT)
            @test maximum(abs.(@view vgeo[
                :,
                :,
                :,
                VGEO3D.ξ3x1,
                :,
            ])) ≤ 10 * eps(FT)
            @test maximum(abs.(@view vgeo[
                :,
                :,
                :,
                VGEO3D.ξ3x2,
                :,
            ])) ≤ 10 * eps(FT)
            @test (@view vgeo[:, :, :, VGEO3D.ξ3x3, :]) ≈ ξ3x3_exact
            @test (@view sgeo[:, :, :, SGEO3D.sJ, :]) ≈ sJ_exact
            @test (@view sgeo[:, :, :, SGEO3D.n1, :]) ≈ nx_exact
            @test (@view sgeo[:, :, :, SGEO3D.n2, :]) ≈ ny_exact
            @test (@view sgeo[:, :, :, SGEO3D.n3, :]) ≈ nz_exact
        end
    end
    #}}}

    # linear test with rotation
    #{{{
    for FT in (Float32, Float64)
        θ1 = 2 * FT(π) * FT(0.9)
        θ2 = 2 * FT(π) * FT(-0.56)
        θ3 = 2 * FT(π) * FT(0.33)
        #=
        θ1 = 2 * FT(π) * rand(FT)
        θ2 = 2 * FT(π) * rand(FT)
        θ3 = 2 * FT(π) * rand(FT)
        =#
        #=
        θ1 = FT(π) / 6
        θ2 = FT(π) / 12
        θ3 = 4 * FT(π) / 5
        =#
        Q = [cos(θ1) -sin(θ1) 0; sin(θ1) cos(θ1) 0; 0 0 1]
        Q *= [cos(θ2) 0 -sin(θ2); 0 1 0; sin(θ2) 0 cos(θ2)]
        Q *= [1 0 0; 0 cos(θ3) -sin(θ3); 0 sin(θ3) cos(θ3)]
        let
            N = 2

            ξ1, w = Elements.lglpoints(FT, N)
            D = Elements.spectralderivative(ξ1)
            Nq = N + 1

            dim = 3
            nface = 2dim
            e2c = Array{FT, 3}(undef, dim, 8, 1)
            e2c[:, :, 1] = [
                0 2 0 2 0 2 0 2
                0 0 2 2 0 0 2 2
                0 0 0 0 2 2 2 2
            ]
            @views (x1, x2, x3) = (e2c[1, :, 1], e2c[2, :, 1], e2c[3, :, 1])
            for i in 1:length(x1)
                (x1[i], x2[i], x3[i]) = Q * [x1[i]; x2[i]; x3[i]]
            end

            nelem = size(e2c, 3)

            x1e = Array{FT, 4}(undef, 3, 3, 3, nelem)
            x1e[1, :, :, 1] .= 0
            x1e[2, :, :, 1] .= 1
            x1e[3, :, :, 1] .= 2

            x2e = Array{FT, 4}(undef, 3, 3, 3, nelem)
            x2e[:, 1, :, 1] .= 0
            x2e[:, 2, :, 1] .= 1
            x2e[:, 3, :, 1] .= 2

            x3e = Array{FT, 4}(undef, 3, 3, 3, nelem)
            x3e[:, :, 1, 1] .= 0
            x3e[:, :, 2, 1] .= 1
            x3e[:, :, 3, 1] .= 2

            for i in 1:length(x1e)
                (x1e[i], x2e[i], x3e[i]) = Q * [x1e[i]; x2e[i]; x3e[i]]
            end

            Je = ones(Int, 3, 3, 3, nelem)

            # By construction
            # Q = [x1ξ1 x1ξ2 x2ξ2; x2ξ1 x2ξ2 x2ξ3; x3ξ1 x3ξ2 x3ξ3] = [ξ1x1 ξ2x1 ξ3x1; ξ1x2 ξ2x2 ξ3x2; ξ1x3 ξ2x3 ξ3x3]
            ξ1x1e = fill(Q[1, 1], 3, 3, 3, nelem)
            ξ1x2e = fill(Q[2, 1], 3, 3, 3, nelem)
            ξ1x3e = fill(Q[3, 1], 3, 3, 3, nelem)
            ξ2x1e = fill(Q[1, 2], 3, 3, 3, nelem)
            ξ2x2e = fill(Q[2, 2], 3, 3, 3, nelem)
            ξ2x3e = fill(Q[3, 2], 3, 3, 3, nelem)
            ξ3x1e = fill(Q[1, 3], 3, 3, 3, nelem)
            ξ3x2e = fill(Q[2, 3], 3, 3, 3, nelem)
            ξ3x3e = fill(Q[3, 3], 3, 3, 3, nelem)

            sJe = ones(Int, Nq, Nq, nface, nelem)

            n1e = zeros(FT, Nq, Nq, nface, nelem)
            n2e = zeros(FT, Nq, Nq, nface, nelem)
            n3e = zeros(FT, Nq, Nq, nface, nelem)

            fill!(@view(n1e[:, :, 1, :]), -Q[1, 1])
            fill!(@view(n1e[:, :, 2, :]), Q[1, 1])
            fill!(@view(n1e[:, :, 3, :]), -Q[1, 2])
            fill!(@view(n1e[:, :, 4, :]), Q[1, 2])
            fill!(@view(n1e[:, :, 5, :]), -Q[1, 3])
            fill!(@view(n1e[:, :, 6, :]), Q[1, 3])
            fill!(@view(n2e[:, :, 1, :]), -Q[2, 1])
            fill!(@view(n2e[:, :, 2, :]), Q[2, 1])
            fill!(@view(n2e[:, :, 3, :]), -Q[2, 2])
            fill!(@view(n2e[:, :, 4, :]), Q[2, 2])
            fill!(@view(n2e[:, :, 5, :]), -Q[2, 3])
            fill!(@view(n2e[:, :, 6, :]), Q[2, 3])
            fill!(@view(n3e[:, :, 1, :]), -Q[3, 1])
            fill!(@view(n3e[:, :, 2, :]), Q[3, 1])
            fill!(@view(n3e[:, :, 3, :]), -Q[3, 2])
            fill!(@view(n3e[:, :, 4, :]), Q[3, 2])
            fill!(@view(n3e[:, :, 5, :]), -Q[3, 3])
            fill!(@view(n3e[:, :, 6, :]), Q[3, 3])

            vgeo = Array{FT, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
            sgeo = Array{FT, 4}(undef, Nq^2, nface, length(SGEO3D), nelem)
            Metrics.creategrid!(
                ntuple(j -> (@view vgeo[:, :, :, j, :]), dim)...,
                e2c,
                ξ1,
            )
            Metrics.computemetric!(
                ntuple(j -> (@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
                ntuple(j -> (@view sgeo[:, :, j, :]), length(SGEO3D))...,
                D,
            )
            sgeo = reshape(sgeo, Nq, Nq, nface, length(SGEO3D), nelem)

            @test (@view vgeo[:, :, :, VGEO3D.x1, :]) ≈ x1e
            @test (@view vgeo[:, :, :, VGEO3D.x2, :]) ≈ x2e
            @test (@view vgeo[:, :, :, VGEO3D.x3, :]) ≈ x3e
            @test (@view vgeo[:, :, :, VGEO3D.J, :]) ≈ Je
            @test (@view vgeo[:, :, :, VGEO3D.ξ1x1, :]) ≈ ξ1x1e
            @test (@view vgeo[:, :, :, VGEO3D.ξ1x2, :]) ≈ ξ1x2e
            @test (@view vgeo[:, :, :, VGEO3D.ξ1x3, :]) ≈ ξ1x3e
            @test (@view vgeo[:, :, :, VGEO3D.ξ2x1, :]) ≈ ξ2x1e
            @test (@view vgeo[:, :, :, VGEO3D.ξ2x2, :]) ≈ ξ2x2e
            @test (@view vgeo[:, :, :, VGEO3D.ξ2x3, :]) ≈ ξ2x3e
            @test (@view vgeo[:, :, :, VGEO3D.ξ3x1, :]) ≈ ξ3x1e
            @test (@view vgeo[:, :, :, VGEO3D.ξ3x2, :]) ≈ ξ3x2e
            @test (@view vgeo[:, :, :, VGEO3D.ξ3x3, :]) ≈ ξ3x3e
            @test (@view sgeo[:, :, :, SGEO3D.sJ, :]) ≈ sJe
            @test (@view sgeo[:, :, :, SGEO3D.n1, :]) ≈ n1e
            @test (@view sgeo[:, :, :, SGEO3D.n2, :]) ≈ n2e
            @test (@view sgeo[:, :, :, SGEO3D.n3, :]) ≈ n3e
        end
    end
    #}}}

    # Polynomial 3-D test
    #{{{
    for FT in (Float32, Float64)
        f(ξ1, ξ2, ξ3) = @.( (
            ξ2 + ξ1 * ξ3 - (ξ1^2 * ξ2^2 * ξ3^2) / 4,
            ξ3 - ((ξ1 * ξ2 * ξ3) / 2 + 1 / 2)^3 + 1,
            ξ1 + (ξ1 / 2 + 1 / 2)^6 * (ξ2 / 2 + 1 / 2)^6 * (ξ3 / 2 + 1 / 2)^6,
        ))

        fx1ξ1(ξ1, ξ2, ξ3) = @.(ξ3 - (ξ1 * ξ2^2 * ξ3^2) / 2)
        fx1ξ2(ξ1, ξ2, ξ3) = @.(1 - (ξ1^2 * ξ2 * ξ3^2) / 2)
        fx1ξ3(ξ1, ξ2, ξ3) = @.(ξ1 - (ξ1^2 * ξ2^2 * ξ3) / 2)
        fx2ξ1(ξ1, ξ2, ξ3) =
            @.(-(3 * ξ2 * ξ3 * ((ξ1 * ξ2 * ξ3) / 2 + 1 / 2)^2) / 2)
        fx2ξ2(ξ1, ξ2, ξ3) =
            @.(-(3 * ξ1 * ξ3 * ((ξ1 * ξ2 * ξ3) / 2 + 1 / 2)^2) / 2)
        fx2ξ3(ξ1, ξ2, ξ3) =
            @.(1 - (3 * ξ1 * ξ2 * ((ξ1 * ξ2 * ξ3) / 2 + 1 / 2)^2) / 2)
        fx3ξ1(ξ1, ξ2, ξ3) = @.(
            3 * (ξ1 / 2 + 1 / 2)^5 * (ξ2 / 2 + 1 / 2)^6 * (ξ3 / 2 + 1 / 2)^6 + 1
        )
        fx3ξ2(ξ1, ξ2, ξ3) =
            @.(3 * (ξ1 / 2 + 1 / 2)^6 * (ξ2 / 2 + 1 / 2)^5 * (ξ3 / 2 + 1 / 2)^6)
        fx3ξ3(ξ1, ξ2, ξ3) =
            @.(3 * (ξ1 / 2 + 1 / 2)^6 * (ξ2 / 2 + 1 / 2)^6 * (ξ3 / 2 + 1 / 2)^5)

        let
            N = 9

            ξ1, w = Elements.lglpoints(FT, N)
            D = Elements.spectralderivative(ξ1)
            Nq = N + 1

            dim = 3
            nface = 2dim
            e2c = Array{FT, 3}(undef, dim, 8, 1)
            e2c[:, :, 1] = [
                -1 1 -1 1 -1 1 -1 1
                -1 -1 1 1 -1 -1 1 1
                -1 -1 -1 -1 1 1 1 1
            ]

            nelem = size(e2c, 3)

            vgeo = Array{FT, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
            sgeo = Array{FT, 4}(undef, Nq^2, nface, length(SGEO3D), nelem)
            Metrics.creategrid!(
                ntuple(j -> (@view vgeo[:, :, :, j, :]), dim)...,
                e2c,
                ξ1,
            )
            x1 = @view vgeo[:, :, :, VGEO3D.x1, :]
            x2 = @view vgeo[:, :, :, VGEO3D.x2, :]
            x3 = @view vgeo[:, :, :, VGEO3D.x3, :]

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

            foreach(
                j -> (x1[j], x2[j], x3[j]) = f(x1[j], x2[j], x3[j]),
                1:length(x1),
            )

            Metrics.computemetric!(
                ntuple(j -> (@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
                ntuple(j -> (@view sgeo[:, :, j, :]), length(SGEO3D))...,
                D,
            )
            sgeo = reshape(sgeo, Nq, Nq, nface, length(SGEO3D), nelem)

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
            n1 = @view sgeo[:, :, :, SGEO3D.n1, :]
            n2 = @view sgeo[:, :, :, SGEO3D.n2, :]
            n3 = @view sgeo[:, :, :, SGEO3D.n3, :]
            sJ = @view sgeo[:, :, :, SGEO3D.sJ, :]
            @test hypot.(n1, n2, n3) ≈ ones(FT, size(n1))
            @test (
                [
                    sJ[:, :, 1, :] .* n1[:, :, 1, :],
                    sJ[:, :, 1, :] .* n2[:, :, 1, :],
                    sJ[:, :, 1, :] .* n3[:, :, 1, :],
                ] ≈ [
                    -J[1, :, :, :] .* ξ1x1[1, :, :, :],
                    -J[1, :, :, :] .* ξ1x2[1, :, :, :],
                    -J[1, :, :, :] .* ξ1x3[1, :, :, :],
                ]
            )
            @test (
                [
                    sJ[:, :, 2, :] .* n1[:, :, 2, :],
                    sJ[:, :, 2, :] .* n2[:, :, 2, :],
                    sJ[:, :, 2, :] .* n3[:, :, 2, :],
                ] ≈ [
                    J[Nq, :, :, :] .* ξ1x1[Nq, :, :, :],
                    J[Nq, :, :, :] .* ξ1x2[Nq, :, :, :],
                    J[Nq, :, :, :] .* ξ1x3[Nq, :, :, :],
                ]
            )
            @test sJ[:, :, 3, :] .* n1[:, :, 3, :] ≈
                  -J[:, 1, :, :] .* ξ2x1[:, 1, :, :]
            @test sJ[:, :, 3, :] .* n2[:, :, 3, :] ≈
                  -J[:, 1, :, :] .* ξ2x2[:, 1, :, :]
            @test sJ[:, :, 3, :] .* n3[:, :, 3, :] ≈
                  -J[:, 1, :, :] .* ξ2x3[:, 1, :, :]
            @test sJ[:, :, 4, :] .* n1[:, :, 4, :] ≈
                  J[:, Nq, :, :] .* ξ2x1[:, Nq, :, :]
            @test sJ[:, :, 4, :] .* n2[:, :, 4, :] ≈
                  J[:, Nq, :, :] .* ξ2x2[:, Nq, :, :]
            @test sJ[:, :, 4, :] .* n3[:, :, 4, :] ≈
                  J[:, Nq, :, :] .* ξ2x3[:, Nq, :, :]
            @test sJ[:, :, 5, :] .* n1[:, :, 5, :] ≈
                  -J[:, :, 1, :] .* ξ3x1[:, :, 1, :]
            @test sJ[:, :, 5, :] .* n2[:, :, 5, :] ≈
                  -J[:, :, 1, :] .* ξ3x2[:, :, 1, :]
            @test sJ[:, :, 5, :] .* n3[:, :, 5, :] ≈
                  -J[:, :, 1, :] .* ξ3x3[:, :, 1, :]
            @test sJ[:, :, 6, :] .* n1[:, :, 6, :] ≈
                  J[:, :, Nq, :] .* ξ3x1[:, :, Nq, :]
            @test sJ[:, :, 6, :] .* n2[:, :, 6, :] ≈
                  J[:, :, Nq, :] .* ξ3x2[:, :, Nq, :]
            @test sJ[:, :, 6, :] .* n3[:, :, 6, :] ≈
                  J[:, :, Nq, :] .* ξ3x3[:, :, Nq, :]
        end
    end
    #}}}

    #{{{
    for FT in (Float32, Float64)
        f(ξ1, ξ2, ξ3) = @.( (
            ξ2 + ξ1 * ξ3 - (ξ1^2 * ξ2^2 * ξ3^2) / 4,
            ξ3 - ((ξ1 * ξ2 * ξ3) / 2 + 1 / 2)^3 + 1,
            ξ1 + (ξ1 / 2 + 1 / 2)^6 * (ξ2 / 2 + 1 / 2)^6 * (ξ3 / 2 + 1 / 2)^6,
        ))

        fx1ξ1(ξ1, ξ2, ξ3) = @.(ξ3 - (ξ1 * ξ2^2 * ξ3^2) / 2)
        fx1ξ2(ξ1, ξ2, ξ3) = @.(1 - (ξ1^2 * ξ2 * ξ3^2) / 2)
        fx1ξ3(ξ1, ξ2, ξ3) = @.(ξ1 - (ξ1^2 * ξ2^2 * ξ3) / 2)
        fx2ξ1(ξ1, ξ2, ξ3) =
            @.(-(3 * ξ2 * ξ3 * ((ξ1 * ξ2 * ξ3) / 2 + 1 / 2)^2) / 2)
        fx2ξ2(ξ1, ξ2, ξ3) =
            @.(-(3 * ξ1 * ξ3 * ((ξ1 * ξ2 * ξ3) / 2 + 1 / 2)^2) / 2)
        fx2ξ3(ξ1, ξ2, ξ3) =
            @.(1 - (3 * ξ1 * ξ2 * ((ξ1 * ξ2 * ξ3) / 2 + 1 / 2)^2) / 2)
        fx3ξ1(ξ1, ξ2, ξ3) = @.(
            3 * (ξ1 / 2 + 1 / 2)^5 * (ξ2 / 2 + 1 / 2)^6 * (ξ3 / 2 + 1 / 2)^6 + 1
        )
        fx3ξ2(ξ1, ξ2, ξ3) =
            @.(3 * (ξ1 / 2 + 1 / 2)^6 * (ξ2 / 2 + 1 / 2)^5 * (ξ3 / 2 + 1 / 2)^6)
        fx3ξ3(ξ1, ξ2, ξ3) =
            @.(3 * (ξ1 / 2 + 1 / 2)^6 * (ξ2 / 2 + 1 / 2)^6 * (ξ3 / 2 + 1 / 2)^5)

        let
            N = 9

            ξ1, w = Elements.lglpoints(FT, N)
            D = Elements.spectralderivative(ξ1)
            Nq = N + 1

            dim = 3
            e2c = Array{FT, 3}(undef, dim, 8, 1)
            e2c[:, :, 1] = [
                -1 1 -1 1 -1 1 -1 1
                -1 -1 1 1 -1 -1 1 1
                -1 -1 -1 -1 1 1 1 1
            ]

            nelem = size(e2c, 3)

            (x1, x2, x3) = Metrics.creategrid3d(e2c, ξ1)

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

            foreach(
                j -> (x1[j], x2[j], x3[j]) = f(x1[j], x2[j], x3[j]),
                1:length(x1),
            )

            metric = Metrics.computemetric(x1, x2, x3, D)

            @test metric.J ≈ J
            @test metric.ξ1x1 ≈ ξ1x1
            @test metric.ξ1x2 ≈ ξ1x2
            @test metric.ξ1x3 ≈ ξ1x3
            @test metric.ξ2x1 ≈ ξ2x1
            @test metric.ξ2x2 ≈ ξ2x2
            @test metric.ξ2x3 ≈ ξ2x3
            @test metric.ξ3x1 ≈ ξ3x1
            @test metric.ξ3x2 ≈ ξ3x2
            @test metric.ξ3x3 ≈ ξ3x3
            ind = LinearIndices((1:Nq, 1:Nq))
            n1 = metric.n1
            n2 = metric.n2
            n3 = metric.n3
            sJ = metric.sJ
            @test hypot.(n1, n2, n3) ≈ ones(FT, size(n1))
            @test (
                [
                    sJ[ind, 1, :] .* n1[ind, 1, :],
                    sJ[ind, 1, :] .* n2[ind, 1, :],
                    sJ[ind, 1, :] .* n3[ind, 1, :],
                ] ≈ [
                    -J[1, :, :, :] .* ξ1x1[1, :, :, :],
                    -J[1, :, :, :] .* ξ1x2[1, :, :, :],
                    -J[1, :, :, :] .* ξ1x3[1, :, :, :],
                ]
            )
            @test (
                [
                    sJ[ind, 2, :] .* n1[ind, 2, :],
                    sJ[ind, 2, :] .* n2[ind, 2, :],
                    sJ[ind, 2, :] .* n3[ind, 2, :],
                ] ≈ [
                    J[Nq, :, :, :] .* ξ1x1[Nq, :, :, :],
                    J[Nq, :, :, :] .* ξ1x2[Nq, :, :, :],
                    J[Nq, :, :, :] .* ξ1x3[Nq, :, :, :],
                ]
            )
            @test sJ[ind, 3, :] .* n1[ind, 3, :] ≈
                  -J[:, 1, :, :] .* ξ2x1[:, 1, :, :]
            @test sJ[ind, 3, :] .* n2[ind, 3, :] ≈
                  -J[:, 1, :, :] .* ξ2x2[:, 1, :, :]
            @test sJ[ind, 3, :] .* n3[ind, 3, :] ≈
                  -J[:, 1, :, :] .* ξ2x3[:, 1, :, :]
            @test sJ[ind, 4, :] .* n1[ind, 4, :] ≈
                  J[:, Nq, :, :] .* ξ2x1[:, Nq, :, :]
            @test sJ[ind, 4, :] .* n2[ind, 4, :] ≈
                  J[:, Nq, :, :] .* ξ2x2[:, Nq, :, :]
            @test sJ[ind, 4, :] .* n3[ind, 4, :] ≈
                  J[:, Nq, :, :] .* ξ2x3[:, Nq, :, :]
            @test sJ[ind, 5, :] .* n1[ind, 5, :] ≈
                  -J[:, :, 1, :] .* ξ3x1[:, :, 1, :]
            @test sJ[ind, 5, :] .* n2[ind, 5, :] ≈
                  -J[:, :, 1, :] .* ξ3x2[:, :, 1, :]
            @test sJ[ind, 5, :] .* n3[ind, 5, :] ≈
                  -J[:, :, 1, :] .* ξ3x3[:, :, 1, :]
            @test sJ[ind, 6, :] .* n1[ind, 6, :] ≈
                  J[:, :, Nq, :] .* ξ3x1[:, :, Nq, :]
            @test sJ[ind, 6, :] .* n2[ind, 6, :] ≈
                  J[:, :, Nq, :] .* ξ3x2[:, :, Nq, :]
            @test sJ[ind, 6, :] .* n3[ind, 6, :] ≈
                  J[:, :, Nq, :] .* ξ3x3[:, :, Nq, :]
        end
    end
    #}}}


    # Constant preserving test
    #{{{
    for FT in (Float32, Float64)
        f(ξ1, ξ2, ξ3) = @.( (
            ξ2 + ξ1 * ξ3 - (ξ1^2 * ξ2^2 * ξ3^2) / 4,
            ξ3 - ((ξ1 * ξ2 * ξ3) / 2 + 1 / 2)^3 + 1,
            ξ1 + (ξ1 / 2 + 1 / 2)^6 * (ξ2 / 2 + 1 / 2)^6 * (ξ3 / 2 + 1 / 2)^6,
        ))
        let
            N = 5

            ξ1, w = Elements.lglpoints(FT, N)
            D = Elements.spectralderivative(ξ1)
            Nq = N + 1

            dim = 3
            nface = 2dim
            e2c = Array{FT, 3}(undef, dim, 8, 1)
            e2c[:, :, 1] = [
                -1 1 -1 1 -1 1 -1 1
                -1 -1 1 1 -1 -1 1 1
                -1 -1 -1 -1 1 1 1 1
            ]

            nelem = size(e2c, 3)

            vgeo = Array{FT, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
            sgeo = Array{FT, 4}(undef, Nq^2, nface, length(SGEO3D), nelem)
            Metrics.creategrid!(
                ntuple(j -> (@view vgeo[:, :, :, j, :]), dim)...,
                e2c,
                ξ1,
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
                D,
            )
            sgeo = reshape(sgeo, Nq, Nq, nface, length(SGEO3D), nelem)

            (Cx1, Cx2, Cx3) = (
                zeros(FT, Nq, Nq, Nq),
                zeros(FT, Nq, Nq, Nq),
                zeros(FT, Nq, Nq, Nq),
            )

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
            for m in 1:Nq
                for n in 1:Nq
                    Cx1[:, n, m] += D * (J[:, n, m, e] .* ξ1x1[:, n, m, e])
                    Cx1[n, :, m] += D * (J[n, :, m, e] .* ξ2x1[n, :, m, e])
                    Cx1[n, m, :] += D * (J[n, m, :, e] .* ξ3x1[n, m, :, e])

                    Cx2[:, n, m] += D * (J[:, n, m, e] .* ξ1x2[:, n, m, e])
                    Cx2[n, :, m] += D * (J[n, :, m, e] .* ξ2x2[n, :, m, e])
                    Cx2[n, m, :] += D * (J[n, m, :, e] .* ξ3x2[n, m, :, e])

                    Cx3[:, n, m] += D * (J[:, n, m, e] .* ξ1x3[:, n, m, e])
                    Cx3[n, :, m] += D * (J[n, :, m, e] .* ξ2x3[n, :, m, e])
                    Cx3[n, m, :] += D * (J[n, m, :, e] .* ξ3x3[n, m, :, e])
                end
            end
            @test maximum(abs.(Cx1)) ≤ 1000 * eps(FT)
            @test maximum(abs.(Cx2)) ≤ 1000 * eps(FT)
            @test maximum(abs.(Cx3)) ≤ 1000 * eps(FT)
        end
    end
    #}}}
end
