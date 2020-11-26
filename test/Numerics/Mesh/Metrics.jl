using ClimateMachine.Mesh.Elements
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Metrics
using LinearAlgebra: I
using Test
using Random: MersenneTwister

const _ξ1x1, _ξ2x1, _ξ3x1 = Grids._ξ1x1, Grids._ξ2x1, Grids._ξ3x1
const _ξ1x2, _ξ2x2, _ξ3x2 = Grids._ξ1x2, Grids._ξ2x2, Grids._ξ3x2
const _ξ1x3, _ξ2x3, _ξ3x3 = Grids._ξ1x3, Grids._ξ2x3, Grids._ξ3x3
const _M, _MI = Grids._M, Grids._MI
const _x1, _x2, _x3 = Grids._x1, Grids._x2, Grids._x3
const _JcV = Grids._JcV
const _nvgeo = Grids._nvgeo

const _n1, _n2, _n3 = Grids._n1, Grids._n2, Grids._n3
const _sM, _vMI = Grids._sM, Grids._vMI
const _nsgeo = Grids._nsgeo

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

            (vgeo, sgeo, _) =
                Grids.computegeometry(e2c, D, ξ, ω, (x...) -> identity(x))
            vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)
            @test vgeo[:, _x1, 1] ≈ (ξ[1] .- 1) / 2
            @test vgeo[:, _x1, 2] ≈ 5 * (ξ[1] .+ 1)

            @test vgeo[:, _M, 1] ≈ ω[1] .* ones(FT, Nq) / 2
            @test vgeo[:, _M, 2] ≈ 5 * ω[1] .* ones(FT, Nq)
            @test vgeo[:, _ξ1x1, 1] ≈ 2 * ones(FT, Nq)
            @test vgeo[:, _ξ1x1, 2] ≈ ones(FT, Nq) / 5
            @test sgeo[_n1, 1, 1, :] ≈ -ones(FT, nelem)
            @test sgeo[_n1, 1, 2, :] ≈ ones(FT, nelem)
            @test sgeo[_sM, 1, 1, :] ≈ ones(FT, nelem)
            @test sgeo[_sM, 1, 2, :] ≈ ones(FT, nelem)
        end
        #}}}
    end

    # N = 0 test
    for FT in (Float32, Float64)
        #{{{
        let
            N = (0,)
            Nq = N .+ 1
            Np = prod(Nq)

            dim = length(N)
            nface = 2dim

            # Create element operators for each polynomial order
            ξω = ntuple(j -> Elements.glpoints(FT, N[j]), dim)
            ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)

            D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

            dim = 1
            e2c = Array{FT, 3}(undef, 1, 2, 2)
            e2c[:, :, 1] = [-1 0]
            e2c[:, :, 2] = [0 10]
            nelem = size(e2c, 3)

            (vgeo, sgeo, _) =
                Grids.computegeometry(e2c, D, ξ, ω, (x...) -> identity(x))
            vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)
            @test vgeo[1, _x1, 1] ≈ sum(e2c[:, :, 1]) / 2
            @test vgeo[1, _x1, 2] ≈ sum(e2c[:, :, 2]) / 2

            @test vgeo[:, _M, 1] ≈ ω[1] .* ones(FT, Nq) / 2
            @test vgeo[:, _M, 2] ≈ 5 * ω[1] .* ones(FT, Nq)
            @test vgeo[:, _ξ1x1, 1] ≈ 2 * ones(FT, Nq)
            @test vgeo[:, _ξ1x1, 2] ≈ ones(FT, Nq) / 5

            @test sgeo[_n1, 1, 1, :] ≈ -ones(FT, nelem)
            @test sgeo[_n1, 1, 2, :] ≈ ones(FT, nelem)
            @test sgeo[_sM, 1, 1, :] ≈ ones(FT, nelem)
            @test sgeo[_sM, 1, 2, :] ≈ ones(FT, nelem)
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

            M_exact =
                ones(FT, Nq..., 4) .* reshape(kron(reverse(ω)...), Nq..., 1)

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

            sM_exact = fill(FT(NaN), maximum(Nfp), nface, nelem)
            sM_exact[1:Nfp[1], 1, :] .= 1 .* ω[2]
            sM_exact[1:Nfp[1], 2, :] .= 1 .* ω[2]
            sM_exact[1:Nfp[2], 3, :] .= 1 .* ω[1]
            sM_exact[1:Nfp[2], 4, :] .= 1 .* ω[1]

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

            (vgeo, sgeo, _) =
                Grids.computegeometry(e2c, D, ξ, ω, (x...) -> identity(x))
            vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)

            @test (@view vgeo[:, :, _x1, :]) ≈ x_exact
            @test (@view vgeo[:, :, _x2, :]) ≈ y_exact
            @test (@view vgeo[:, :, _M, :]) ≈ M_exact
            @test (@view vgeo[:, :, _ξ1x1, :]) ≈ ξ1x1_exact
            @test (@view vgeo[:, :, _ξ1x2, :]) ≈ ξ1x2_exact
            @test (@view vgeo[:, :, _ξ2x1, :]) ≈ ξ2x1_exact
            @test (@view vgeo[:, :, _ξ2x2, :]) ≈ ξ2x2_exact
            msk = isfinite.(sM_exact)
            @test sgeo[_sM, :, :, :][msk] ≈ sM_exact[msk]
            @test sgeo[_n1, :, :, :][msk] ≈ nx_exact[msk]
            @test sgeo[_n2, :, :, :][msk] ≈ ny_exact[msk]

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

            # Create the metrics
            (x1ξ1, x1ξ2, x2ξ1, x2ξ2) = let
                (vgeo, _) = Grids.computegeometry(
                    e2c,
                    D,
                    ξ,
                    ω,
                    (x...) -> identity(x),
                )
                vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)
                ξ1, ξ2 = vgeo[:, :, _x1, :], vgeo[:, :, _x2, :]
                (fx1ξ1(ξ1, ξ2), fx1ξ2(ξ1, ξ2), fx2ξ1(ξ1, ξ2), fx2ξ2(ξ1, ξ2))
            end
            J = (x1ξ1 .* x2ξ2 - x1ξ2 .* x2ξ1)
            M = J .* reshape(kron(reverse(ω)...), Nq..., 1)

            meshwarp(ξ1, ξ2, _) = (f(ξ1, ξ2)..., 0)
            (vgeo, sgeo, _) = Grids.computegeometry(e2c, D, ξ, ω, meshwarp)
            vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)
            x1 = @view vgeo[:, :, _x1, :]
            x2 = @view vgeo[:, :, _x2, :]

            @test M ≈ (@view vgeo[:, :, _M, :])
            @test (@view vgeo[:, :, _ξ1x1, :]) ≈ x2ξ2 ./ J
            @test (@view vgeo[:, :, _ξ2x1, :]) ≈ -x2ξ1 ./ J
            @test (@view vgeo[:, :, _ξ1x2, :]) ≈ -x1ξ2 ./ J
            @test (@view vgeo[:, :, _ξ2x2, :]) ≈ x1ξ1 ./ J

            # check the normals?
            sM = @view sgeo[_sM, :, :, :]
            n1 = @view sgeo[_n1, :, :, :]
            n2 = @view sgeo[_n2, :, :, :]
            @test all(hypot.(n1[1:Nfp[1], 1:2, :], n2[1:Nfp[1], 1:2, :]) .≈ 1)
            @test all(hypot.(n1[1:Nfp[2], 3:4, :], n2[1:Nfp[2], 3:4, :]) .≈ 1)
            @test sM[1:Nfp[1], 1, :] .* n1[1:Nfp[1], 1, :] ≈
                  -x2ξ2[1, :, :] .* ω[2]
            @test sM[1:Nfp[1], 1, :] .* n2[1:Nfp[1], 1, :] ≈
                  x1ξ2[1, :, :] .* ω[2]
            @test sM[1:Nfp[1], 2, :] .* n1[1:Nfp[1], 2, :] ≈
                  x2ξ2[Nq[1], :, :] .* ω[2]
            @test sM[1:Nfp[1], 2, :] .* n2[1:Nfp[1], 2, :] ≈
                  -x1ξ2[Nq[1], :, :] .* ω[2]
            @test sM[1:Nfp[2], 3, :] .* n1[1:Nfp[2], 3, :] ≈
                  x2ξ1[:, 1, :] .* ω[1]
            @test sM[1:Nfp[2], 3, :] .* n2[1:Nfp[2], 3, :] ≈
                  -x1ξ1[:, 1, :] .* ω[1]
            @test sM[1:Nfp[2], 4, :] .* n1[1:Nfp[2], 4, :] ≈
                  -x2ξ1[:, Nq[2], :] .* ω[1]
            @test sM[1:Nfp[2], 4, :] .* n2[1:Nfp[2], 4, :] ≈
                  x1ξ1[:, Nq[2], :] .* ω[1]
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

            meshwarp(ξ1, ξ2, _) = (f(ξ1, ξ2)..., 0)
            (vgeo, sgeo, _) = Grids.computegeometry(e2c, D, ξ, ω, meshwarp)
            vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)
            x1 = @view vgeo[:, :, _x1, :]
            x2 = @view vgeo[:, :, _x2, :]

            (Cx1, Cx2) = (zeros(FT, Nq...), zeros(FT, Nq...))

            J = vgeo[:, :, _M, :] ./ reshape(kron(reverse(ω)...), Nq..., 1)
            ξ1x1 = @view vgeo[:, :, _ξ1x1, :]
            ξ1x2 = @view vgeo[:, :, _ξ1x2, :]
            ξ2x1 = @view vgeo[:, :, _ξ2x1, :]
            ξ2x2 = @view vgeo[:, :, _ξ2x2, :]

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

    #N = 0 test
    #{{{
    let
        for FT in (Float32, Float64)
            N = (2, 0)
            Nq = N .+ 1
            Np = prod(Nq)
            Nfp = div.(Np, Nq)

            dim = length(N)
            nface = 2dim

            # Create element operators for each polynomial order
            ξω = ntuple(
                j ->
                    Nq[j] == 1 ? Elements.glpoints(FT, N[j]) :
                    Elements.lglpoints(FT, N[j]),
                dim,
            )
            ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)
            D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

            fx1(ξ1, ξ2) = ξ1 + (1 + ξ1)^2 * ξ2 / 10
            fx1ξ1(ξ1, ξ2) = 1 + 2(1 + ξ1) * ξ2 / 10
            fx1ξ2(ξ1, ξ2) = (1 + ξ1)^2 / 10
            fx2(ξ1, ξ2) = ξ2 - (1 + ξ1)^2
            fx2ξ1(ξ1, ξ2) = -2(1 + ξ1)
            fx2ξ2(ξ1, ξ2) = 1

            e2c = Array{FT, 3}(undef, 2, 4, 1)
            e2c[:, :, 1] = [-1 1 -1 1; -1 -1 1 1]
            nelem = size(e2c, 3)

            # Create the metrics
            (x1, x2, x1ξ1, x1ξ2, x2ξ1, x2ξ2) = let
                ξ1 = zeros(FT, Np, nelem)
                ξ2 = zeros(FT, Np, nelem)
                Metrics.creategrid!(ξ1, ξ2, e2c, ξ...)
                (
                    fx1.(ξ1, ξ2),
                    fx2.(ξ1, ξ2),
                    fx1ξ1.(ξ1, ξ2),
                    fx1ξ2.(ξ1, ξ2),
                    fx2ξ1.(ξ1, ξ2),
                    fx2ξ2.(ξ1, ξ2),
                )
            end
            J = (x1ξ1 .* x2ξ2 - x1ξ2 .* x2ξ1)

            M = J .* reshape(kron(reverse(ω)...), Nq..., 1)

            meshwarp(ξ1, ξ2, _) = (fx1(ξ1, ξ2), fx2(ξ1, ξ2), 0)
            (vgeo, sgeo, _) = Grids.computegeometry(e2c, D, ξ, ω, meshwarp)
            vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)
            @test x1 ≈ vgeo[:, :, _x1, :]
            @test x2 ≈ vgeo[:, :, _x2, :]

            @test M ≈ vgeo[:, :, _M, :]
            @test (@view vgeo[:, :, _ξ2x1, :]) .* J ≈ -x2ξ1
            @test (@view vgeo[:, :, _ξ2x2, :]) .* J ≈ x1ξ1
            @test (@view vgeo[:, :, _ξ1x1, :]) .* J ≈ x2ξ2
            @test (@view vgeo[:, :, _ξ1x2, :]) .* J ≈ -x1ξ2

            # check the normals?
            sM = @view sgeo[_sM, :, :, :]
            n1 = @view sgeo[_n1, :, :, :]
            n2 = @view sgeo[_n2, :, :, :]
            @test all(hypot.(n1[1:Nfp[1], 1:2, :], n2[1:Nfp[1], 1:2, :]) .≈ 1)
            @test all(hypot.(n1[1:Nfp[2], 3:4, :], n2[1:Nfp[2], 3:4, :]) .≈ 1)
            @test sM[1:Nfp[1], 1, :] .* n1[1:Nfp[1], 1, :] ≈
                  -x2ξ2[1, :, :] .* ω[2]
            @test sM[1:Nfp[1], 1, :] .* n2[1:Nfp[1], 1, :] ≈
                  x1ξ2[1, :, :] .* ω[2]
            @test sM[1:Nfp[1], 2, :] .* n1[1:Nfp[1], 2, :] ≈
                  x2ξ2[Nq[1], :, :] .* ω[2]
            @test sM[1:Nfp[1], 2, :] .* n2[1:Nfp[1], 2, :] ≈
                  -x1ξ2[Nq[1], :, :] .* ω[2]

            # for these faces we need the N = 1 metrics
            (x1ξ1, x2ξ1) = let
                @assert Nq[2] == 1 && Nq[1] != 1
                Nq_N1 = max.(2, Nq)
                ξ1 = zeros(FT, Nq_N1..., nelem)
                ξ2 = zeros(FT, Nq_N1..., nelem)
                Metrics.creategrid!(
                    ξ1,
                    ξ2,
                    e2c,
                    ξ[1],
                    Elements.lglpoints(FT, 1)[1],
                )
                (fx1ξ1.(ξ1, ξ2), fx2ξ1.(ξ1, ξ2))
            end
            @test sM[1:Nfp[2], 3, :] .* n1[1:Nfp[2], 3, :] ≈
                  x2ξ1[:, 1, :] .* ω[1]
            @test sM[1:Nfp[2], 3, :] .* n2[1:Nfp[2], 3, :] ≈
                  -x1ξ1[:, 1, :] .* ω[1]
            @test sM[1:Nfp[2], 4, :] .* n1[1:Nfp[2], 4, :] ≈
                  -x2ξ1[:, 2, :] .* ω[1]
            @test sM[1:Nfp[2], 4, :] .* n2[1:Nfp[2], 4, :] ≈
                  x1ξ1[:, 2, :] .* ω[1]
        end
    end
    #}}}

    # Constant preserving test for N = 0
    #{{{
    let
        for FT in (Float32, Float64), N in ((4, 0), (0, 4))
            Nq = N .+ 1
            Np = prod(Nq)
            Nfp = div.(Np, Nq)

            dim = length(N)
            nface = 2dim

            # Create element operators for each polynomial order
            ξω = ntuple(
                j ->
                    Nq[j] == 1 ? Elements.glpoints(FT, N[j]) :
                    Elements.lglpoints(FT, N[j]),
                dim,
            )
            ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)
            D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

            rng = MersenneTwister(777)
            fx1(ξ1, ξ2) = ξ1 + (ξ1 * ξ2 * rand(rng) + rand(rng)) / 10
            fx2(ξ1, ξ2) = ξ2 + (ξ1 * ξ2 * rand(rng) + rand(rng)) / 10

            e2c = Array{FT, 3}(undef, 2, 4, 1)
            e2c[:, :, 1] = [-1 1 -1 1; -1 -1 1 1]
            nelem = size(e2c, 3)

            meshwarp(ξ1, ξ2, _) = (fx1(ξ1, ξ2), fx2(ξ1, ξ2), 0)
            (vgeo, sgeo, _) = Grids.computegeometry(e2c, D, ξ, ω, meshwarp)

            M = vgeo[:, _M, :]
            ξ1x1 = vgeo[:, _ξ1x1, :]
            ξ2x1 = vgeo[:, _ξ2x1, :]
            ξ1x2 = vgeo[:, _ξ1x2, :]
            ξ2x2 = vgeo[:, _ξ2x2, :]

            I1 = Matrix(I, Nq[1], Nq[1])
            I2 = Matrix(I, Nq[2], Nq[2])
            D1 = kron(I2, D[1])
            D2 = kron(D[2], I1)

            # Face interpolation operators
            L = (
                kron(I2, I1[1, :]'),
                kron(I2, I1[Nq[1], :]'),
                kron(I2[1, :]', I1),
                kron(I2[Nq[2], :]', I1),
            )
            sM = ntuple(f -> sgeo[_sM, 1:Nfp[cld(f, 2)], f, :], nface)
            n1 = ntuple(f -> sgeo[_n1, 1:Nfp[cld(f, 2)], f, :], nface)
            n2 = ntuple(f -> sgeo[_n2, 1:Nfp[cld(f, 2)], f, :], nface)

            # If constant preserving then:
            #   \sum_{j} = D' * M * ξjxk = \sum_{f} L_f' * sM_f * n1_f
            @test D1' * (M .* ξ1x1) + D2' * (M .* ξ2x1) ≈
                  mapreduce((L, sM, n1) -> L' * (sM .* n1), +, L, sM, n1)
            @test D1' * (M .* ξ1x2) + D2' * (M .* ξ2x2) ≈
                  mapreduce((L, sM, n2) -> L' * (sM .* n2), +, L, sM, n2)
        end
    end
    #}}}
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

        M_exact =
            ones(Int, Nq..., nelem) .* reshape(kron(reverse(ω)...), Nq..., 1)

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

        (vgeo, sgeo, _) =
            Grids.computegeometry(e2c, D, ξ, ω, (x...) -> identity(x))
        vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)

        @test (@view vgeo[:, :, :, _x1, :]) ≈ x_exact
        @test (@view vgeo[:, :, :, _x2, :]) ≈ y_exact
        @test (@view vgeo[:, :, :, _x3, :]) ≈ z_exact
        @test (@view vgeo[:, :, :, _M, :]) ≈ M_exact
        @test (@view vgeo[:, :, :, _ξ1x1, :]) ≈ ξ1x1_exact
        @test (@view vgeo[:, :, :, _ξ1x2, :]) ≈ ξ1x2_exact
        @test maximum(abs.(@view vgeo[:, :, :, _ξ1x3, :])) ≤ 100 * eps(FT)
        @test (@view vgeo[:, :, :, _ξ2x1, :]) ≈ ξ2x1_exact
        @test (@view vgeo[:, :, :, _ξ2x2, :]) ≈ ξ2x2_exact
        @test maximum(abs.(@view vgeo[:, :, :, _ξ2x3, :])) ≤ 100 * eps(FT)
        @test maximum(abs.(@view vgeo[:, :, :, _ξ3x1, :])) ≤ 100 * eps(FT)
        @test maximum(abs.(@view vgeo[:, :, :, _ξ3x2, :])) ≤ 100 * eps(FT)
        @test (@view vgeo[:, :, :, _ξ3x3, :]) ≈ ξ3x3_exact
        for d in 1:dim
            for f in (2d - 1):(2d)
                ωf = ntuple(j -> ω[mod1(d + j, dim)], dim - 1)
                if !(dim == 3 && d == 2)
                    ωf = reverse(ωf)
                end
                Mf = kron(1, ωf...)
                @test isapprox(
                    (@view sgeo[_sM, 1:Nfp[d], f, :]),
                    sJ_exact[1:Nfp[d], f, :] .* Mf,
                    atol = √eps(FT),
                    rtol = √eps(FT),
                )
                @test isapprox(
                    (@view sgeo[_n1, 1:Nfp[d], f, :]),
                    nx_exact[1:Nfp[d], f, :];
                    atol = √eps(FT),
                    rtol = √eps(FT),
                )
                @test isapprox(
                    (@view sgeo[_n2, 1:Nfp[d], f, :]),
                    ny_exact[1:Nfp[d], f, :];
                    atol = √eps(FT),
                    rtol = √eps(FT),
                )
                @test isapprox(
                    (@view sgeo[_n3, 1:Nfp[d], f, :]),
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

        # Compute exact metrics
        (x1ξ1, x1ξ2, x1ξ3, x2ξ1, x2ξ2, x2ξ3, x3ξ1, x3ξ2, x3ξ3) = let
            (vgeo, _) =
                Grids.computegeometry(e2c, D, ξ, ω, (x...) -> identity(x))
            vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)
            ξ1 = vgeo[:, :, :, _x1, :]
            ξ2 = vgeo[:, :, :, _x2, :]
            ξ3 = vgeo[:, :, :, _x3, :]
            (
                fx1ξ1(ξ1, ξ2, ξ3),
                fx1ξ2(ξ1, ξ2, ξ3),
                fx1ξ3(ξ1, ξ2, ξ3),
                fx2ξ1(ξ1, ξ2, ξ3),
                fx2ξ2(ξ1, ξ2, ξ3),
                fx2ξ3(ξ1, ξ2, ξ3),
                fx3ξ1(ξ1, ξ2, ξ3),
                fx3ξ2(ξ1, ξ2, ξ3),
                fx3ξ3(ξ1, ξ2, ξ3),
            )
        end
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

        (vgeo, sgeo, _) = Grids.computegeometry(e2c, D, ξ, ω, f)
        vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)

        @test (@view vgeo[:, :, :, _M, :]) ≈
              J .* reshape(kron(reverse(ω)...), Nq..., 1)
        @test (@view vgeo[:, :, :, _ξ1x1, :]) ≈ ξ1x1
        @test (@view vgeo[:, :, :, _ξ1x2, :]) ≈ ξ1x2
        @test (@view vgeo[:, :, :, _ξ1x3, :]) ≈ ξ1x3
        @test (@view vgeo[:, :, :, _ξ2x1, :]) ≈ ξ2x1
        @test (@view vgeo[:, :, :, _ξ2x2, :]) ≈ ξ2x2
        @test (@view vgeo[:, :, :, _ξ2x3, :]) ≈ ξ2x3
        @test (@view vgeo[:, :, :, _ξ3x1, :]) ≈ ξ3x1
        @test (@view vgeo[:, :, :, _ξ3x2, :]) ≈ ξ3x2
        @test (@view vgeo[:, :, :, _ξ3x3, :]) ≈ ξ3x3
        n1 = @view sgeo[_n1, :, :, :]
        n2 = @view sgeo[_n2, :, :, :]
        n3 = @view sgeo[_n3, :, :, :]
        sM = @view sgeo[_sM, :, :, :]
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
        Mf = kron(1, ω[3], ω[2])
        @test [
            (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (-J[1, :, :, :] .* ξ1x1[1, :, :, :])[:] .* Mf,
            (-J[1, :, :, :] .* ξ1x2[1, :, :, :])[:] .* Mf,
            (-J[1, :, :, :] .* ξ1x3[1, :, :, :])[:] .* Mf,
        ]
        d, f = 1, 2
        Mf = kron(1, ω[3], ω[2])
        @test [
            (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (J[Nq[d], :, :, :] .* ξ1x1[Nq[d], :, :, :])[:] .* Mf,
            (J[Nq[d], :, :, :] .* ξ1x2[Nq[d], :, :, :])[:] .* Mf,
            (J[Nq[d], :, :, :] .* ξ1x3[Nq[d], :, :, :])[:] .* Mf,
        ]
        d, f = 2, 3
        Mf = kron(1, ω[3], ω[1])
        @test [
            (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (-J[:, 1, :, :] .* ξ2x1[:, 1, :, :])[:] .* Mf,
            (-J[:, 1, :, :] .* ξ2x2[:, 1, :, :])[:] .* Mf,
            (-J[:, 1, :, :] .* ξ2x3[:, 1, :, :])[:] .* Mf,
        ]
        d, f = 2, 4
        Mf = kron(1, ω[3], ω[1])
        @test [
            (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (J[:, Nq[d], :, :] .* ξ2x1[:, Nq[d], :, :])[:] .* Mf,
            (J[:, Nq[d], :, :] .* ξ2x2[:, Nq[d], :, :])[:] .* Mf,
            (J[:, Nq[d], :, :] .* ξ2x3[:, Nq[d], :, :])[:] .* Mf,
        ]
        d, f = 3, 5
        Mf = kron(1, ω[2], ω[1])
        @test [
            (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (-J[:, :, 1, :] .* ξ3x1[:, :, 1, :])[:] .* Mf,
            (-J[:, :, 1, :] .* ξ3x2[:, :, 1, :])[:] .* Mf,
            (-J[:, :, 1, :] .* ξ3x3[:, :, 1, :])[:] .* Mf,
        ]
        d, f = 3, 6
        Mf = kron(1, ω[2], ω[1])
        @test [
            (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
            (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
        ] ≈ [
            (J[:, :, Nq[d], :] .* ξ3x1[:, :, Nq[d], :])[:] .* Mf,
            (J[:, :, Nq[d], :] .* ξ3x2[:, :, Nq[d], :])[:] .* Mf,
            (J[:, :, Nq[d], :] .* ξ3x3[:, :, Nq[d], :])[:] .* Mf,
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

        (vgeo, sgeo, _) = Grids.computegeometry(e2c, D, ξ, ω, f)
        vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)

        (Cx1, Cx2, Cx3) = (zeros(FT, Nq...), zeros(FT, Nq...), zeros(FT, Nq...))

        J =
            (@view vgeo[:, :, :, _M, :]) ./
            reshape(kron(reverse(ω)...), Nq..., 1)
        ξ1x1 = @view vgeo[:, :, :, _ξ1x1, :]
        ξ1x2 = @view vgeo[:, :, :, _ξ1x2, :]
        ξ1x3 = @view vgeo[:, :, :, _ξ1x3, :]
        ξ2x1 = @view vgeo[:, :, :, _ξ2x1, :]
        ξ2x2 = @view vgeo[:, :, :, _ξ2x2, :]
        ξ2x3 = @view vgeo[:, :, :, _ξ2x3, :]
        ξ3x1 = @view vgeo[:, :, :, _ξ3x1, :]
        ξ3x2 = @view vgeo[:, :, :, _ξ3x2, :]
        ξ3x3 = @view vgeo[:, :, :, _ξ3x3, :]

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

    #N = 0 test
    #{{{
    let
        for FT in (Float32, Float64)
            N = (4, 4, 0)
            Nq = N .+ 1
            Np = prod(Nq)
            Nfp = div.(Np, Nq)

            dim = length(N)
            nface = 2dim

            # Create element operators for each polynomial order
            ξω = ntuple(
                j ->
                    Nq[j] == 1 ? Elements.glpoints(FT, N[j]) :
                    Elements.lglpoints(FT, N[j]),
                dim,
            )
            ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)
            D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

            fx1(ξ1, ξ2, ξ3) = ξ1 + (1 + ξ1)^2 * (1 + ξ2)^2 + ξ3 / 10
            fx1ξ1(ξ1, ξ2, ξ3) = 1 + 2(1 + ξ1) * (1 + ξ2)^2
            fx1ξ2(ξ1, ξ2, ξ3) = (1 + ξ1)^2 * 2(1 + ξ2)
            fx1ξ3(ξ1, ξ2, ξ3) = 1 / 10

            fx2(ξ1, ξ2, ξ3) = ξ2 - (1 + ξ1)^2 + (2 + ξ3) / 2
            fx2ξ1(ξ1, ξ2, ξ3) = -2(1 + ξ1)
            fx2ξ2(ξ1, ξ2, ξ3) = 1
            fx2ξ3(ξ1, ξ2, ξ3) = 1 / 2

            fx3(ξ1, ξ2, ξ3) = ξ3 + (1 + ξ1)^2 * (1 + ξ2)^2 / 10
            fx3ξ1(ξ1, ξ2, ξ3) = 2(1 + ξ1) * (1 + ξ2)^2 / 10
            fx3ξ2(ξ1, ξ2, ξ3) = (1 + ξ1)^2 * 2(1 + ξ2) / 10
            fx3ξ3(ξ1, ξ2, ξ3) = 1

            e2c = Array{FT, 3}(undef, 3, 8, 1)
            e2c[:, :, 1] = [
                -1 +1 -1 +1 -1 +1 -1 +1
                -1 -1 +1 +1 -1 -1 +1 +1
                -1 -1 -1 -1 +1 +1 +1 +1
            ]
            nelem = size(e2c, 3)

            # Create the metrics
            (x1, x2, x3, x1ξ1, x1ξ2, x1ξ3, x2ξ1, x2ξ2, x2ξ3, x3ξ1, x3ξ2, x3ξ3) =
                let
                    ξ1 = zeros(FT, Nq..., nelem)
                    ξ2 = zeros(FT, Nq..., nelem)
                    ξ3 = zeros(FT, Nq..., nelem)
                    Metrics.creategrid!(ξ1, ξ2, ξ3, e2c, ξ...)
                    (
                        fx1.(ξ1, ξ2, ξ3),
                        fx2.(ξ1, ξ2, ξ3),
                        fx3.(ξ1, ξ2, ξ3),
                        fx1ξ1.(ξ1, ξ2, ξ3),
                        fx1ξ2.(ξ1, ξ2, ξ3),
                        fx1ξ3.(ξ1, ξ2, ξ3),
                        fx2ξ1.(ξ1, ξ2, ξ3),
                        fx2ξ2.(ξ1, ξ2, ξ3),
                        fx2ξ3.(ξ1, ξ2, ξ3),
                        fx3ξ1.(ξ1, ξ2, ξ3),
                        fx3ξ2.(ξ1, ξ2, ξ3),
                        fx3ξ3.(ξ1, ξ2, ξ3),
                    )
                end
            J = @.(
                x1ξ1 * (x2ξ2 * x3ξ3 - x3ξ2 * x2ξ3) +
                x2ξ1 * (x3ξ2 * x1ξ3 - x1ξ2 * x3ξ3) +
                x3ξ1 * (x1ξ2 * x2ξ3 - x2ξ2 * x1ξ3)
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

            M = J .* reshape(kron(reverse(ω)...), Nq..., 1)

            meshwarp(ξ1, ξ2, ξ3) =
                (fx1(ξ1, ξ2, ξ3), fx2(ξ1, ξ2, ξ3), fx3(ξ1, ξ2, ξ3))
            (vgeo, sgeo, _) = Grids.computegeometry(e2c, D, ξ, ω, meshwarp)
            vgeo = reshape(vgeo, Nq..., _nvgeo, nelem)

            @test x1 ≈ vgeo[:, :, :, _x1, :]
            @test x2 ≈ vgeo[:, :, :, _x2, :]
            @test x3 ≈ vgeo[:, :, :, _x3, :]

            @test M ≈ vgeo[:, :, :, _M, :]

            @test (@view vgeo[:, :, :, _ξ1x1, :]) ≈ ξ1x1
            @test (@view vgeo[:, :, :, _ξ1x2, :]) ≈ ξ1x2
            @test (@view vgeo[:, :, :, _ξ1x3, :]) ≈ ξ1x3

            @test (@view vgeo[:, :, :, _ξ2x1, :]) ≈ ξ2x1
            @test (@view vgeo[:, :, :, _ξ2x2, :]) ≈ ξ2x2
            @test (@view vgeo[:, :, :, _ξ2x3, :]) ≈ ξ2x3

            @test (@view vgeo[:, :, :, _ξ3x1, :]) ≈ ξ3x1
            @test (@view vgeo[:, :, :, _ξ3x2, :]) ≈ ξ3x2
            @test (@view vgeo[:, :, :, _ξ3x3, :]) ≈ ξ3x3

            # check the normals?
            sM = @view sgeo[_sM, :, :, :]
            n1 = @view sgeo[_n1, :, :, :]
            n2 = @view sgeo[_n2, :, :, :]
            n3 = @view sgeo[_n3, :, :, :]
            @test all(
                hypot.(
                    n1[1:Nfp[1], 1:2, :],
                    n2[1:Nfp[1], 1:2, :],
                    n3[1:Nfp[1], 1:2, :],
                ) .≈ 1,
            )
            @test all(
                hypot.(
                    n1[1:Nfp[2], 3:4, :],
                    n2[1:Nfp[2], 3:4, :],
                    n3[1:Nfp[2], 3:4, :],
                ) .≈ 1,
            )
            @test all(
                hypot.(
                    n1[1:Nfp[3], 5:6, :],
                    n2[1:Nfp[3], 5:6, :],
                    n3[1:Nfp[3], 5:6, :],
                ) .≈ 1,
            )

            d, f = 1, 1
            Mf = kron(1, ω[3], ω[2])
            @test [
                (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
            ] ≈ [
                (-J[1, :, :, :] .* ξ1x1[1, :, :, :])[:] .* Mf,
                (-J[1, :, :, :] .* ξ1x2[1, :, :, :])[:] .* Mf,
                (-J[1, :, :, :] .* ξ1x3[1, :, :, :])[:] .* Mf,
            ]
            d, f = 1, 2
            Mf = kron(1, ω[3], ω[2])
            @test [
                (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
            ] ≈ [
                (J[Nq[d], :, :, :] .* ξ1x1[Nq[d], :, :, :])[:] .* Mf,
                (J[Nq[d], :, :, :] .* ξ1x2[Nq[d], :, :, :])[:] .* Mf,
                (J[Nq[d], :, :, :] .* ξ1x3[Nq[d], :, :, :])[:] .* Mf,
            ]
            d, f = 2, 3
            Mf = kron(1, ω[3], ω[1])
            @test [
                (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
            ] ≈ [
                (-J[:, 1, :, :] .* ξ2x1[:, 1, :, :])[:] .* Mf,
                (-J[:, 1, :, :] .* ξ2x2[:, 1, :, :])[:] .* Mf,
                (-J[:, 1, :, :] .* ξ2x3[:, 1, :, :])[:] .* Mf,
            ]
            d, f = 2, 4
            Mf = kron(1, ω[3], ω[1])
            @test [
                (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
            ] ≈ [
                (J[:, Nq[d], :, :] .* ξ2x1[:, Nq[d], :, :])[:] .* Mf,
                (J[:, Nq[d], :, :] .* ξ2x2[:, Nq[d], :, :])[:] .* Mf,
                (J[:, Nq[d], :, :] .* ξ2x3[:, Nq[d], :, :])[:] .* Mf,
            ]

            # for these faces we need the N = 1 metrics
            (x1ξ1, x1ξ2, x1ξ3, x2ξ1, x2ξ2, x2ξ3, x3ξ1, x3ξ2, x3ξ3) = let
                @assert Nq[1] != 1 && Nq[2] != 1 && Nq[3] == 1
                Nq_N1 = max.(2, Nq)
                ξ1 = zeros(FT, Nq_N1..., nelem)
                ξ2 = zeros(FT, Nq_N1..., nelem)
                ξ3 = zeros(FT, Nq_N1..., nelem)
                Metrics.creategrid!(
                    ξ1,
                    ξ2,
                    ξ3,
                    e2c,
                    ξ[1],
                    ξ[2],
                    Elements.lglpoints(FT, 1)[1],
                )
                (
                    fx1ξ1.(ξ1, ξ2, ξ3),
                    fx1ξ2.(ξ1, ξ2, ξ3),
                    fx1ξ3.(ξ1, ξ2, ξ3),
                    fx2ξ1.(ξ1, ξ2, ξ3),
                    fx2ξ2.(ξ1, ξ2, ξ3),
                    fx2ξ3.(ξ1, ξ2, ξ3),
                    fx3ξ1.(ξ1, ξ2, ξ3),
                    fx3ξ2.(ξ1, ξ2, ξ3),
                    fx3ξ3.(ξ1, ξ2, ξ3),
                )
            end
            J = @.(
                x1ξ1 * (x2ξ2 * x3ξ3 - x3ξ2 * x2ξ3) +
                x2ξ1 * (x3ξ2 * x1ξ3 - x1ξ2 * x3ξ3) +
                x3ξ1 * (x1ξ2 * x2ξ3 - x2ξ2 * x1ξ3)
            )
            ξ3x1 = (x2ξ1 .* x3ξ2 - x2ξ2 .* x3ξ1) ./ J
            ξ3x2 = (x3ξ1 .* x1ξ2 - x3ξ2 .* x1ξ1) ./ J
            ξ3x3 = (x1ξ1 .* x2ξ2 - x1ξ2 .* x2ξ1) ./ J

            d, f = 3, 5
            Mf = kron(1, ω[2], ω[1])
            @test [
                (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
            ] ≈ [
                (-J[:, :, 1, :] .* ξ3x1[:, :, 1, :])[:] .* Mf,
                (-J[:, :, 1, :] .* ξ3x2[:, :, 1, :])[:] .* Mf,
                (-J[:, :, 1, :] .* ξ3x3[:, :, 1, :])[:] .* Mf,
            ]

            d, f = 3, 6
            Mf = kron(1, ω[2], ω[1])
            @test [
                (sM[1:Nfp[d], f, :] .* n1[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n2[1:Nfp[d], f, :])[:],
                (sM[1:Nfp[d], f, :] .* n3[1:Nfp[d], f, :])[:],
            ] ≈ [
                (J[:, :, 2, :] .* ξ3x1[:, :, 2, :])[:] .* Mf,
                (J[:, :, 2, :] .* ξ3x2[:, :, 2, :])[:] .* Mf,
                (J[:, :, 2, :] .* ξ3x3[:, :, 2, :])[:] .* Mf,
            ]
        end
    end
    #}}}

    # Constant preserving test for N = 0
    #{{{
    let
        for FT in (Float64, Float32),
            N in ((4, 4, 0), (0, 0, 2), (0, 3, 4), (2, 0, 3))

            Nq = N .+ 1

            Np = prod(Nq)
            Nfp = div.(Np, Nq)

            dim = length(N)
            nface = 2dim

            # Create element operators for each polynomial order
            ξω = ntuple(
                j ->
                    Nq[j] == 1 ? Elements.glpoints(FT, N[j]) :
                    Elements.lglpoints(FT, N[j]),
                dim,
            )
            ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)
            D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

            rng = MersenneTwister(777)
            fx1(ξ1, ξ2, ξ3) = ξ1 + (ξ1 * ξ2 * ξ3 * rand(rng) + rand(rng)) / 10
            fx2(ξ1, ξ2, ξ3) = ξ2 + (ξ1 * ξ2 * ξ3 * rand(rng) + rand(rng)) / 10
            fx3(ξ1, ξ2, ξ3) = ξ3 + (ξ1 * ξ2 * ξ3 * rand(rng) + rand(rng)) / 10

            e2c = Array{FT, 3}(undef, 3, 8, 1)
            e2c[:, :, 1] = [
                -1 +1 -1 +1 -1 +1 -1 +1
                -1 -1 +1 +1 -1 -1 +1 +1
                -1 -1 -1 -1 +1 +1 +1 +1
            ]
            nelem = size(e2c, 3)

            meshwarp(ξ1, ξ2, ξ3) =
                (fx1(ξ1, ξ2, ξ3), fx2(ξ1, ξ2, ξ3), fx2(ξ1, ξ2, ξ3))
            (vgeo, sgeo, _) = Grids.computegeometry(e2c, D, ξ, ω, meshwarp)

            M = vgeo[:, _M, :]
            ξ1x1 = vgeo[:, _ξ1x1, :]
            ξ2x1 = vgeo[:, _ξ2x1, :]
            ξ3x1 = vgeo[:, _ξ3x1, :]
            ξ1x2 = vgeo[:, _ξ1x2, :]
            ξ2x2 = vgeo[:, _ξ2x2, :]
            ξ3x2 = vgeo[:, _ξ3x2, :]
            ξ1x3 = vgeo[:, _ξ1x3, :]
            ξ2x3 = vgeo[:, _ξ2x3, :]
            ξ3x3 = vgeo[:, _ξ3x3, :]

            M = vgeo[:, _M, :]
            ξ1x1 = vgeo[:, _ξ1x1, :]
            ξ2x1 = vgeo[:, _ξ2x1, :]
            ξ1x2 = vgeo[:, _ξ1x2, :]
            ξ2x2 = vgeo[:, _ξ2x2, :]

            I1 = Matrix(I, Nq[1], Nq[1])
            I2 = Matrix(I, Nq[2], Nq[2])
            I3 = Matrix(I, Nq[3], Nq[3])
            D1 = kron(I3, I2, D[1])
            D2 = kron(I3, D[2], I1)
            D3 = kron(D[3], I2, I1)

            # Face interpolation operators
            L = (
                kron(I3, I2, I1[1, :]'),
                kron(I3, I2, I1[Nq[1], :]'),
                kron(I3, I2[1, :]', I1),
                kron(I3, I2[Nq[2], :]', I1),
                kron(I3[1, :]', I2, I1),
                kron(I3[Nq[3], :]', I2, I1),
            )
            sM = ntuple(f -> sgeo[_sM, 1:Nfp[cld(f, 2)], f, :], nface)
            n1 = ntuple(f -> sgeo[_n1, 1:Nfp[cld(f, 2)], f, :], nface)
            n2 = ntuple(f -> sgeo[_n2, 1:Nfp[cld(f, 2)], f, :], nface)
            n3 = ntuple(f -> sgeo[_n3, 1:Nfp[cld(f, 2)], f, :], nface)

            # If constant preserving then:
            #   \sum_{j} = D' * M * ξjxk = \sum_{f} L_f' * sM_f * n1_f
            @test D1' * (M .* ξ1x1) + D2' * (M .* ξ2x1) + D3' * (M .* ξ3x1) ≈
                  mapreduce((L, sM, n1) -> L' * (sM .* n1), +, L, sM, n1)
            @test D1' * (M .* ξ1x2) + D2' * (M .* ξ2x2) + D3' * (M .* ξ3x2) ≈
                  mapreduce((L, sM, n2) -> L' * (sM .* n2), +, L, sM, n2)
            @test D1' * (M .* ξ1x3) + D2' * (M .* ξ2x3) + D3' * (M .* ξ3x3) ≈
                  mapreduce((L, sM, n3) -> L' * (sM .* n3), +, L, sM, n3)
        end
    end
    #}}}

    # Constant preserving test with all N = 0
    #{{{
    let
        for FT in (Float64, Float32)
            N = (0, 0, 0)
            Nq = N .+ 1

            Np = prod(Nq)
            Nfp = div.(Np, Nq)

            dim = length(N)
            nface = 2dim

            # Create element operators for each polynomial order
            ξω = ntuple(
                j ->
                    Nq[j] == 1 ? Elements.glpoints(FT, N[j]) :
                    Elements.lglpoints(FT, N[j]),
                dim,
            )
            ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)
            D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

            rng = MersenneTwister(777)
            fx1(ξ1, ξ2, ξ3) = ξ1 + (ξ1 * ξ2 * ξ3 * rand(rng) + rand(rng)) / 10
            fx2(ξ1, ξ2, ξ3) = ξ2 + (ξ1 * ξ2 * ξ3 * rand(rng) + rand(rng)) / 10
            fx3(ξ1, ξ2, ξ3) = ξ3 + (ξ1 * ξ2 * ξ3 * rand(rng) + rand(rng)) / 10

            e2c = Array{FT, 3}(undef, 3, 8, 1)
            e2c[:, :, 1] = [
                -1 +1 -1 +1 -1 +1 -1 +1
                -1 -1 +1 +1 -1 -1 +1 +1
                -1 -1 -1 -1 +1 +1 +1 +1
            ]
            nelem = size(e2c, 3)

            meshwarp(ξ1, ξ2, ξ3) =
                (fx1(ξ1, ξ2, ξ3), fx2(ξ1, ξ2, ξ3), fx2(ξ1, ξ2, ξ3))
            (vgeo, sgeo, _) = Grids.computegeometry(e2c, D, ξ, ω, meshwarp)

            M = vgeo[:, _M, :]
            ξ1x1 = vgeo[:, _ξ1x1, :]
            ξ2x1 = vgeo[:, _ξ2x1, :]
            ξ3x1 = vgeo[:, _ξ3x1, :]
            ξ1x2 = vgeo[:, _ξ1x2, :]
            ξ2x2 = vgeo[:, _ξ2x2, :]
            ξ3x2 = vgeo[:, _ξ3x2, :]
            ξ1x3 = vgeo[:, _ξ1x3, :]
            ξ2x3 = vgeo[:, _ξ2x3, :]
            ξ3x3 = vgeo[:, _ξ3x3, :]

            M = vgeo[:, _M, :]
            ξ1x1 = vgeo[:, _ξ1x1, :]
            ξ2x1 = vgeo[:, _ξ2x1, :]
            ξ1x2 = vgeo[:, _ξ1x2, :]
            ξ2x2 = vgeo[:, _ξ2x2, :]

            I1 = Matrix(I, Nq[1], Nq[1])
            I2 = Matrix(I, Nq[2], Nq[2])
            I3 = Matrix(I, Nq[3], Nq[3])
            D1 = kron(I3, I2, D[1])
            D2 = kron(I3, D[2], I1)
            D3 = kron(D[3], I2, I1)

            # Face interpolation operators
            L = (
                kron(I3, I2, I1[1, :]'),
                kron(I3, I2, I1[Nq[1], :]'),
                kron(I3, I2[1, :]', I1),
                kron(I3, I2[Nq[2], :]', I1),
                kron(I3[1, :]', I2, I1),
                kron(I3[Nq[3], :]', I2, I1),
            )
            sM = ntuple(f -> sgeo[_sM, 1:Nfp[cld(f, 2)], f, :], nface)
            n1 = ntuple(f -> sgeo[_n1, 1:Nfp[cld(f, 2)], f, :], nface)
            n2 = ntuple(f -> sgeo[_n2, 1:Nfp[cld(f, 2)], f, :], nface)
            n3 = ntuple(f -> sgeo[_n3, 1:Nfp[cld(f, 2)], f, :], nface)

            # If constant preserving then \sum_{f} L_f' * sM_f * n1_f ≈ 0
            @test abs(mapreduce(
                (L, sM, n1) -> L' * (sM .* n1),
                +,
                L,
                sM,
                n1,
            )[1]) < 10eps(FT)
            @test abs(mapreduce(
                (L, sM, n2) -> L' * (sM .* n2),
                +,
                L,
                sM,
                n2,
            )[1]) < 10eps(FT)
            @test abs(mapreduce(
                (L, sM, n3) -> L' * (sM .* n3),
                +,
                L,
                sM,
                n3,
            )[1]) < 10eps(FT)
        end
    end
    #}}}
end
