using ClimateMachine.Mesh.Elements
using GaussQuadrature
using Test

@testset "GaussQuadrature" begin
    for T in (Float32, Float64, BigFloat)
        let
            x, w = GaussQuadrature.legendre(T, 1)
            @test iszero(x)
            @test w ≈ [2 * one(T)]
        end

        let
            endpt = GaussQuadrature.left
            x, w = GaussQuadrature.legendre(T, 1, endpt)
            @test x ≈ [-one(T)]
            @test w ≈ [2 * one(T)]
        end

        let
            endpt = GaussQuadrature.right
            x, w = GaussQuadrature.legendre(T, 1, endpt)
            @test x ≈ [one(T)]
            @test w ≈ [2 * one(T)]
        end

        let
            endpt = GaussQuadrature.left
            x, w = GaussQuadrature.legendre(T, 2, endpt)
            @test x ≈ [-one(T); T(1 // 3)]
            @test w ≈ [T(1 // 2); T(3 // 2)]
        end

        let
            endpt = GaussQuadrature.right
            x, w = GaussQuadrature.legendre(T, 2, endpt)
            @test x ≈ [T(-1 // 3); one(T)]
            @test w ≈ [T(3 // 2); T(1 // 2)]
        end
    end

    let
        err = ErrorException("Must have at least two points for both ends.")
        endpt = GaussQuadrature.both
        @test_throws err GaussQuadrature.legendre(1, endpt)
    end

    let
        T = Float64
        n = 100
        endpt = GaussQuadrature.both

        a, b = GaussQuadrature.legendre_coefs(T, n)

        err = ErrorException(
            "No convergence after 1 iterations " * "(try increasing maxits)",
        )

        @test_throws err GaussQuadrature.custom_gauss_rule(
            -one(T),
            one(T),
            a,
            b,
            endpt,
            1,
        )
    end
end

@testset "Operators" begin
    P5(r::AbstractVector{T}) where {T} =
        T(1) / T(8) * (T(15) * r - T(70) * r .^ 3 + T(63) * r .^ 5)

    P6(r::AbstractVector{T}) where {T} =
        T(1) / T(16) *
        (-T(5) .+ T(105) * r .^ 2 - T(315) * r .^ 4 + T(231) * r .^ 6)
    DP6(r::AbstractVector{T}) where {T} =
        T(1) / T(16) *
        (T(2 * 105) * r - T(4 * 315) * r .^ 3 + T(6 * 231) * r .^ 5)

    IPN(::Type{T}, N) where {T} = T(2) / T(2 * N + 1)

    N = 6
    for test_type in (Float32, Float64, BigFloat)
        r, w = Elements.lglpoints(test_type, N)
        D = Elements.spectralderivative(r)
        x = LinRange{test_type}(-1, 1, 101)
        I = Elements.interpolationmatrix(r, x)

        @test sum(P5(r) .^ 2 .* w) ≈ IPN(test_type, 5)
        @test D * P6(r) ≈ DP6(r)
        @test I * P6(r) ≈ P6(x)
    end

    for test_type in (Float32, Float64, BigFloat)
        r, w = Elements.lgpoints(test_type, N)
        D = Elements.spectralderivative(r)

        @test sum(P5(r) .^ 2 .* w) ≈ IPN(test_type, 5)
        @test sum(P6(r) .^ 2 .* w) ≈ IPN(test_type, 6)
        @test D * P6(r) ≈ DP6(r)
    end
end
