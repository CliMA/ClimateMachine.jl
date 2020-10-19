using Test
using ClimateMachine.SurfaceFluxes.UniversalFunctions
using ClimateMachine.SurfaceFluxes.UniversalFunctions: b_m, a_m, a_h, b_h, Pr_0
using CLIMAParameters
using CLIMAParameters.SurfaceFluxes.UniversalFunctions
using CLIMAParameters.Planet
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# TODO: Right now, we test these functions for
# type stability and correctness in the asymptotic
# limit. We may want to extend correctness tests.

@testset "UniversalFunctions" begin
    @testset "Type stability" begin
        FT = Float32
        ζ = FT(-2):FT(0.01):FT(200)
        for L in (-FT(10), FT(10))
            args = (param_set, L)
            for uf in (Gryanik(args...), Grachev(args...), Businger(args...))
                for transport in (MomentumTransport(), HeatTransport())
                    ϕ = phi.(uf, ζ, transport)
                    @test eltype(ϕ) == FT
                    ψ = psi.(uf, ζ, transport)
                    @test eltype(ψ) == FT
                end
            end
        end
    end
    @testset "Conversions" begin
        FT = Float32
        ζ = FT(10)
        L = FT(10)
        args = (param_set, L)

        uf = Gryanik(args...)
        @test Businger(uf) isa Businger
        @test Grachev(uf) isa Grachev

        uf = Grachev(args...)
        @test Businger(uf) isa Businger
        @test Gryanik(uf) isa Gryanik

        uf = Businger(args...)
        @test Grachev(uf) isa Grachev
        @test Gryanik(uf) isa Gryanik
    end
    @testset "Asymptotic range" begin
        FT = Float32

        ϕ_h_ζ∞(uf::Grachev) = 1 + FT(b_h(uf))
        ϕ_m_ζ∞(uf::Grachev, ζ) = FT(a_m(uf)) / FT(b_m(uf)) * ζ^FT(1 / 3)

        ϕ_h_ζ∞(uf::Gryanik) = FT(Pr_0(uf)) * (1 + FT(a_h(uf) / b_h(uf)))
        ϕ_m_ζ∞(uf::Gryanik, ζ) = FT(a_m(uf) / b_m(uf)^FT(2 / 3)) * ζ^FT(1 / 3)

        for L in (-FT(10), FT(10))
            args = (param_set, L)
            for uf in (Grachev(args...), Gryanik(args...))
                for ζ in FT(10) .^ (4, 6, 8, 10)
                    ϕ_h = phi(uf, ζ, HeatTransport())
                    @test isapprox(ϕ_h, ϕ_h_ζ∞(uf))
                end
                for ζ in FT(10) .^ (8, 9, 10)
                    ϕ_m = phi(uf, ζ, MomentumTransport())
                    @test isapprox(ϕ_m, ϕ_m_ζ∞(uf, ζ))
                end
            end
        end

    end

end
