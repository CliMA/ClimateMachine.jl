using ClimateMachine

ClimateMachine.init()

using ClimateMachine.Ocean.Domains

@testset "$(@__FILE__)" begin

    for FT in (Float64, Float32)
        domain = RectangularDomain(
            FT,
            Ne = (16, 24, 1),
            Np = 4,
            x = (0, π),
            y = (0, 1.1),
            z = (-1, 0),
            periodicity = (false, false, false),
        )

        @test eltype(domain) == FT
        @test domain.Ne == (x = 16, y = 24, z = 1)
        @test domain.Np == 4
        @test domain.L.x == FT(π)
        @test domain.L.y == FT(1.1)
        @test domain.L.z == FT(1)
    end
end
