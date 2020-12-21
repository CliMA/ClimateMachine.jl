using ClimateMachine

ClimateMachine.init()

using ClimateMachine.Ocean.Domains

using CLIMAParameters: AbstractEarthParameterSet
struct DefaultParameters <: AbstractEarthParameterSet end

using ClimateMachine.Ocean:
    HydrostaticBoussinesqSuperModel, current_time, current_step, Δt

@testset "$(@__FILE__)" begin

    domain = RectangularDomain(
        Ne = (1, 1, 1),
        Np = 4,
        x = (-1, 1),
        y = (-1, 1),
        z = (-1, 0),
    )

    model = HydrostaticBoussinesqSuperModel(
        domain = domain,
        time_step = 0.1,
        parameters = DefaultParameters(),
    )

    @test model isa HydrostaticBoussinesqSuperModel
    @test Δt(model) == 0.1
    @test current_step(model) == 0
    @test current_time(model) == 0.0
end
