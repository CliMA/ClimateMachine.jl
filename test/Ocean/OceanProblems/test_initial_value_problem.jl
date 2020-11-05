using ClimateMachine

ClimateMachine.init()

using ClimateMachine.Ocean.OceanProblems: InitialConditions, InitialValueProblem

@testset "$(@__FILE__)" begin
    U = 0.1
    L = 0.2
    a = 0.3
    U = 0.4

    Ψ(x, L) = exp(-x^2 / 2L^2) # a Gaussian

    uᵢ(x, y, z) = + U * y / L * Ψ(x, L)
    vᵢ(x, y, z) = - U * x / L * Ψ(x, L)
    ηᵢ(x, y, z) = a * Ψ(x, L)
    θᵢ(x, y, z) = 20.0 + 1e-3 * z

    initial_conditions = InitialConditions(u=uᵢ, v=vᵢ, η=ηᵢ, θ=θᵢ)

    problem = InitialValueProblem(dimensions = (π, 42, 1.1), initial_conditions = initial_conditions)

    @test problem.Lˣ == Float64(π)
    @test problem.Lʸ == 42.0
    @test problem.H == 1.1

    @test problem.initial_conditions.u === uᵢ
    @test problem.initial_conditions.v === vᵢ
    @test problem.initial_conditions.η === ηᵢ
    @test problem.initial_conditions.θ === θᵢ
end
