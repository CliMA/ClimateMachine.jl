# This file establishes the default initial conditions, boundary conditions and sources
# for the baroclinicwave_problem experiment, following [Ullrich2014](@cite)

# Override default CLIMAParameters for consistency with literature on this case
CLIMAParameters.Planet.press_triple(::EarthParameterSet) = 610.78

struct BaroclinicWaveProblem{BCS, ISP, ISA, WP, BS, MP} <: AbstractAtmosProblem
    boundaryconditions::BCS
    init_state_prognostic::ISP
    init_state_auxiliary::ISA
    perturbation::WP
    base_state::BS
    moisture_profile::MP
end
function BaroclinicWaveProblem(;
    boundaryconditions = (AtmosBC(), AtmosBC()),
    perturbation = nothing,
    base_state = nothing,
    moisture_profile = nothing,
)
    # Set up defaults
    if isnothing(perturbation)
        perturbation = DeterministicPerturbation()
    end
    if isnothing(base_state)
        base_state = BCWaveBaseState()
    end
    if isnothing(moisture_profile)
        moisture_profile = MoistLowTropicsMoistureProfile()
    end

    problem = (
        boundaryconditions,
        init_gcm_experiment!,
        (_...) -> nothing,
        perturbation,
        base_state,
        moisture_profile,
    )
    return BaroclinicWaveProblem{typeof.(problem)...}(problem...)
end

problem_name(::BaroclinicWaveProblem) = "BaroclinicWave"

setup_source(::BaroclinicWaveProblem) = (Gravity(), Coriolis())
