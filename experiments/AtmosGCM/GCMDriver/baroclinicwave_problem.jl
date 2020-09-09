# This file establishes the default initial conditions, boundary conditions and sources
# for the baroclinicwave_problem experiment, following
#
# Ullrich, P. A., Melvin, T., Jablonowski, C., and Staniforth, A.:
# A proposed baroclinic wave test case for deep- and shallow atmosphere
# dynamical cores, Q. J. Roy. Meteor. Soc., 140, 1590-1602, doi:10.1002/qj.2241, 2014.

# Override default CLIMAParameters for consistency with literature on this case
CLIMAParameters.Planet.press_triple(::EarthParameterSet) = 610.78

struct BaroclinicWaveProblem{BC, ISP, ISA, WP, BS, MP} <: AbstractAtmosProblem
    boundarycondition::BC
    init_state_prognostic::ISP
    init_state_auxiliary::ISA
    perturbation::WP
    base_state::BS
    moisture_profile::MP
end
function BaroclinicWaveProblem(;
    boundarycondition = (AtmosBC(), AtmosBC()),
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
        boundarycondition,
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
