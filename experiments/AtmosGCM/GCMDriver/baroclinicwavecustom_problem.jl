# This file establishes the default initial conditions, boundary conditions and sources
# for the baroclinicwave_problem experiment, following
#
# Ullrich, P. A., Melvin, T., Jablonowski, C., and Staniforth, A.:
# A proposed baroclinic wave test case for deep- and shallow atmosphere
# dynamical cores, Q. J. Roy. Meteor. Soc., 140, 1590-1602, doi:10.1002/qj.2241, 2014.

# Override default CLIMAParameters for consistency with literature on this case
CLIMAParameters.Planet.press_triple(::EarthParameterSet) = 610.78

struct BaroclinicWaveCustomProblem{BC, ISP, ISA, WP, BS, MP} <: AbstractAtmosProblem
    boundarycondition::BC
    init_state_prognostic::ISP
    init_state_auxiliary::ISA
    perturbation::WP
    base_state::BS
    moisture_profile::MP
end
function BaroclinicWaveCustomProblem(;
    boundarycondition = (AtmosBC(), AtmosBC()),
    perturbation = DeterministicPerturbation(),
    base_state = BCWaveBaseState(),
    moisture_profile = ZeroMoistureProfile(),
)
    problem = (
        boundarycondition,
        init_gcm_experiment!,
        (_...) -> nothing,
        perturbation,
        base_state,
        moisture_profile,
    )
    return BaroclinicWaveCustomProblem{typeof.(problem)...}(problem...)
end

problem_name(::BaroclinicWaveCustomProblem) = "BaroclinicWaveCustom_micro0_RH100"

setup_source(::BaroclinicWaveCustomProblem) = (Gravity(), Coriolis(), NudgeToSaturation(), RemovePrecipitation(true))#, NudgeToSaturation())
