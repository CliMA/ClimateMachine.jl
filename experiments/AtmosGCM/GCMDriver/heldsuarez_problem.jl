# This file establishes the default initial conditions, boundary conditions and sources
# for the heldsuarez_problem experiment, following:
#
# - Held, I. M. and Suarez, M. J.: A proposal for the intercomparison
# of the dynamical cores of atmospheric general circulation models,
# B. Am. Meteorol. Soc., 75, 1825–1830, 1994.
#
# - Thatcher, D. R. and Jablonowski, C.: A moist aquaplanet variant of the
# Held–Suarez test for atmospheric model dynamical cores, Geosci. Model Dev.,
# 9, 1263–1292, 2016.

# Override default CLIMAParameters for consistency with literature on this case
using ClimateMachine.Thermodynamics

using CLIMAParameters.Planet

struct HeldSuarezProblem{BC, ISP, ISA, WP, BS, MP} <: AbstractAtmosProblem
    boundarycondition::BC
    init_state_prognostic::ISP
    init_state_auxiliary::ISA
    perturbation::WP
    base_state::BS
    moisture_profile::MP
end
function HeldSuarezProblem(;
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
        base_state = HeldSuarezBaseState()
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
    return HeldSuarezProblem{typeof.(problem)...}(problem...)
end

problem_name(::HeldSuarezProblem) = "HeldSuarez"

setup_source(::HeldSuarezProblem) =
    (Gravity(), Coriolis(), held_suarez_forcing!)
