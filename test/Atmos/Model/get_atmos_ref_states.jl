using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.Checkpoint
using ClimateMachine.Mesh.Grids
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using ClimateMachine.SingleStackUtils
using ClimateMachine.VariableTemplates

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using Test
ClimateMachine.init()

function get_atmos_ref_states(nelem_vert, N_poly, RH)

    FT = Float64
    model = AtmosModel{FT}(
        SingleStackConfigType,
        param_set;
        ref_state = HydrostaticState(
            DecayingTemperatureProfile{FT}(param_set),
            RH,
        ),
        init_state_prognostic = (problem, bl, state, aux, localgeo, t) ->
            nothing,
    )
    driver_config = ClimateMachine.SingleStackConfiguration(
        "ref_state",
        N_poly,
        nelem_vert,
        FT(25e3),
        param_set,
        model,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        FT(0),
        FT(10),
        driver_config;
        skip_update_aux = true,
        ode_dt = FT(1),
    )

    return solver_config
end
