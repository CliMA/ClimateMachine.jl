using ClimateMachine
ClimateMachine.init()

using ClimateMachine.MPIStateArrays
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.DGMethods.NumericalFluxes: CentralNumericalFluxFirstOrder
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.Atmos: ReferenceState
import ClimateMachine.Atmos:
    atmos_init_ref_state_pressure!, atmos_init_aux!, vars_state_auxiliary

using LinearAlgebra
using StaticArrays
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# a simple wrapper around a hydrostatic state that
# prevents factoring out of the state in the
# momentum equation and lets us test the balance
struct TestRefState{HS} <: ReferenceState
    hydrostatic_state::HS
end

vars_state_auxiliary(m::TestRefState, FT) =
    vars_state_auxiliary(m.hydrostatic_state, FT)
function atmos_init_ref_state_pressure!(
    m::TestRefState,
    atmos::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    atmos_init_ref_state_pressure!(m.hydrostatic_state, atmos, aux, geom)
end
function atmos_init_aux!(
    m::TestRefState,
    atmos::AtmosModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    atmos_init_aux!(m.hydrostatic_state, atmos, aux, tmp, geom)
end

function init_to_ref_state!(bl, state, aux, coords, t)
    FT = eltype(state)
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = aux.ref_state.ρe
end

function config_balanced(
    FT,
    poly_order,
    temp_profile,
    (config_type, config_fun, config_args),
)
    ref_state = HydrostaticState(temp_profile)

    model = AtmosModel{FT}(
        config_type,
        param_set;
        ref_state = TestRefState(ref_state),
        turbulence = ConstantViscosityWithDivergence{FT}(0),
        hyperdiffusion = NoHyperDiffusion(),
        moisture = DryModel(),
        source = Gravity(),
        init_state_conservative = init_to_ref_state!,
    )

    config = config_fun(
        "balanced state",
        poly_order,
        config_args...,
        param_set,
        nothing;
        model = model,
        numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
    )

    return config
end

function main()
    FT = Float64
    poly_order = 4

    timestart = FT(0)
    timeend = FT(100)
    domain_height = FT(50e3)

    LES_params = let
        LES_resolution = ntuple(_ -> domain_height / 3poly_order, 3)
        LES_domain = ntuple(_ -> domain_height, 3)
        (LES_resolution, LES_domain...)
    end

    GCM_params = let
        GCM_resolution = (3, 3)
        (GCM_resolution, domain_height)
    end

    GCM = (AtmosGCMConfigType, ClimateMachine.AtmosGCMConfiguration, GCM_params)
    LES = (AtmosLESConfigType, ClimateMachine.AtmosLESConfiguration, LES_params)

    imex_solver_type = ClimateMachine.IMEXSolverType(
        splitting_type = HEVISplitting(),
        implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
    )
    explicit_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK54CarpenterKennedy,
    )

    @testset for config in (LES, GCM)
        @testset for ode_solver_type in (explicit_solver_type, imex_solver_type)
            @testset for temp_profile in (
                IsothermalProfile(param_set, FT),
                DecayingTemperatureProfile{FT}(param_set),
            )
                driver_config =
                    config_balanced(FT, poly_order, temp_profile, config)

                solver_config = ClimateMachine.SolverConfiguration(
                    timestart,
                    timeend,
                    driver_config,
                    Courant_number = FT(0.1),
                    init_on_cpu = true,
                    ode_solver_type = ode_solver_type,
                    CFL_direction = EveryDirection(),
                    diffdir = HorizontalDirection(),
                )

                Qinit = similar(solver_config.Q)
                Qinit .= solver_config.Q

                ClimateMachine.invoke!(solver_config)

                error = euclidean_distance(solver_config.Q, Qinit) / norm(Qinit)
                @test error <= 100 * eps(FT)
            end
        end
    end
end

main()
