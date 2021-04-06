using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.MPIStateArrays
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Geometry: LocalGeometry

using LinearAlgebra
using StaticArrays
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_to_ref_state!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.energy.ρe = aux.ref_state.ρe
end

function config_balanced(
    FT,
    poly_order,
    temp_profile,
    numflux,
    (config_type, config_fun, config_args),
)
    ref_state = HydrostaticState(temp_profile; subtract_off = false)

    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = ref_state,
        turbulence = ConstantDynamicViscosity(FT(0)),
        hyperdiffusion = NoHyperDiffusion(),
        moisture = DryModel(),
    )
    model = AtmosModel{FT}(
        config_type,
        physics;
        source = (Gravity(),),
        init_state_prognostic = init_to_ref_state!,
    )

    config = config_fun(
        "balanced state",
        poly_order,
        config_args...,
        param_set,
        nothing;
        model = model,
        numerical_flux_first_order = numflux,
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
            @testset for numflux in (
                CentralNumericalFluxFirstOrder(),
                RoeNumericalFlux(),
                HLLCNumericalFlux(),
            )
                @testset for temp_profile in (
                    IsothermalProfile(param_set, FT),
                    DecayingTemperatureProfile{FT}(param_set),
                )
                    driver_config = config_balanced(
                        FT,
                        poly_order,
                        temp_profile,
                        numflux,
                        config,
                    )

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

                    error =
                        euclidean_distance(solver_config.Q, Qinit) / norm(Qinit)
                    @test error <= 100 * eps(FT)
                end
            end
        end
    end
end

main()
