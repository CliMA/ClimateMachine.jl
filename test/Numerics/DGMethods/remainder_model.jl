using Test
using LinearAlgebra
using MPI
using Random
using StaticArrays

using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Orientations
using ClimateMachine.TemperatureProfiles
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

"""
    main()

Run this test problem
"""
function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 5
    numelem_horz = 10
    numelem_vert = 5

    @testset "remainder model" begin
        for FT in (Float64,)# Float32)
            result = test_run(
                mpicomm,
                polynomialorder,
                numelem_horz,
                numelem_vert,
                ArrayType,
                FT,
            )
        end
    end
end

function test_run(
    mpicomm,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    ArrayType,
    FT,
)

    # Structure to pass around to setup the simulation
    setup = RemainderTestSetup{FT}()

    # Create the cubed sphere mesh
    _planet_radius::FT = planet_radius(param_set)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + setup.domain_height),
        nelem = numelem_vert,
    )
    topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
        meshwarp = equiangular_cubed_sphere_warp,
    )
    T_profile = IsothermalProfile(param_set, setup.T_ref)

    # This is the base model which defines all the data (all other DGModels
    # for substepping components will piggy-back off of this models data)
    fullphysics = AtmosPhysics{FT}(
        param_set;
        ref_state = HydrostaticState(T_profile),
        turbulence = Vreman(FT(0.23)),
        moisture = DryModel(),
    )
    fullmodel = AtmosModel{FT}(
        AtmosLESConfigType,
        fullphysics;
        orientation = SphericalOrientation(),
        init_state_prognostic = setup,
        source = (Gravity(),),
    )
    dg = DGModel(
        fullmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )
    Random.seed!(1235)
    Q = init_ode_state(dg, FT(0); init_on_cpu = true)

    acousticmodel = AtmosAcousticGravityLinearModel(fullmodel)

    acoustic_dg = DGModel(
        acousticmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = EveryDirection(),
        state_auxiliary = dg.state_auxiliary,
    )
    vacoustic_dg = DGModel(
        acousticmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = VerticalDirection(),
        state_auxiliary = dg.state_auxiliary,
    )
    hacoustic_dg = DGModel(
        acousticmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = HorizontalDirection(),
        state_auxiliary = dg.state_auxiliary,
    )

    # Create some random data to check the wavespeed function with
    nM = rand(3)
    nM /= norm(nM)
    state_prognostic = Vars{vars_state(dg.balance_law, Prognostic(), FT)}(rand(
        FT,
        number_states(dg.balance_law, Prognostic()),
    ))
    state_auxiliary = Vars{vars_state(dg.balance_law, Auxiliary(), FT)}(rand(
        FT,
        number_states(dg.balance_law, Auxiliary()),
    ))
    full_wavespeed = wavespeed(
        dg.balance_law,
        nM,
        state_prognostic,
        state_auxiliary,
        FT(0),
        (EveryDirection(),),
    )
    acoustic_wavespeed = wavespeed(
        acoustic_dg.balance_law,
        nM,
        state_prognostic,
        state_auxiliary,
        FT(0),
        (EveryDirection(),),
    )

    # Evaluate the full tendency
    full_tendency = similar(Q)
    dg(full_tendency, Q, nothing, 0; increment = false)

    # Evaluate various splittings
    split_tendency = similar(Q)

    # Check pulling acoustic model out
    @testset "full acoustic" begin
        rem_dg = remainder_DGModel(
            dg,
            (acoustic_dg,);
            numerical_flux_first_order = (
                dg.numerical_flux_first_order,
                (acoustic_dg.numerical_flux_first_order,),
            ),
        )
        rem_dg(split_tendency, Q, nothing, 0; increment = false)
        acoustic_dg(split_tendency, Q, nothing, 0; increment = true)
        A = Array(full_tendency.data)
        B = Array(split_tendency.data)
        # Test that we have a fully discrete splitting
        @test all(isapprox.(
            A,
            B,
            rtol = sqrt(eps(FT)),
            atol = 10 * sqrt(eps(FT)),
        ))

        # Test that wavespeeds are split by direction
        every_wavespeed = full_wavespeed .- acoustic_wavespeed
        horz_wavespeed = -zero(FT)
        vert_wavespeed = -zero(FT)
        @test all(
            every_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(),),
            ),
        )
        @test all(
            horz_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (HorizontalDirection(),),
            ),
        )
        @test all(
            vert_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (VerticalDirection(),),
            ),
        )
        @test all(
            every_wavespeed .+ horz_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(), HorizontalDirection()),
            ),
        )
        @test all(
            every_wavespeed .+ vert_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(), VerticalDirection()),
            ),
        )
    end

    # Check pulling acoustic model but as two pieces
    @testset "horizontal and vertical acoustic" begin
        rem_dg = remainder_DGModel(
            dg,
            (hacoustic_dg, vacoustic_dg);
            numerical_flux_first_order = (
                dg.numerical_flux_first_order,
                (
                    hacoustic_dg.numerical_flux_first_order,
                    vacoustic_dg.numerical_flux_first_order,
                ),
            ),
        )
        rem_dg(split_tendency, Q, nothing, 0; increment = false)
        vacoustic_dg(split_tendency, Q, nothing, 0; increment = true)
        hacoustic_dg(split_tendency, Q, nothing, 0; increment = true)
        A = Array(full_tendency.data)
        B = Array(split_tendency.data)
        # Test that we have a fully discrete splitting
        @test all(isapprox.(
            A,
            B,
            rtol = sqrt(eps(FT)),
            atol = 10 * sqrt(eps(FT)),
        ))

        # Test that wavespeeds are split by direction
        every_wavespeed = full_wavespeed
        horz_wavespeed = -acoustic_wavespeed
        vert_wavespeed = -acoustic_wavespeed
        @test all(
            every_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(),),
            ),
        )
        @test all(
            horz_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (HorizontalDirection(),),
            ),
        )
        @test all(
            vert_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (VerticalDirection(),),
            ),
        )
        @test all(
            every_wavespeed .+ horz_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(), HorizontalDirection()),
            ),
        )
        @test all(
            every_wavespeed .+ vert_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(), VerticalDirection()),
            ),
        )
    end

    # Check pulling horizontal acoustic model
    @testset "horizontal acoustic" begin
        rem_dg = remainder_DGModel(
            dg,
            (hacoustic_dg,);
            numerical_flux_first_order = (
                dg.numerical_flux_first_order,
                (hacoustic_dg.numerical_flux_first_order,),
            ),
        )
        rem_dg(split_tendency, Q, nothing, 0; increment = false)
        hacoustic_dg(split_tendency, Q, nothing, 0; increment = true)
        A = Array(full_tendency.data)
        B = Array(split_tendency.data)
        # Test that we have a fully discrete splitting
        @test all(isapprox.(
            A,
            B,
            rtol = sqrt(eps(FT)),
            atol = 10 * sqrt(eps(FT)),
        ))

        # Test that wavespeeds are split by direction
        every_wavespeed = full_wavespeed
        horz_wavespeed = -acoustic_wavespeed
        vert_wavespeed = -zero(eltype(FT))
        @test all(
            every_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(),),
            ),
        )
        @test all(
            horz_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (HorizontalDirection(),),
            ),
        )
        @test all(
            vert_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (VerticalDirection(),),
            ),
        )
        @test all(
            every_wavespeed .+ horz_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(), HorizontalDirection()),
            ),
        )
        @test all(
            every_wavespeed .+ vert_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(), VerticalDirection()),
            ),
        )
    end

    # Check pulling vertical acoustic model
    @testset "vertical acoustic" begin
        rem_dg = remainder_DGModel(
            dg,
            (vacoustic_dg,);
            numerical_flux_first_order = (
                dg.numerical_flux_first_order,
                (vacoustic_dg.numerical_flux_first_order,),
            ),
        )
        rem_dg(split_tendency, Q, nothing, 0; increment = false)
        vacoustic_dg(split_tendency, Q, nothing, 0; increment = true)
        A = Array(full_tendency.data)
        B = Array(split_tendency.data)
        # Test that we have a fully discrete splitting
        @test all(isapprox.(
            A,
            B,
            rtol = sqrt(eps(FT)),
            atol = 10 * sqrt(eps(FT)),
        ))

        # Test that wavespeeds are split by direction
        every_wavespeed = full_wavespeed
        horz_wavespeed = -zero(eltype(FT))
        vert_wavespeed = -acoustic_wavespeed
        @test all(
            every_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(),),
            ),
        )
        @test all(
            horz_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (HorizontalDirection(),),
            ),
        )
        @test all(
            vert_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (VerticalDirection(),),
            ),
        )
        @test all(
            every_wavespeed .+ horz_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(), HorizontalDirection()),
            ),
        )
        @test all(
            every_wavespeed .+ vert_wavespeed .≈ wavespeed(
                rem_dg.balance_law,
                nM,
                state_prognostic,
                state_auxiliary,
                FT(0),
                (EveryDirection(), VerticalDirection()),
            ),
        )
    end
end

Base.@kwdef struct RemainderTestSetup{FT}
    domain_height::FT = 10e3
    T_ref::FT = 300
end

function (setup::RemainderTestSetup)(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)

    # Vary around the reference state by 10% and a random velocity field
    state.ρ = (4 + rand(FT)) / 5 + aux.ref_state.ρ
    state.ρu = 2 * (@SVector rand(FT, 3)) .- 1
    state.energy.ρe = (4 + rand(FT)) / 5 + aux.ref_state.ρe

    nothing
end

main()
