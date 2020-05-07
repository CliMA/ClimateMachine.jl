using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies:
    StackedCubedSphereTopology, cubedshellwarp, grid1d
using ClimateMachine.Mesh.Grids:
    DiscontinuousSpectralElementGrid,
    VerticalDirection,
    HorizontalDirection,
    EveryDirection
using ClimateMachine.DGmethods: DGModel, init_ode_state, remainder_DGModel
using ClimateMachine.DGmethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder
using ClimateMachine.Atmos:
    AtmosModel,
    SphericalOrientation,
    DryModel,
    Vreman,
    Gravity,
    HydrostaticState,
    IsothermalProfile,
    AtmosAcousticGravityLinearModel

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI
using Test
using StaticArrays

using Random

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
            result = run(
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

function run(
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
        meshwarp = cubedshellwarp,
    )

    # This is the base model which defines all the data (all other DGModels
    # for substepping components will piggy-back off of this models data)
    fullmodel = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        orientation = SphericalOrientation(),
        ref_state = HydrostaticState(IsothermalProfile(setup.T_ref), FT(0)),
        turbulence = Vreman(FT(0.23)),
        moisture = DryModel(),
        source = Gravity(),
        init_state_conservative = setup,
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

    # Evaluate the full tendency
    full_tendency = similar(Q)
    dg(full_tendency, Q, nothing, 0; increment = false)

    # Evaluate various splittings
    split_tendency = similar(Q)

    # Check pulling acoustic model out
    rem_dg = remainder_DGModel(dg, (acoustic_dg,))
    rem_dg(split_tendency, Q, nothing, 0; increment = false)
    acoustic_dg(split_tendency, Q, nothing, 0; increment = true)
    A = Array(full_tendency.data)
    B = Array(split_tendency.data)
    @test all(isapprox.(A, B, rtol = sqrt(eps(FT)), atol = 10 * sqrt(eps(FT))))

    # Check pulling acoustic model but as two pieces
    rem_dg = remainder_DGModel(dg, (hacoustic_dg, vacoustic_dg))
    rem_dg(split_tendency, Q, nothing, 0; increment = false)
    vacoustic_dg(split_tendency, Q, nothing, 0; increment = true)
    hacoustic_dg(split_tendency, Q, nothing, 0; increment = true)
    A = Array(full_tendency.data)
    B = Array(split_tendency.data)
    @test all(isapprox.(A, B, rtol = sqrt(eps(FT)), atol = 10 * sqrt(eps(FT))))

    # Check pulling horizontal acoustic model
    rem_dg = remainder_DGModel(dg, (hacoustic_dg,))
    rem_dg(split_tendency, Q, nothing, 0; increment = false)
    hacoustic_dg(split_tendency, Q, nothing, 0; increment = true)
    A = Array(full_tendency.data)
    B = Array(split_tendency.data)
    @test all(isapprox.(A, B, rtol = sqrt(eps(FT)), atol = 10 * sqrt(eps(FT))))

    # Check pulling vertical acoustic model
    rem_dg = remainder_DGModel(dg, (vacoustic_dg,))
    rem_dg(split_tendency, Q, nothing, 0; increment = false)
    vacoustic_dg(split_tendency, Q, nothing, 0; increment = true)
    A = Array(full_tendency.data)
    B = Array(split_tendency.data)
    @test all(isapprox.(A, B, rtol = sqrt(eps(FT)), atol = 10 * sqrt(eps(FT))))
end

Base.@kwdef struct RemainderTestSetup{FT}
    domain_height::FT = 10e3
    T_ref::FT = 300
end

function (setup::RemainderTestSetup)(bl, state, aux, coords, t)
    FT = eltype(state)

    # Vary around the reference state by 10% and a random velocity field
    state.ρ = (4 + rand(FT)) / 5 + aux.ref_state.ρ
    state.ρu = 2 * (@SVector rand(FT, 3)) .- 1
    state.ρe = (4 + rand(FT)) / 5 + aux.ref_state.ρe

    nothing
end

main()
