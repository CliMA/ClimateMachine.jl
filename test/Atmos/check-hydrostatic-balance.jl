using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies: StackedBrickTopology
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Atmos

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, LinearAlgebra

function main()
    ClimateMachine.init()
    ArrayType = Array

    mpicomm = MPI.COMM_WORLD

    FT = Float64
    polynomialorder = 4
    for prof in (IsothermalProfile(param_set, FT),
                 IsothermalProfile(param_set, FT(300)),
                 DecayingTemperatureProfile{FT}(param_set),
                 DryAdiabaticProfile{FT}(param_set))

      println("Profile = $prof")
      refstate = HydrostaticState(prof)
      for numelems_z in (2, 4, 8)
        imbalance = check_balance(
            mpicomm,
            refstate,
            ArrayType,
            polynomialorder,
            numelems_z,
            FT,
        )
        println("Ne_z = $numelems_z, max |dp/dz + ρ * g| = $imbalance")
      end
    end
end

function check_balance(
    mpicomm,
    refstate,
    ArrayType,
    polynomialorder,
    numelems_z,
    FT,
)
    range_x = (-FT(1), FT(1))
    range_y = range_x
    top = FT(1000)
    range_z = range(FT(0), stop = FT(top), length = numelems_z)
    brickrange = (range_x, range_y, range_z)

    topology = StackedBrickTopology(
        mpicomm,
        brickrange;
        periodicity = (true, true, false),
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        orientation = FlatOrientation(),
        ref_state = refstate,
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        moisture = DryModel(),
        source = Gravity(),
        init_state_conservative = (),
    )
    
    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
    )

    Nq = polynomialorder + 1
    Nq2 = Nq ^ 2

    JI = 2 / step(range_z)
    D = grid.D
    
    # get one dimensional profiles
    ρ = dg.state_auxiliary.ref_state[1:Nq2:end, 1, :]
    p = dg.state_auxiliary.ref_state[1:Nq2:end, 2, :]

    _grav = FT(grav(param_set))

    imbalance = maximum(abs.(JI * D * p + _grav * ρ))
    return imbalance
end

main()
