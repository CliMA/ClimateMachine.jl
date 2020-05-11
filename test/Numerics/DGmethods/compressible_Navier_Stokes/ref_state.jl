using MPI
using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.GenericCallbacks
using ClimateMachine.Atmos
using ClimateMachine.VariableTemplates
using ClimateMachine.MoistThermodynamics
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using ClimateMachine.VTK

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

using ClimateMachine.Atmos
using ClimateMachine.Atmos: internal_energy, thermo_state
import ClimateMachine.Atmos: MoistureModel, temperature, pressure, soundspeed

init_state_conservative!(bl, state, aux, coords, t) = nothing

# initial condition
using ClimateMachine.Atmos: vars_state_auxiliary

function run1(mpicomm, ArrayType, dim, topl, N, timeend, FT, dt)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    T_s = 320.0
    RH = 0.01
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        ref_state = HydrostaticState(IsothermalProfile(T_s), RH),
        init_state_conservative = init_state_conservative!,
    )

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))

    mkpath("vtk")
    outprefix = @sprintf("vtk/refstate")
    writevtk(
        outprefix,
        dg.state_auxiliary,
        dg,
        flattenednames(vars_state_auxiliary(model, FT)),
    )
    return FT(0)
end

function run2(mpicomm, ArrayType, dim, topl, N, timeend, FT, dt)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    T_min, T_s, Γ = FT(290), FT(320), FT(6.5 * 10^-3)
    RH = 0.01
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        ref_state = HydrostaticState(
            LinearTemperatureProfile(T_min, T_s, Γ),
            RH,
        ),
        init_state_conservative = init_state_conservative!,
    )

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))

    mkpath("vtk")
    outprefix = @sprintf("vtk/refstate")
    writevtk(
        outprefix,
        dg.state_auxiliary,
        dg,
        flattenednames(vars_state_auxiliary(model, FT)),
    )
    return FT(0)
end

using Test
let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    base_num_elem = 4

    expected_result = [0.0 0.0 0.0; 0.0 0.0 0.0]
    lvls = integration_testing ? size(expected_result, 2) : 1

    for FT in (Float64,) #Float32)
        result = zeros(FT, lvls)
        x_max = FT(25 * 10^3)
        y_max = FT(25 * 10^3)
        z_max = FT(25 * 10^3)
        dim = 3
        for l in 1:lvls
            Ne = (2^(l - 1) * base_num_elem, 2^(l - 1) * base_num_elem)
            brickrange = (
                range(FT(0); length = Ne[1] + 1, stop = x_max),
                range(FT(0); length = Ne[2] + 1, stop = y_max),
                range(FT(0); length = Ne[2] + 1, stop = z_max),
            )
            topl = BrickTopology(
                mpicomm,
                brickrange,
                periodicity = (false, false, false),
            )
            dt = 5e-3 / Ne[1]

            timeend = 2 * dt
            nsteps = ceil(Int64, timeend / dt)
            dt = timeend / nsteps

            @info (ArrayType, FT, dim)
            result[l] = run1(
                mpicomm,
                ArrayType,
                dim,
                topl,
                polynomialorder,
                timeend,
                FT,
                dt,
            )
            result[l] = run2(
                mpicomm,
                ArrayType,
                dim,
                topl,
                polynomialorder,
                timeend,
                FT,
                dt,
            )
        end
        if integration_testing
            @info begin
                msg = ""
                for l in 1:(lvls - 1)
                    rate = log2(result[l]) - log2(result[l + 1])
                    msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
                end
                msg
            end
        end
    end
end


#nothing
