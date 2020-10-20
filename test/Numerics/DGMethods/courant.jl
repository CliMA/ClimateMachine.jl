using Test
using LinearAlgebra
using MPI
using StaticArrays

using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Courant
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.ODESolvers
using ClimateMachine.Orientations
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.VTK

using CLIMAParameters
using CLIMAParameters.Planet: kappa_d
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

const p∞ = 10^5
const T∞ = 300.0

function initialcondition!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)

    coord = localgeo.coord

    translation_speed::FT = 150
    translation_angle::FT = pi / 4
    α = translation_angle
    u∞ = SVector(
        FT(translation_speed * coord[1]),
        FT(translation_speed * coord[1]),
        FT(0),
    )
    _kappa_d::FT = kappa_d(param_set)

    u = u∞
    T = FT(T∞)
    # adiabatic/isentropic relation
    p = FT(p∞) * (T / FT(T∞))^(FT(1) / _kappa_d)
    ρ = air_density(bl.param_set, T, p)

    state.ρ = ρ
    state.ρu = ρ * u
    e_kin = u' * u / 2
    state.ρe = ρ * total_energy(bl.param_set, e_kin, FT(0), T)

    nothing
end


let
    # boiler plate MPI stuff
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    # Mesh generation parameters
    N = 4
    Nq = N + 1
    Neh = 10
    Nev = 4

    @testset "$(@__FILE__) DGModel matrix" begin
        for FT in (Float64, Float32)
            for dim in (2, 3)
                if dim == 2
                    brickrange = (
                        range(FT(0); length = Neh + 1, stop = 1),
                        range(FT(1); length = Nev + 1, stop = 2),
                    )
                elseif dim == 3
                    brickrange = (
                        range(FT(0); length = Neh + 1, stop = 1),
                        range(FT(0); length = Neh + 1, stop = 1),
                        range(FT(1); length = Nev + 1, stop = 2),
                    )
                end
                μ = FT(2)
                topl = StackedBrickTopology(mpicomm, brickrange)



                grid = DiscontinuousSpectralElementGrid(
                    topl,
                    FloatType = FT,
                    DeviceArray = ArrayType,
                    polynomialorder = N,
                )

                problem = AtmosProblem(
                    boundarycondition = (),
                    init_state_prognostic = initialcondition!,
                )

                model = AtmosModel{FT}(
                    AtmosLESConfigType,
                    param_set;
                    problem = problem,
                    ref_state = NoReferenceState(),
                    turbulence = ConstantDynamicViscosity(μ, WithDivergence()),
                    moisture = DryModel(),
                    source = Gravity(),
                )

                dg = DGModel(
                    model,
                    grid,
                    RusanovNumericalFlux(),
                    CentralNumericalFluxSecondOrder(),
                    CentralNumericalFluxGradient(),
                )

                Δt = FT(1 // 200)

                Q = init_ode_state(dg, FT(0))

                Δx = min_node_distance(grid, EveryDirection())
                Δx_v = min_node_distance(grid, VerticalDirection())
                Δx_h = min_node_distance(grid, HorizontalDirection())

                translation_speed = FT(norm([150.0, 150.0, 0.0]))
                diff_speed_h =
                    FT(μ / air_density(model.param_set, FT(T∞), FT(p∞)))
                diff_speed_v =
                    FT(μ / air_density(model.param_set, FT(T∞), FT(p∞)))
                c_h =
                    Δt * (
                        translation_speed +
                        soundspeed_air(model.param_set, FT(T∞))
                    ) / Δx_h
                c_v = Δt * (soundspeed_air(model.param_set, FT(T∞))) / Δx_v
                d_h = Δt * diff_speed_h / Δx_h^2
                d_v = Δt * diff_speed_v / Δx_v^2
                simtime = FT(0)

                # tests for non diffusive courant number
                rtol = FT === Float64 ? 1e-4 : 1f-2
                @test courant(
                    nondiffusive_courant,
                    dg,
                    model,
                    Q,
                    Δt,
                    simtime,
                    HorizontalDirection(),
                ) ≈ c_h rtol = rtol
                @test courant(
                    nondiffusive_courant,
                    dg,
                    model,
                    Q,
                    Δt,
                    simtime,
                    VerticalDirection(),
                ) ≈ c_v rtol = rtol

                # tests for diffusive courant number
                @test courant(
                    diffusive_courant,
                    dg,
                    model,
                    Q,
                    Δt,
                    simtime,
                    HorizontalDirection(),
                ) ≈ d_h
                @test courant(
                    diffusive_courant,
                    dg,
                    model,
                    Q,
                    Δt,
                    simtime,
                    VerticalDirection(),
                ) ≈ d_v
            end
        end
    end
end

nothing
