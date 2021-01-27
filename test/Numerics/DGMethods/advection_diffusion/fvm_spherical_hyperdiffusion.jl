using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
import ClimateMachine.DGMethods.FVReconstructions: FVConstant, FVLinear

using ClimateMachine.MPIStateArrays
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.ODESolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.Mesh.Grids: min_node_distance
using ClimateMachine.Atmos: SphericalOrientation, latitude, longitude

using ClimateMachine.Orientations
using CLIMAParameters
using CLIMAParameters.Planet

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 60e3



const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

include("hyperdiffusion_model.jl")

struct ConstantHyperDiffusion{dim, dir, FT} <: HyperDiffusionProblem
    D::FT
    l::FT
    m::FT
end

function nodal_init_state_auxiliary!(
    balance_law::HyperDiffusion,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    aux.D = balance_law.problem.D * SMatrix{3, 3, Float64}(I)
    aux.coord = geom.coord
end

"""
    initial condition is given by ρ0 = ρ0(r)
    test: ∇^4_horz ρ0(r) = 0
"""

function initial_condition!(
    problem::ConstantHyperDiffusion{dim, dir},
    state,
    aux,
    x,
    t,
) where {dim, dir}
    @inbounds begin
        FT = eltype(state)
        # import planet paraset
        _a::FT = planet_radius(param_set)
        r = norm(aux.coord)
        z = r - _a
        domain_height = 30.0e3
        state.ρ = cos(z / domain_height)
    end
end

function run(mpicomm, ArrayType, dim, topl, N, FT, direction, τ, l, m)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
        meshwarp = ClimateMachine.Mesh.Topologies.cubedshellwarp,
    )
    dx = min_node_distance(grid, HorizontalDirection())
    dz = min_node_distance(grid, VerticalDirection())

    D = (dx / 2)^4 / 2 / τ

    model = HyperDiffusion{dim}(ConstantHyperDiffusion{dim, direction(), FT}(
        D,
        l,
        m,
    ))
    dg = DGFVModel(
        model,
        grid,
        FVLinear(),
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        direction = direction(),
    )

    Q0 = init_ode_state(dg, FT(0))
    @info "Δ(horz) Δ(vert)" (dx, dz)

    rhs_DGsource = similar(Q0)

    dg(rhs_DGsource, Q0, nothing, 0)

    result = norm(rhs_DGsource)

    return result
end

using Test
let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    dim = 3
    domain_height = 30.0e3

    _a = planet_radius(param_set)
    vert_num_elem = 20
    polynomialorder = (5, 0)

    @testset "$(@__FILE__)" begin
        for FT in (Float64,)
            for base_num_elem in (8,)
                for direction in (HorizontalDirection,)
                    for τ in (1,) # time scale for hyperdiffusion

                        topl = StackedCubedSphereTopology(
                            mpicomm,
                            base_num_elem,
                            grid1d(
                                _a,
                                _a + domain_height,
                                nelem = vert_num_elem,
                            ),
                        )

                        @info "Array FT nhorz nvert poly τ" (
                            ArrayType,
                            FT,
                            base_num_elem,
                            vert_num_elem,
                            polynomialorder,
                            τ,
                        )
                        result = run(
                            mpicomm,
                            ArrayType,
                            dim,
                            topl,
                            polynomialorder,
                            FT,
                            direction,
                            τ * 3600,
                            7,
                            4,
                        )

                        @test result < 1.0e-12

                    end
                end
            end
        end
    end
end
nothing
