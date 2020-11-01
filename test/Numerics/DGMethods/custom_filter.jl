using Test
using ClimateMachine
using ClimateMachine.VariableTemplates: @vars, Vars
using ClimateMachine.BalanceLaws
using ClimateMachine.DGMethods: AbstractCustomFilter, apply!
import ClimateMachine
import ClimateMachine.BalanceLaws:
    vars_state, init_state_auxiliary!, init_state_prognostic!
using MPI
using LinearAlgebra

struct CustomFilterTestModel <: BalanceLaw end
struct CustomTestFilter <: AbstractCustomFilter end

vars_state(::CustomFilterTestModel, ::Auxiliary, FT) = @vars()
vars_state(::CustomFilterTestModel, ::Prognostic, FT) where {N} = @vars(q::FT)

init_state_auxiliary!(::CustomFilterTestModel, _...) = nothing

function init_state_prognostic!(
    ::CustomFilterTestModel,
    state::Vars,
    aux::Vars,
    localgeo,
)
    coord = localgeo.coord
    state.q = hypot(coord[1], coord[2])
end

@testset "Test custom filter" begin
    let
        ClimateMachine.init()
        N = 4
        Ne = (2, 2, 2)

        function ClimateMachine.DGMethods.custom_filter!(
            ::CustomTestFilter,
            bl::CustomFilterTestModel,
            state::Vars,
            aux::Vars,
        )
            state.q = state.q^2
        end

        @testset for FT in (Float64, Float32)
            dim = 2
            brickrange =
                ntuple(j -> range(FT(-1); length = Ne[j] + 1, stop = 1), dim)
            topl = ClimateMachine.Mesh.Topologies.BrickTopology(
                MPI.COMM_WORLD,
                brickrange,
                periodicity = ntuple(j -> true, dim),
            )

            grid = ClimateMachine.Mesh.Grids.DiscontinuousSpectralElementGrid(
                topl,
                FloatType = FT,
                DeviceArray = ClimateMachine.array_type(),
                polynomialorder = N,
            )

            model = CustomFilterTestModel()
            dg = ClimateMachine.DGMethods.DGModel(
                model,
                grid,
                nothing,
                nothing,
                nothing;
                state_gradient_flux = nothing,
            )

            Q = ClimateMachine.DGMethods.init_ode_state(dg)
            data = Array(Q.realdata)

            apply!(CustomTestFilter(), grid, model, Q, dg.state_auxiliary)

            @test all(Array(Q.realdata) .== data .^ 2)
        end
    end
end
