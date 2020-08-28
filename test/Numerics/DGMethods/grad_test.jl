using MPI
using StaticArrays
using ClimateMachine
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.MPIStateArrays
using ClimateMachine.DGMethods

using ClimateMachine.BalanceLaws

import ClimateMachine.BalanceLaws: vars_state, nodal_init_state_auxiliary!

using ClimateMachine.Mesh.Geometry: LocalGeometry

struct GradTestModel{dim, dir} <: BalanceLaw end

vars_state(m::GradTestModel, ::Auxiliary, T) = @vars begin
    a::T
    ∇a::SVector{3, T}
    ∇a_exact::SVector{3, T}
end
vars_state(::GradTestModel, ::Prognostic, T) = @vars()

function nodal_init_state_auxiliary!(
    ::GradTestModel{dim, dir},
    aux::Vars,
    tmp::Vars,
    g::LocalGeometry,
) where {dim, dir}
    x, y, z = g.coord
    if dim == 2
        aux.a = x^2 + y^3 - x * y
        if dir isa EveryDirection
            aux.∇a_exact = SVector(2 * x - y, 3 * y^2 - x, 0)
        elseif dir isa HorizontalDirection
            aux.∇a_exact = SVector(2 * x - y, 0, 0)
        elseif dir isa VerticalDirection
            aux.∇a_exact = SVector(0, 3 * y^2 - x, 0)
        end
    else
        aux.a = x^2 + y^3 + z^2 * y^2 - x * y * z
        if dir isa EveryDirection
            aux.∇a_exact = SVector(
                2 * x - y * z,
                3 * y^2 + 2 * z^2 * y - x * z,
                2 * z * y^2 - x * y,
            )
        elseif dir isa HorizontalDirection
            aux.∇a_exact =
                SVector(2 * x - y * z, 3 * y^2 + 2 * z^2 * y - x * z, 0)
        elseif dir isa VerticalDirection
            aux.∇a_exact = SVector(0, 0, 2 * z * y^2 - x * y)
        end
    end
end

using Test
function run(mpicomm, dim, direction, Ne, N, FT, ArrayType)

    brickrange = ntuple(j -> range(FT(0); length = Ne[j] + 1, stop = 3), dim)
    topl = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = ntuple(j -> true, dim),
    )

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    model = GradTestModel{dim, direction}()
    dg = DGModel(
        model,
        grid,
        nothing,
        nothing,
        nothing;
        state_gradient_flux = nothing,
    )

    continuous_field_gradient!(
        model,
        dg.state_auxiliary,
        (:∇a,),
        dg.state_auxiliary,
        (:a,),
        grid,
        direction,
    )

    # Wrapping in Array ensure both GPU and CPU code use same approx
    @test Array(dg.state_auxiliary.∇a) ≈ Array(dg.state_auxiliary.∇a_exact)
end

let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    numelem = (5, 5, 1)
    lvls = 1
    polynomialorder = 4

    @testset for FT in (Float64, Float32)
        @testset for dim in 2:3
            @testset for direction in (
                EveryDirection(),
                HorizontalDirection(),
                VerticalDirection(),
            )
                for l in 1:lvls
                    run(
                        mpicomm,
                        dim,
                        direction,
                        ntuple(j -> 2^(l - 1) * numelem[j], dim),
                        polynomialorder,
                        FT,
                        ArrayType,
                    )
                end
            end
        end
    end
end

nothing
