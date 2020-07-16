using MPI
using StaticArrays
using ClimateMachine
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.MPIStateArrays
using ClimateMachine.DGMethods

using ClimateMachine.BalanceLaws: BalanceLaw

import ClimateMachine.BalanceLaws:
    vars_state_auxiliary, vars_state_conservative, init_state_auxiliary!

using ClimateMachine.Mesh.Geometry: LocalGeometry

struct GradTestModel{dim} <: BalanceLaw end

vars_state_auxiliary(m::GradTestModel, T) = @vars begin
    a::T
    ∇a::SVector{3, T}
    ∇a_exact::SVector{3, T}
end
vars_state_conservative(::GradTestModel, T) = @vars()

function grad_nodal_init_state_auxiliary!(
    ::GradTestModel{dim},
    aux::Vars,
    tmp::Vars,
    g::LocalGeometry,
) where {dim}
    x, y, z = g.coord
    if dim == 2
        aux.a = x^2 + y^3 - x * y
        aux.∇a_exact = SVector(2 * x - y, 3 * y^2 - x, 0)
    else
        aux.a = x^2 + y^3 + z^2 * y^2 - x * y * z
        aux.∇a_exact = SVector(
            2 * x - y * z,
            3 * y^2 + 2 * z^2 * y - x * z,
            2 * z * y^2 - x * y,
        )
    end
end

function init_state_auxiliary!(
    m::GradTestModel,
    state_auxiliary::MPIStateArray,
    grid,
)
    nodal_init_state_auxiliary!(
        m,
        grad_nodal_init_state_auxiliary!,
        state_auxiliary,
        grid,
    )
end

using Test
function run(mpicomm, dim, Ne, N, FT, ArrayType)

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

    model = GradTestModel{dim}()
    dg = DGModel(
        model,
        grid,
        nothing,
        nothing,
        nothing;
        state_gradient_flux = nothing,
    )

    contiguous_field_gradient!(
        model,
        dg.state_auxiliary,
        (:∇a,),
        dg.state_auxiliary,
        (:a,),
        grid,
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
            for l in 1:lvls
                run(
                    mpicomm,
                    dim,
                    ntuple(j -> 2^(l - 1) * numelem[j], dim),
                    polynomialorder,
                    FT,
                    ArrayType,
                )
            end
        end
    end
end

nothing
