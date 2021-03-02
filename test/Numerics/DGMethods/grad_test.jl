using MPI
using StaticArrays
using ClimateMachine
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.MPIStateArrays
using ClimateMachine.DGMethods
using Printf

using ClimateMachine.BalanceLaws

import ClimateMachine.BalanceLaws: vars_state, nodal_init_state_auxiliary!

using ClimateMachine.Mesh.Geometry: LocalGeometry

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

struct GradTestModel{dim, dir} <: BalanceLaw end

vars_state(m::GradTestModel, ::Auxiliary, T) = @vars begin
    a::T
    ∇a::SVector{3, T}
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
            aux.∇a = SVector(2 * x - y, 3 * y^2 - x, 0)
        elseif dir isa HorizontalDirection
            aux.∇a = SVector(2 * x - y, 0, 0)
        elseif dir isa VerticalDirection
            aux.∇a = SVector(0, 3 * y^2 - x, 0)
        end
    else
        aux.a = x^2 + y^3 + z^2 * y^2 - x * y * z
        if dir isa EveryDirection
            aux.∇a = SVector(
                2 * x - y * z,
                3 * y^2 + 2 * z^2 * y - x * z,
                2 * z * y^2 - x * y,
            )
        elseif dir isa HorizontalDirection
            aux.∇a = SVector(2 * x - y * z, 3 * y^2 + 2 * z^2 * y - x * z, 0)
        elseif dir isa VerticalDirection
            aux.∇a = SVector(0, 0, 2 * z * y^2 - x * y)
        end
    end
end

using Test
function test_run(mpicomm, dim, direction, Ne, N, FT, ArrayType)
    connectivity = dim == 3 ? :full : :face
    brickrange = ntuple(j -> range(FT(0); length = Ne[j] + 1, stop = 3), dim)
    topl = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = ntuple(j -> false, dim),
        connectivity = connectivity,
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

    exact_aux = copy(dg.state_auxiliary)

    auxiliary_field_gradient!(
        model,
        dg.state_auxiliary,
        (:∇a,),
        dg.state_auxiliary,
        (:a,),
        grid,
        direction,
    )

    # Wrapping in Array ensure both GPU and CPU code use same approx
    approx = Array(dg.state_auxiliary.∇a) ≈ Array(exact_aux.∇a)
    err = euclidean_distance(exact_aux, dg.state_auxiliary)
    return approx, err
end

let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    numelem = (5, 5, 5)

    expected_result = Dict()
    expected_result[Float64, 2, 1] = 6.2135207410935696e+00
    expected_result[Float64, 2, 2] = 2.3700094936518794e+00
    expected_result[Float64, 2, 3] = 8.7105082013050261e-01
    expected_result[Float64, 2, 4] = 3.1401219279927611e-01

    expected_result[Float64, 3, 1] = 7.9363467666175236e+00
    expected_result[Float64, 3, 2] = 2.8059223082616098e+00
    expected_result[Float64, 3, 3] = 9.9204334582727094e-01
    expected_result[Float64, 3, 4] = 3.5074028853276679e-01

    expected_result[Float32, 2, 1] = 6.2135176658630371e+00
    expected_result[Float32, 2, 2] = 2.3700129985809326e+00
    expected_result[Float32, 3, 1] = 7.9363408088684082e+00
    expected_result[Float32, 3, 2] = 2.8059237003326416e+00


    @testset for FT in (Float64, Float32)
        lvls =
            integration_testing || ClimateMachine.Settings.integration_testing ?
            (FT === Float32 ? 2 : 4) : 1

        @testset for polynomialorder in ((4, 4), (4, 0))
            @testset for dim in 2:3
                @testset for direction in (
                    EveryDirection(),
                    HorizontalDirection(),
                    VerticalDirection(),
                )
                    err = zeros(FT, lvls)
                    for l in 1:lvls
                        approx, err[l] = test_run(
                            mpicomm,
                            dim,
                            direction,
                            ntuple(j -> 2^(l - 1) * numelem[j], dim),
                            polynomialorder,
                            FT,
                            ArrayType,
                        )
                        if polynomialorder[end] != 0 ||
                           direction isa HorizontalDirection
                            @test approx
                        else
                            @test err[l] ≈ expected_result[FT, dim, l]
                        end
                    end
                    if polynomialorder[end] != 0 ||
                       direction isa HorizontalDirection
                        @info begin
                            msg = "Polynomial order = $polynomialorder, direction = $direction\n"
                            for l in 1:(lvls - 1)
                                rate = log2(err[l]) - log2(err[l + 1])
                                msg *= @sprintf(
                                    "\n  rate for level %d = %e\n",
                                    l,
                                    rate
                                )
                            end
                            msg
                        end
                    end
                end
            end
        end
    end
end

nothing
