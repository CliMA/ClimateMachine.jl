using MPI
using StaticArrays
using LinearAlgebra
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

struct GradSphereTestModel{dir} <: BalanceLaw end

vars_state(m::GradSphereTestModel, ::Auxiliary, T) = @vars begin
    a::T
    ∇a::SVector{3, T}
end
vars_state(::GradSphereTestModel, ::Prognostic, T) = @vars()

function nodal_init_state_auxiliary!(
    ::GradSphereTestModel{dir},
    aux::Vars,
    tmp::Vars,
    g::LocalGeometry,
) where {dir}
    x, y, z = g.coord
    r = hypot(x, y, z)
    aux.a = r^3
    if !(dir isa HorizontalDirection)
        aux.∇a = 3 * r^2 * g.coord / r
    else
        aux.∇a = SVector(0, 0, 0)
    end
end

using Test
function test_run(mpicomm, Ne_horz, Ne_vert, N, FT, ArrayType, direction)

    Rrange = range(FT(1 // 2); length = Ne_vert + 1, stop = FT(1))
    topl = StackedCubedSphereTopology(mpicomm, Ne_horz, Rrange)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
        meshwarp = Topologies.cubedshellwarp,
    )

    model = GradSphereTestModel{direction}()
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

    err = euclidean_distance(exact_aux, dg.state_auxiliary)
    return err
end

let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    base_Nhorz = 4
    base_Nvert = 2

    expected_result = Dict()
    expected_result[(4, 4), 1] = 4.3759489495202896e-04
    expected_result[(4, 4), 2] = 2.9065372851175251e-05
    expected_result[(4, 4), 3] = 1.8457379514995729e-06
    expected_result[(4, 4), 4] = 1.1582093840084037e-07

    expected_result[(4, 0), 1] = 1.1070045305138052e+00
    expected_result[(4, 0), 2] = 4.2750547196265593e-01
    expected_result[(4, 0), 3] = 1.6041478911787385e-01
    expected_result[(4, 0), 4] = 5.8570590697850776e-02

    lvls =
        integration_testing || ClimateMachine.Settings.integration_testing ? 4 :
        1

    @testset for FT in (Float64,)
        @testset for polynomialorder in ((4, 4), (4, 0))
            @testset for direction in (
                EveryDirection(),
                HorizontalDirection(),
                VerticalDirection(),
            )
                err = zeros(FT, lvls)
                @testset for l in 1:lvls
                    Ne_horz = 2^(l - 1) * base_Nhorz
                    Ne_vert = 2^(l - 1) * base_Nvert

                    err[l] = test_run(
                        mpicomm,
                        Ne_horz,
                        Ne_vert,
                        polynomialorder,
                        FT,
                        ArrayType,
                        direction,
                    )
                    if !(direction isa HorizontalDirection)
                        @test err[l] ≈ expected_result[polynomialorder, l]
                    else
                        @test abs(err[l]) < FT(1.3e-13)
                    end
                end
                if !(direction isa HorizontalDirection)
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

nothing
