module TestVTK

using Test
using GaussQuadrature: legendre, both
using ClimateMachine.VTK: writemesh_highorder, writemesh_raw

@testset "VTK" begin
    for writemesh in (writemesh_highorder, writemesh_raw)
        Ns = writemesh == writemesh_raw ? ((4, 4, 4), (3, 5, 7)) : ((4, 4, 4),)
        for N in Ns
            for dim in 1:3
                nelem = 3
                T = Float64

                Nq = N .+ 1
                r = legendre(T, Nq[1], both)[1]
                s = dim < 2 ? [0] : legendre(T, Nq[2], both)[1]
                t = dim < 3 ? [0] : legendre(T, Nq[3], both)[1]
                Nq1 = length(r)
                Nq2 = length(s)
                Nq3 = length(t)
                Np = Nq1 * Nq2 * Nq3

                x1 = Array{T, 4}(undef, Nq1, Nq2, Nq3, nelem)
                x2 = Array{T, 4}(undef, Nq1, Nq2, Nq3, nelem)
                x3 = Array{T, 4}(undef, Nq1, Nq2, Nq3, nelem)

                for e in 1:nelem, k in 1:Nq3, j in 1:Nq2, i in 1:Nq1
                    xoffset = nelem + 1 - 2e
                    x1[i, j, k, e], x2[i, j, k, e], x3[i, j, k, e] =
                        r[i] - xoffset, s[j], t[k]
                end

                if dim == 1
                    x1 = x1 .^ 3
                elseif dim == 2
                    x1, x2 = x1 + sin.(π * x2) / 5, x2 + exp.(-x1 .^ 2)
                else
                    x1, x2, x3 = x1 + sin.(π * x2) / 5,
                    x2 + exp.(-hypot.(x1, x3) .^ 2),
                    x3 + sin.(π * x1) / 5
                end
                d = exp.(sin.(hypot.(x1, x2, x3)))
                s = copy(d)

                if dim == 1
                    @test "test$(dim)d.vtu" == writemesh(
                        "test$(dim)d",
                        x1;
                        fields = (("d", d), ("s", s)),
                    )[1]
                    @test "test$(dim)d.vtu" == writemesh(
                        "test$(dim)d",
                        x1;
                        x2 = x2,
                        fields = (("d", d), ("s", s)),
                    )[1]
                    @test "test$(dim)d.vtu" == writemesh(
                        "test$(dim)d",
                        x1;
                        x2 = x2,
                        x3 = x3,
                        fields = (("d", d), ("s", s)),
                    )[1]
                elseif dim == 2
                    @test "test$(dim)d.vtu" == writemesh(
                        "test$(dim)d",
                        x1,
                        x2;
                        fields = (("d", d), ("s", s)),
                    )[1]
                    @test "test$(dim)d.vtu" == writemesh(
                        "test$(dim)d",
                        x1,
                        x2;
                        x3 = x3,
                        fields = (("d", d), ("s", s)),
                    )[1]
                elseif dim == 3
                    @test "test$(dim)d.vtu" == writemesh(
                        "test$(dim)d",
                        x1,
                        x2,
                        x3;
                        fields = (("d", d), ("s", s)),
                    )[1]
                end
            end
        end
    end
end


using MPI
MPI.Initialized() || MPI.Init()
using ClimateMachine.Mesh.Topologies: BrickTopology
using ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid
using ClimateMachine.VTK: writevtk_helper

let
    mpicomm = MPI.COMM_SELF
    for FT in (Float64,) #Float32)
        for dim in 2:3
            for _N in ((2, 3, 4), (0, 2, 5), (3, 0, 0), (0, 0, 0))
                N = _N[1:dim]
                if dim == 2
                    Ne = (4, 5)
                    brickrange = (
                        range(FT(0); length = Ne[1] + 1, stop = 1),
                        range(FT(0); length = Ne[2] + 1, stop = 1),
                    )
                    topl = BrickTopology(
                        mpicomm,
                        brickrange,
                        periodicity = (false, false),
                        connectivity = :face,
                    )
                    warpfun =
                        (x1, x2, _) -> begin
                            (x1 + sin(x1 * x2), x2 + sin(2 * x1 * x2), 0)
                        end
                elseif dim == 3
                    Ne = (3, 4, 5)
                    brickrange = (
                        range(FT(0); length = Ne[1] + 1, stop = 1),
                        range(FT(0); length = Ne[2] + 1, stop = 1),
                        range(FT(0); length = Ne[3] + 1, stop = 1),
                    )
                    topl = BrickTopology(
                        mpicomm,
                        brickrange,
                        periodicity = (false, false, false),
                        connectivity = :face,
                    )
                    warpfun =
                        (x1, x2, x3) -> begin
                            (
                                x1 + (x1 - 1 / 2) * cos(2 * π * x2 * x3) / 4,
                                x2 + exp(sin(2π * (x1 * x2 + x3))) / 20,
                                x3 + x1 / 4 + x2^2 / 2 + sin(x1 * x2 * x3),
                            )
                        end
                end
                grid = DiscontinuousSpectralElementGrid(
                    topl,
                    FloatType = FT,
                    DeviceArray = Array,
                    polynomialorder = N,
                    meshwarp = warpfun,
                )
                Q = rand(FT, prod(N .+ 1), 3, prod(Ne))
                prefix = "test$(dim)d_raw$(prod(ntuple(i->"_$(N[i])", dim)))"
                @test "$(prefix).vtu" == writevtk_helper(
                    prefix,
                    grid.vgeo,
                    Q,
                    grid,
                    ("a", "b", "c");
                    number_sample_points = 0,
                )[1]
                prefix = "test$(dim)d_high_order$(prod(ntuple(i->"_$(N[i])", dim)))"
                @test "$(prefix).vtu" == writevtk_helper(
                    prefix,
                    grid.vgeo,
                    Q,
                    grid,
                    ("a", "b", "c");
                    number_sample_points = 10,
                )[1]
            end
        end
    end
end

end #module TestVTK
