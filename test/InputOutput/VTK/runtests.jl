using Test
using GaussQuadrature
using ClimateMachine.VTK: writemesh_highorder, writemesh_raw

@testset "VTK" begin
    for writemesh in (writemesh_highorder, writemesh_raw)
        for dim in 1:3
            N = 5
            nelem = 3
            T = Float64

            Nq = N + 1
            (r, _) = GaussQuadrature.legendre(T, Nq, GaussQuadrature.both)
            s = dim > 1 ? r : [0]
            t = dim > 2 ? r : [0]
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
