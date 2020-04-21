using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative
end
using KernelAbstractions.Extras: @unroll

using StaticArrays

const _x1 = Grids._x1
const _x2 = Grids._x2
const _x3 = Grids._x3

@doc """
    knl_min_neighbor_distance!(::Val{N}, ::Val{dim}, direction,
                             min_neighbor_distance, vgeo, topology.realelems)

Computational kernel: Computes the minimum physical distance between node
neighbors within an element.

The `direction` in the reference element controls which nodes are considered
neighbors.
""" knl_min_neighbor_distance!
@kernel function knl_min_neighbor_distance!(
    ::Val{N},
    ::Val{dim},
    direction,
    min_neighbor_distance,
    vgeo,
    elems,
) where {N, dim}

    @uniform begin
        FT = eltype(min_neighbor_distance)
        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq

        if direction isa EveryDirection
            mininξ = (true, true, true)
        elseif direction isa HorizontalDirection
            mininξ = (true, dim == 2 ? false : true, false)
        elseif direction isa VerticalDirection
            mininξ = (false, dim == 2 ? true : false, dim == 2 ? false : true)
        end
    end

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    md = typemax(FT)

    ijk = i + Nq * (j - 1) + Nq * Nq * (k - 1)
    x = SVector(vgeo[ijk, _x1, e], vgeo[ijk, _x2, e], vgeo[ijk, _x3, e])

    if mininξ[1]
        @unroll for î in (i - 1, i + 1)
            if 1 ≤ î ≤ Nq
                îjk = î + Nq * (j - 1) + Nq * Nq * (k - 1)
                x̂ = SVector(
                    vgeo[îjk, _x1, e],
                    vgeo[îjk, _x2, e],
                    vgeo[îjk, _x3, e],
                )
                md = min(md, norm(x - x̂))
            end
        end
    end

    if mininξ[2]
        @unroll for ĵ in (j - 1, j + 1)
            if 1 ≤ ĵ ≤ Nq
                iĵk = i + Nq * (ĵ - 1) + Nq * Nq * (k - 1)
                x̂ = SVector(
                    vgeo[iĵk, _x1, e],
                    vgeo[iĵk, _x2, e],
                    vgeo[iĵk, _x3, e],
                )
                md = min(md, norm(x - x̂))
            end
        end
    end

    if mininξ[3]
        @unroll for k̂ in (k - 1, k + 1)
            if 1 ≤ k̂ ≤ Nqk
                ijk̂ = i + Nq * (j - 1) + Nq * Nq * (k̂ - 1)
                x̂ = SVector(
                    vgeo[ijk̂, _x1, e],
                    vgeo[ijk̂, _x2, e],
                    vgeo[ijk̂, _x3, e],
                )
                md = min(md, norm(x - x̂))
            end
        end
    end

    min_neighbor_distance[ijk, e] = md
end
