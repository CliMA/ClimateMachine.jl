using Requires
using ..Mesh.Grids: EveryDirection, VerticalDirection, HorizontalDirection
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative
end
using KernelAbstractions.Extras: @unroll

const _M = Grids._M

@doc """
    knl_apply_filter!(::Val{dim}, ::Val{N}, ::Val{nstate}, ::Val{direction},
                      Q, ::Val{states}, filtermatrix,
                      elems) where {dim, N, nstate, states, direction}

Computational kernel: Applies the `filtermatrix` to the `states` of `Q`.

The `direction` argument is used to control if the filter is applied in the
horizontal and/or vertical reference directions.
""" knl_apply_filter!
@kernel function knl_apply_filter!(
    ::Val{dim},
    ::Val{N},
    ::Val{nstate},
    ::Val{direction},
    Q,
    ::Val{states},
    filtermatrix,
    elems,
) where {dim, N, nstate, direction, states}
    @uniform begin
        FT = eltype(Q)

        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq

        if direction isa EveryDirection
            filterinξ1 = filterinξ2 = filterinξ3 = true
        elseif direction isa HorizontalDirection
            filterinξ1 = true
            filterinξ2 = dim == 2 ? false : true
            filterinξ3 = false
        elseif direction isa VerticalDirection
            filterinξ1 = false
            filterinξ2 = dim == 2 ? true : false
            filterinξ3 = dim == 2 ? false : true
        end

        nfilterstates = length(states)
    end

    s_filter = @localmem FT (Nq, Nq)
    s_Q = @localmem FT (Nq, Nq, Nqk, nfilterstates)
    l_Qfiltered = @private FT (nfilterstates,)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        s_filter[i, j] = filtermatrix[i, j]

        @unroll for fs in 1:nfilterstates
            l_Qfiltered[fs] = zero(FT)
        end

        ijk = i + Nq * ((j - 1) + Nq * (k - 1))

        @unroll for fs in 1:nfilterstates
            s_Q[i, j, k, fs] = Q[ijk, states[fs], e]
        end

        if filterinξ1
            @synchronize
            @unroll for n in 1:Nq
                @unroll for fs in 1:nfilterstates
                    l_Qfiltered[fs] += s_filter[i, n] * s_Q[n, j, k, fs]
                end
            end

            if filterinξ2 || filterinξ3
                @synchronize
                @unroll for fs in 1:nfilterstates
                    s_Q[i, j, k, fs] = l_Qfiltered[fs]
                    l_Qfiltered[fs] = zero(FT)
                end
            end
        end

        if filterinξ2
            @synchronize
            @unroll for n in 1:Nq
                @unroll for fs in 1:nfilterstates
                    l_Qfiltered[fs] += s_filter[j, n] * s_Q[i, n, k, fs]
                end
            end

            if filterinξ3
                @synchronize
                @unroll for fs in 1:nfilterstates
                    s_Q[i, j, k, fs] = l_Qfiltered[fs]
                    l_Qfiltered[fs] = zero(FT)
                end
            end
        end

        if filterinξ3
            @synchronize
            @unroll for n in 1:Nq
                @unroll for fs in 1:nfilterstates
                    l_Qfiltered[fs] += s_filter[k, n] * s_Q[i, j, n, fs]
                end
            end
        end

        # Store result
        ijk = i + Nq * ((j - 1) + Nq * (k - 1))
        @unroll for fs in 1:nfilterstates
            Q[ijk, states[fs], e] = l_Qfiltered[fs]
        end

        @synchronize
    end
end

@kernel function knl_apply_TMAR_filter!(
    ::Val{nreduce},
    ::Val{dim},
    ::Val{N},
    Q,
    ::Val{filterstates},
    vgeo,
    elems,
) where {nreduce, dim, N, filterstates}
    @uniform begin
        FT = eltype(Q)

        Nq = N + 1
        Nqj = dim == 2 ? 1 : Nq

        nfilterstates = length(filterstates)
        nelemperblock = 1
    end

    l_Q = @private FT (nfilterstates, Nq)
    l_MJ = @private FT (Nq,)

    s_MJQ = @localmem FT (Nq * Nqj, nfilterstates)
    s_MJQclipped = @localmem FT (Nq * Nqj, nfilterstates)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        # loop up the pencil and load Q and MJ
        @unroll for k in 1:Nq
            ijk = i + Nq * ((j - 1) + Nqj * (k - 1))

            @unroll for sf in 1:nfilterstates
                s = filterstates[sf]
                l_Q[sf, k] = Q[ijk, s, e]
            end

            l_MJ[k] = vgeo[ijk, _M, e]
        end

        @unroll for sf in 1:nfilterstates
            MJQ, MJQclipped = zero(FT), zero(FT)

            @unroll for k in 1:Nq
                MJ = l_MJ[k]
                Qs = l_Q[sf, k]
                Qsclipped = Qs ≥ 0 ? Qs : zero(Qs)

                MJQ += MJ * Qs
                MJQclipped += MJ * Qsclipped
            end

            ij = i + Nq * (j - 1)

            s_MJQ[ij, sf] = MJQ
            s_MJQclipped[ij, sf] = MJQclipped
        end
        @synchronize

        @unroll for n in 11:-1:1
            if nreduce ≥ 2^n
                ij = i + Nq * (j - 1)
                ijshift = ij + 2^(n - 1)
                if ij ≤ 2^(n - 1) && ijshift ≤ Nq * Nqj
                    @unroll for sf in 1:nfilterstates
                        s_MJQ[ij, sf] += s_MJQ[ijshift, sf]
                        s_MJQclipped[ij, sf] += s_MJQclipped[ijshift, sf]
                    end
                end
                @synchronize
            end
        end

        @unroll for sf in 1:nfilterstates
            qs_average = s_MJQ[1, sf]
            qs_clipped_average = s_MJQclipped[1, sf]

            r = qs_average > 0 ? qs_average / qs_clipped_average : zero(FT)

            s = filterstates[sf]
            @unroll for k in 1:Nq
                ijk = i + Nq * ((j - 1) + Nqj * (k - 1))

                Qs = l_Q[sf, k]
                Q[ijk, s, e] = Qs ≥ 0 ? r * Qs : zero(Qs)
            end
        end
    end
end
