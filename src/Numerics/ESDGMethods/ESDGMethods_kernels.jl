# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _ξ1x1, _ξ2x1, _ξ3x1 = Grids._ξ1x1, Grids._ξ2x1, Grids._ξ3x1
const _ξ1x2, _ξ2x2, _ξ3x2 = Grids._ξ1x2, Grids._ξ2x2, Grids._ξ3x2
const _ξ1x3, _ξ2x3, _ξ3x3 = Grids._ξ1x3, Grids._ξ2x3, Grids._ξ3x3
const _M = Grids._M

const _n1, _n2, _n3 = Grids._n1, Grids._n2, Grids._n3
const _sM, _vMI = Grids._sM, Grids._vMI
# }}}

@doc """
    volume_tendency!(balance_law::BalanceLaw, Val(polyorder),
                     tendency, state_conservative, state_gradient_flux,
                     state_auxiliary, vgeo, t, D, elems)

Computational kernel: Evaluate the volume integrals on right-hand side of a
`BalanceLaw` semi-discretization.
""" volume_tendency!
@kernel function volume_tendency!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    tendency,
    state_conservative,
    state_auxiliary,
    vgeo,
    D,
    α,
    β,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(state_conservative)
        num_state = number_state_conservative(balance_law, FT)
        num_aux = number_state_auxiliary(balance_law, FT)

        local_H = MArray{Tuple{3, num_state}, FT}(undef)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq
    end

    # FIXME: We're using a lot of shared memory!
    shared_state = @localmem FT (2, Nq, Nq, num_state)
    # FIXME: Needs to become a partial aux (not full aux)
    shared_aux = @localmem FT (2, Nq, Nq, num_aux)
    G = @localmem FT (Nq, Nq, 2, dim)

    s_D = @localmem FT (Nq, Nq)

    local_tendency = @private FT (num_state, )
    local_MI = @private FT (1, )

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        # Load derivative into shared memory
        s_D[i, j] = D[i, j]

        # Nqk - number of horizontal slices
        @unroll for k in 1:Nqk
            @synchronize
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))

            # Store one horizontal slabs state and aux into shared
            @unroll for s in 1:num_state
                shared_state[i, j, s] = state_conservative[ijk, s, e]
            end
            @unroll for s in 1:num_aux
                shared_aux[i, j, s] = state_auxiliary[ijk, s, e]
            end
            @unroll for s in 1:num_state
                local_tendency[s] = β != 0 ? tendency[ijk, s, e] : -zero(FT)
            end

            M = vgeo[ijk, _M, e]
            ξ1x1 = vgeo[ijk, _ξ1x1, e]
            ξ1x2 = vgeo[ijk, _ξ1x2, e]
            ξ1x3 = vgeo[ijk, _ξ1x3, e]
            ξ2x1 = vgeo[ijk, _ξ2x1, e]
            ξ2x2 = vgeo[ijk, _ξ2x2, e]
            ξ2x3 = vgeo[ijk, _ξ2x3, e]
            G[i, j, 1, 1] = M * ξ1x1
            G[i, j, 1, 2] = M * ξ1x2
            G[i, j, 1, 3] = M * ξ1x3
            G[i, j, 2, 1] = M * ξ2x1
            G[i, j, 2, 2] = M * ξ2x2
            G[i, j, 2, 3] = M * ξ2x3
            # Build ode scaling into mass matrix (so it doesn't show up later)
            local_MI[1] = α / M

            @synchronize

            @unroll for s in 1:num_state
                local_tendency[s] *= β
            end

            # ∑_{i,j}^{d} ( G_ij (Q_i ∘ H_j) - (H_j ∘ Q_i^T) G_ij)
            #  =
            # ( G_11 (Q_1 ∘ H_1) - (H_1 ∘ Q_1^T) G_11) 1 +
            # ( G_12 (Q_1 ∘ H_2) - (H_2 ∘ Q_1^T) G_12) 1 +
            # ( G_13 (Q_1 ∘ H_3) - (H_3 ∘ Q_1^T) G_13) 1 +
            # ( G_21 (Q_2 ∘ H_1) - (H_1 ∘ Q_2^T) G_21) 1 +
            # ( G_22 (Q_2 ∘ H_2) - (H_2 ∘ Q_2^T) G_22) 1 +
            # ( G_23 (Q_2 ∘ H_3) - (H_3 ∘ Q_2^T) G_23) 1 +
            # {SKIP} ( G_31 (Q_3 ∘ H_1) - (H_1 ∘ Q_3^T) G_31) 1 +
            # {SKIP} ( G_32 (Q_3 ∘ H_2) - (H_2 ∘ Q_3^T) G_32) 1 +
            # {SKIP} ( G_33 (Q_3 ∘ H_3) - (H_3 ∘ Q_3^T) G_33) 1
            @views for l in 1:Nq
                # Compute derivatives wrt ξ1
                # ( G_11 (Q_1 ∘ H_1) - (H_1 ∘ Q_1^T) G_11) 1 +
                # ( G_12 (Q_1 ∘ H_2) - (H_2 ∘ Q_1^T) G_12) 1 +
                # ( G_13 (Q_1 ∘ H_3) - (H_3 ∘ Q_1^T) G_13) 1 +
                # FIXME: We may want to use local arrays here
                #        (not shared arrays)
                numerical_volume_fluctuation!(local_H,
                  shared_state[i, j, :], shared_aux[i, j, :],
                  shared_state[l, j, :], shared_aux[l, j, :]
                )
                for s in 1:num_state
                    # G_11 (Q_1 ∘ H_1) 1 +
                    # G_12 (Q_1 ∘ H_2) 1 +
                    # G_13 (Q_1 ∘ H_3) 1
                    local_tendency[s] += local_MI[1] * s_D[i, l] * (
                       G[i, j, 1, 1] * local_H[1, s] +
                       G[i, j, 1, 2] * local_H[1, s] +
                       G[i, j, 1, 3] * local_H[3, s]
                    )
                    #  (H_1 ∘ Q_1^T) G_11 1 +
                    #  (H_2 ∘ Q_1^T) G_12 1 +
                    #  (H_3 ∘ Q_1^T) G_13 1
                    local_tendency[s] -= local_MI[1] * (
                       local_H[1, s] * G[l, j, 1, 1] +
                       local_H[2, s] * G[l, j, 1, 2] +
                       local_H[3, s] * G[l, j, 1, 3]
                    ) * s_D[l, i]
                end

                # Compute derivatives wrt ξ2
                # ( G_21 (Q_2 ∘ H_1) - (H_1 ∘ Q_2^T) G_21) 1 +
                # ( G_22 (Q_2 ∘ H_2) - (H_2 ∘ Q_2^T) G_22) 1 +
                # ( G_23 (Q_2 ∘ H_3) - (H_3 ∘ Q_2^T) G_23) 1 +
                numerical_volume_fluctuation!(local_H,
                 shared_state[i, j, :], shared_aux[i, j, :],
                 shared_state[i, l, :], shared_aux[i, l, :]
                )
                for s in 1:num_state
                    # G_21 (Q_2 ∘ H_1) 1 +
                    # G_22 (Q_2 ∘ H_2) 1 +
                    # G_23 (Q_2 ∘ H_3) 1
                    local_tendency[s] += local_MI[1] * s_D[i, l] * (
                       G[i, j, 2, 1] * local_H[1, s] +
                       G[i, j, 2, 2] * local_H[1, s] +
                       G[i, j, 2, 3] * local_H[3, s]
                    )
                    #  (H_1 ∘ Q_2^T) G_21 1 +
                    #  (H_2 ∘ Q_2^T) G_22 1 +
                    #  (H_3 ∘ Q_2^T) G_23 1
                    local_tendency[s] -= local_MI[1] * (
                       local_H[1, s] * G[i, l, 2, 1] +
                       local_H[2, s] * G[i, l, 2, 2] +
                       local_H[3, s] * G[i, l, 2, 3]
                    ) * s_D[l, i]
                end

                ijk = i + Nq * ((j - 1) + Nq * (k - 1))
                @unroll for s in 1:num_state
                    tendency[ijk, s, e] = local_tendency[s]
                end
            end
        end
    end
end