using .NumericalFluxes: numerical_volume_flux_first_order!

# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _ξ1x1, _ξ2x1, _ξ3x1 = Grids._ξ1x1, Grids._ξ2x1, Grids._ξ3x1
const _ξ1x2, _ξ2x2, _ξ3x2 = Grids._ξ1x2, Grids._ξ2x2, Grids._ξ3x2
const _ξ1x3, _ξ2x3, _ξ3x3 = Grids._ξ1x3, Grids._ξ2x3, Grids._ξ3x3
const _M = Grids._M

const _n1, _n2, _n3 = Grids._n1, Grids._n2, Grids._n3
const _sM, _vMI = Grids._sM, Grids._vMI
# }}}

@doc """
    esdg_volume_tendency!(
        balance_law::BalanceLaw,
        ::Val{dim},
        ::Val{polyorder},
        tendency,
        state_prognostic,
        state_auxiliary,
        vgeo,
        D,
        α,
        β,
    ) where {dim, polyorder}

Computes the reference element horizontal tendency using a two-point flux
approximation of all derivatives.
"""
@kernel function esdg_volume_tendency!(
    balance_law::BalanceLaw,
    ::Val{dir},
    ::Val{info},
    volume_numerical_flux_first_order,
    tendency,
    state_prognostic,
    state_auxiliary,
    vgeo,
    D,
    α,
    β,
    add_source = false,
) where {dir, info}
    @uniform begin
        dim = info.dim

        FT = eltype(state_prognostic)
        num_state = number_states(balance_law, Prognostic(), FT)
        num_aux = number_states(balance_law, Auxiliary(), FT)

        local_H = MArray{Tuple{3, num_state}, FT}(undef)

        state_2 = MArray{Tuple{num_state}, FT}(undef)
        aux_2 = MArray{Tuple{num_aux}, FT}(undef)

        local_source = MArray{Tuple{num_state}, FT}(undef)

        @inbounds Nq1 = info.Nq[1]
        @inbounds Nq2 = info.Nq[2]
        Nq3 = info.Nqk

        if dir == 1
            _ξdx1, _ξdx2, _ξdx3 = _ξ1x1, _ξ1x2, _ξ1x3
            Nqd = Nq1
        elseif dir == 2
            _ξdx1, _ξdx2, _ξdx3 = _ξ2x1, _ξ2x2, _ξ2x3
            Nqd = Nq2
        elseif dir == 3
            _ξdx1, _ξdx2, _ξdx3 = _ξ3x1, _ξ3x2, _ξ3x3
            Nqd = Nq3
        end
    end

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    state_1 = @private FT (num_state,)
    aux_1 = @private FT (num_aux,)
    local_tendency = @private FT (num_state,)
    local_MI = @private FT (1,)
    shared_G = @localmem FT (Nq1 * Nq2 * Nq3, 3)
    shared_D = @localmem FT (Nqd, Nqd)

    @inbounds begin
        ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

        # generalization for different polynomial orders
        @unroll for l in 1:Nqd
            if dir == 1
                id = i
                ild = l + Nq1 * ((j - 1) + Nq2 * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq1 * ((l - 1) + Nq2 * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq1 * ((j - 1) + Nq2 * (l - 1))
            end
            shared_D[id, l] = D[id, l]
        end

        M = vgeo[ijk, _M, e]
        shared_G[ijk, 1] = M * vgeo[ijk, _ξdx1, e]
        shared_G[ijk, 2] = M * vgeo[ijk, _ξdx2, e]
        if dim == 3
            shared_G[ijk, 3] = M * vgeo[ijk, _ξdx3, e]
        end

        # Build ode scaling into mass matrix (so it doesn't show up later)
        local_MI[1] = α / M

        # Get the volume tendency (scaling by β done below)
        @unroll for s in 1:num_state
            local_tendency[s] = β != 0 ? tendency[ijk, s, e] : -zero(FT)
        end

        # Scale β into the volume tendency
        @unroll for s in 1:num_state
            local_tendency[s] *= β
        end

        # Compute the volume tendency
        # ∑_{i,j}^{d} ( G_ij (Q_i ∘ H_j) - (H_j ∘ Q_i^T) G_ij)
        #  =
        # ( G_11 (Q_1 ∘ H_1) - (H_1 ∘ Q_1^T) G_11) 1 +
        # ( G_12 (Q_1 ∘ H_2) - (H_2 ∘ Q_1^T) G_12) 1 +
        # ( G_13 (Q_1 ∘ H_3) - (H_3 ∘ Q_1^T) G_13) 1 +
        # ( G_21 (Q_2 ∘ H_1) - (H_1 ∘ Q_2^T) G_21) 1 +
        # ( G_22 (Q_2 ∘ H_2) - (H_2 ∘ Q_2^T) G_22) 1 +
        # ( G_23 (Q_2 ∘ H_3) - (H_3 ∘ Q_2^T) G_23) 1 +
        # ( G_31 (Q_3 ∘ H_1) - (H_1 ∘ Q_3^T) G_31) 1 +
        # ( G_32 (Q_3 ∘ H_2) - (H_2 ∘ Q_3^T) G_32) 1 +
        # ( G_33 (Q_3 ∘ H_3) - (H_3 ∘ Q_3^T) G_33) 1
        @unroll for s in 1:num_state
            state_1[s] = state_prognostic[ijk, s, e]
        end
        @unroll for s in 1:num_aux
            aux_1[s] = state_auxiliary[ijk, s, e]
        end

        if add_source
            fill!(local_source, -zero(eltype(local_source)))
            source!(
                balance_law,
                Vars{vars_state(balance_law, Prognostic(), FT)}(local_source),
                Vars{vars_state(balance_law, Prognostic(), FT)}(state_1),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(aux_1),
            )

            @unroll for s in 1:num_state
                local_tendency[s] += α * local_source[s]
            end
        end

        @synchronize

        ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

        # Note: unrolling this loop makes things slower
        @views for l in 1:Nqd
            if dir == 1
                id = i
                ild = l + Nq1 * ((j - 1) + Nq2 * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq1 * ((l - 1) + Nq2 * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq1 * ((j - 1) + Nq2 * (l - 1))
            end

            # Compute derivatives wrt ξd
            # ( G_31 (Q_3 ∘ H_1) - (H_1 ∘ Q_3^T) G_31) 1 +
            # ( G_32 (Q_3 ∘ H_2) - (H_2 ∘ Q_3^T) G_32) 1 +
            # ( G_33 (Q_3 ∘ H_3) - (H_3 ∘ Q_3^T) G_33) 1 +
            @unroll for s in 1:num_state
                state_2[s] = state_prognostic[ild, s, e]
            end
            @unroll for s in 1:num_aux
                aux_2[s] = state_auxiliary[ild, s, e]
            end
            fill!(local_H, -zero(FT))
            numerical_volume_flux_first_order!(
                volume_numerical_flux_first_order,
                balance_law,
                local_H,
                state_1[:],
                aux_1[:],
                state_2,
                aux_2,
            )
            @unroll for s in 1:num_state
                # G_21 (Q_2 ∘ H_1) 1 +
                # G_22 (Q_2 ∘ H_2) 1 +
                # G_23 (Q_2 ∘ H_3) 1
                local_tendency[s] -=
                    local_MI[1] *
                    shared_D[id, l] *
                    (
                        shared_G[ijk, 1] * local_H[1, s] +
                        shared_G[ijk, 2] * local_H[2, s] +
                        (
                            dim == 3 ? shared_G[ijk, 3] * local_H[3, s] :
                            -zero(FT)
                        )
                    )
                #  (H_1 ∘ Q_2^T) G_21 1 +
                #  (H_2 ∘ Q_2^T) G_22 1 +
                #  (H_3 ∘ Q_2^T) G_23 1
                local_tendency[s] +=
                    local_MI[1] *
                    (
                        local_H[1, s] * shared_G[ild, 1] +
                        local_H[2, s] * shared_G[ild, 2] +
                        (
                            dim == 3 ? local_H[3, s] * shared_G[ild, 3] :
                            -zero(FT)
                        )
                    ) *
                    shared_D[l, id]
            end
        end

        @unroll for s in 1:num_state
            tendency[ijk, s, e] = local_tendency[s]
        end
    end
end
