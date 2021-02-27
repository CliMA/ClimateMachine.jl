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
""" esdg_volume_tendency!
@kernel function esdg_volume_tendency!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    volume_numerical_flux_first_order,
    tendency,
    state_prognostic,
    state_auxiliary,
    vgeo,
    D,
    α,
    β,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(state_prognostic)
        num_state = number_states(balance_law, Prognostic(), FT)
        num_aux = number_states(balance_law, Auxiliary(), FT)

        local_H = MArray{Tuple{3, num_state}, FT}(undef)

        state_1 = MArray{Tuple{num_state}, FT}(undef)
        aux_1 = MArray{Tuple{num_aux}, FT}(undef)
        state_2 = MArray{Tuple{num_state}, FT}(undef)
        aux_2 = MArray{Tuple{num_aux}, FT}(undef)

        local_source = MArray{Tuple{num_state}, FT}(undef)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq
    end

    # FIXME: We're using a lot of shared memory!
    shared_state = @localmem FT (Nq, Nq, num_state)

    # FIXME: Needs to become a partial aux (not full aux)
    shared_aux = @localmem FT (Nq, Nq, num_aux)

    shared_G = @localmem FT (Nq, Nq, 2, 3)

    shared_D = @localmem FT (Nq, Nq)

    local_tendency = @private FT (num_state,)
    local_MI = @private FT (1,)

    pencil_state = @private FT (num_state, Nqk)
    pencil_aux = @private FT (num_aux, Nqk)
    pencil_G_3 = @private FT (3, Nqk)
    pencil_M = @private FT (Nqk,)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        # Load derivative into shared memory
        shared_D[i, j] = D[i, j]

        # Load data pencils
        @unroll for k in 1:Nqk
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            pencil_M[k] = vgeo[ijk, _M, e]
            pencil_G_3[1, k] = vgeo[ijk, _ξ3x1, e]
            pencil_G_3[2, k] = vgeo[ijk, _ξ3x2, e]
            pencil_G_3[3, k] = vgeo[ijk, _ξ3x3, e]
            @unroll for s in 1:num_state
                pencil_state[s, k] = state_prognostic[ijk, s, e]
            end
            @unroll for s in 1:num_aux
                pencil_aux[s, k] = state_auxiliary[ijk, s, e]
            end
            pencil_G_3[1, k] *= pencil_M[k]
            pencil_G_3[2, k] *= pencil_M[k]
            pencil_G_3[3, k] *= pencil_M[k]
        end

        # Loop up the elements slabs and apply the operators slab by slab
        @unroll for k in 1:Nqk
            @synchronize
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))

            # Load the geometry terms
            ξ1x1 = vgeo[ijk, _ξ1x1, e]
            ξ1x2 = vgeo[ijk, _ξ1x2, e]
            ξ1x3 = vgeo[ijk, _ξ1x3, e]
            ξ2x1 = vgeo[ijk, _ξ2x1, e]
            ξ2x2 = vgeo[ijk, _ξ2x2, e]
            ξ2x3 = vgeo[ijk, _ξ2x3, e]

            # Get the volume tendency (scaling by β done below)
            @unroll for s in 1:num_state
                local_tendency[s] = β != 0 ? tendency[ijk, s, e] : -zero(FT)
            end

            # load up the shared memory
            @unroll for s in 1:num_state
                shared_state[i, j, s] = pencil_state[s, k]
            end
            @unroll for s in 1:num_aux
                shared_aux[i, j, s] = pencil_aux[s, k]
            end

            # scale in the mass matrix and save to shared memory
            shared_G[i, j, 1, 1] = pencil_M[k] * ξ1x1
            shared_G[i, j, 1, 2] = pencil_M[k] * ξ1x2
            shared_G[i, j, 1, 3] = pencil_M[k] * ξ1x3
            shared_G[i, j, 2, 1] = pencil_M[k] * ξ2x1
            shared_G[i, j, 2, 2] = pencil_M[k] * ξ2x2
            shared_G[i, j, 2, 3] = pencil_M[k] * ξ2x3

            # Build ode scaling into mass matrix (so it doesn't show up later)
            local_MI[1] = α / pencil_M[k]

            # Scale β into the volume tendency
            @unroll for s in 1:num_state
                local_tendency[s] *= β
            end

            @synchronize

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
                state_1[s] = shared_state[i, j, s]
            end
            @unroll for s in 1:num_aux
                aux_1[s] = shared_aux[i, j, s]
            end

            fill!(local_source, -zero(eltype(local_source)))
            source!(
                balance_law,
                Vars{vars_state(balance_law, Prognostic(), FT)}(local_source),
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    state_1,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    aux_1,
                ),
            )

            @unroll for s in 1:num_state
                local_tendency[s] += α * local_source[s]
            end

            for l in 1:Nq
                # Compute derivatives wrt ξ1
                # ( G_11 (Q_1 ∘ H_1) - (H_1 ∘ Q_1^T) G_11) 1 +
                # ( G_12 (Q_1 ∘ H_2) - (H_2 ∘ Q_1^T) G_12) 1 +
                # ( G_13 (Q_1 ∘ H_3) - (H_3 ∘ Q_1^T) G_13) 1 +
                @unroll for s in 1:num_state
                    state_2[s] = shared_state[l, j, s]
                end
                @unroll for s in 1:num_aux
                    aux_2[s] = shared_aux[l, j, s]
                end
                fill!(local_H, -zero(FT))
                numerical_volume_flux_first_order!(
                    volume_numerical_flux_first_order,
                    balance_law,
                    local_H,
                    state_1,
                    aux_1,
                    state_2,
                    aux_2,
                )
                for s in 1:num_state
                    # G_11 (Q_1 ∘ H_1) 1 +
                    # G_12 (Q_1 ∘ H_2) 1 +
                    # G_13 (Q_1 ∘ H_3) 1
                    local_tendency[s] -=
                        local_MI[1] *
                        shared_D[i, l] *
                        (
                            shared_G[i, j, 1, 1] * local_H[1, s] +
                            shared_G[i, j, 1, 2] * local_H[2, s] +
                            (
                                dim == 3 ?
                                shared_G[i, j, 1, 3] * local_H[3, s] :
                                -zero(FT)
                            )
                        )
                    #  (H_1 ∘ Q_1^T) G_11 1 +
                    #  (H_2 ∘ Q_1^T) G_12 1 +
                    #  (H_3 ∘ Q_1^T) G_13 1
                    local_tendency[s] +=
                        local_MI[1] *
                        (
                            local_H[1, s] * shared_G[l, j, 1, 1] +
                            local_H[2, s] * shared_G[l, j, 1, 2] +
                            (
                                dim == 3 ?
                                local_H[3, s] * shared_G[l, j, 1, 3] :
                                -zero(FT)
                            )
                        ) *
                        shared_D[l, i]
                end

                # Compute derivatives wrt ξ2
                # ( G_21 (Q_2 ∘ H_1) - (H_1 ∘ Q_2^T) G_21) 1 +
                # ( G_22 (Q_2 ∘ H_2) - (H_2 ∘ Q_2^T) G_22) 1 +
                # ( G_23 (Q_2 ∘ H_3) - (H_3 ∘ Q_2^T) G_23) 1 +
                @unroll for s in 1:num_state
                    state_2[s] = shared_state[i, l, s]
                end
                @unroll for s in 1:num_aux
                    aux_2[s] = shared_aux[i, l, s]
                end
                fill!(local_H, -zero(FT))
                numerical_volume_flux_first_order!(
                    volume_numerical_flux_first_order,
                    balance_law,
                    local_H,
                    state_1,
                    aux_1,
                    state_2,
                    aux_2,
                )
                for s in 1:num_state
                    # G_21 (Q_2 ∘ H_1) 1 +
                    # G_22 (Q_2 ∘ H_2) 1 +
                    # G_23 (Q_2 ∘ H_3) 1
                    local_tendency[s] -=
                        local_MI[1] *
                        shared_D[j, l] *
                        (
                            shared_G[i, j, 2, 1] * local_H[1, s] +
                            shared_G[i, j, 2, 2] * local_H[2, s] +
                            (
                                dim == 3 ?
                                shared_G[i, j, 2, 3] * local_H[3, s] :
                                -zero(FT)
                            )
                        )
                    #  (H_1 ∘ Q_2^T) G_21 1 +
                    #  (H_2 ∘ Q_2^T) G_22 1 +
                    #  (H_3 ∘ Q_2^T) G_23 1
                    local_tendency[s] +=
                        local_MI[1] *
                        (
                            local_H[1, s] * shared_G[i, l, 2, 1] +
                            local_H[2, s] * shared_G[i, l, 2, 2] +
                            (
                                dim == 3 ?
                                local_H[3, s] * shared_G[i, l, 2, 3] :
                                -zero(FT)
                            )
                        ) *
                        shared_D[l, j]
                end

                # Compute derivatives wrt ξ3
                # ( G_31 (Q_3 ∘ H_1) - (H_1 ∘ Q_3^T) G_31) 1 +
                # ( G_32 (Q_3 ∘ H_2) - (H_2 ∘ Q_3^T) G_32) 1 +
                # ( G_33 (Q_3 ∘ H_3) - (H_3 ∘ Q_3^T) G_33) 1 +
                if dim == 3
                    @unroll for s in 1:num_state
                        state_2[s] = pencil_state[s, l]
                    end
                    @unroll for s in 1:num_aux
                        aux_2[s] = pencil_aux[s, l]
                    end
                    fill!(local_H, -zero(FT))
                    numerical_volume_flux_first_order!(
                        volume_numerical_flux_first_order,
                        balance_law,
                        local_H,
                        state_1,
                        aux_1,
                        state_2,
                        aux_2,
                    )
                    for s in 1:num_state
                        # G_21 (Q_2 ∘ H_1) 1 +
                        # G_22 (Q_2 ∘ H_2) 1 +
                        # G_23 (Q_2 ∘ H_3) 1
                        local_tendency[s] -=
                            local_MI[1] *
                            shared_D[k, l] *
                            (
                                pencil_G_3[1, k] * local_H[1, s] +
                                pencil_G_3[2, k] * local_H[2, s] +
                                pencil_G_3[3, k] * local_H[3, s]
                            )
                        #  (H_1 ∘ Q_2^T) G_21 1 +
                        #  (H_2 ∘ Q_2^T) G_22 1 +
                        #  (H_3 ∘ Q_2^T) G_23 1
                        local_tendency[s] +=
                            local_MI[1] *
                            (
                                local_H[1, s] * pencil_G_3[1, l] +
                                local_H[2, s] * pencil_G_3[2, l] +
                                local_H[3, s] * pencil_G_3[3, l]
                            ) *
                            shared_D[l, k]
                    end
                end

                ijk = i + Nq * ((j - 1) + Nq * (k - 1))
                @unroll for s in 1:num_state
                    tendency[ijk, s, e] = local_tendency[s]
                end
            end
        end
    end
end

# For PBL experiment
@kernel function kernel_drag_source!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    ::Val{nvertelem},
    tendency,
    state_prognostic,
    state_auxiliary,
    elems,
) where {dim, N, nvertelem}
    @uniform begin
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_auxiliary = number_states(balance_law, Auxiliary())

        # Number of Gauss-Lobatto quadrature points in each direction
        Nq = N + 1
        Nq1 = Nq
        Nq2 = dim == 2 ? 1 : Nq
        Nq3 = Nq

        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic_bot = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_source = MArray{Tuple{num_state_prognostic}, FT}(undef)
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]

        # Loop up the stack of elements
        for ev in 1:nvertelem
            e = ev + (eh - 1) * nvertelem

            if ev == 1
              ij = i + Nq1 * (j - 1)
              @unroll for s in 1:num_state_prognostic
                  local_state_prognostic_bot[s] = state_prognostic[ij, s, e]
              end
            end

            @unroll for k in 1:Nq3
                ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
                @unroll for s in 1:num_state_prognostic
                    local_state_prognostic[s] = state_prognostic[ijk, s, e]
                end

                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
                end

                fill!(local_source, -zero(eltype(local_source)))
                drag_source!(
                    balance_law,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_source,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic_bot,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary,
                    ),
                )
                
                @unroll for s in 1:num_state_prognostic
                    tendency[ijk, s, e] += local_source[s]
                end
            end
        end
    end
end

@kernel function esdg_volume_tendency_naive!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    volume_numerical_flux_first_order,
    tendency,
    state_prognostic,
    state_auxiliary,
    vgeo,
    D,
    α,
    β,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(state_prognostic)
        num_state = number_states(balance_law, Prognostic(), FT)
        num_aux = number_states(balance_law, Auxiliary(), FT)

        local_H = MArray{Tuple{3, num_state}, FT}(undef)

        state_1 = MArray{Tuple{num_state}, FT}(undef)
        aux_1 = MArray{Tuple{num_aux}, FT}(undef)
        state_2 = MArray{Tuple{num_state}, FT}(undef)
        aux_2 = MArray{Tuple{num_aux}, FT}(undef)

        local_source = MArray{Tuple{num_state}, FT}(undef)
        local_tendency = MArray{Tuple{num_state}, FT}(undef)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq
    end

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        ijk = i + Nq * ((j - 1) + Nq * (k - 1))
        
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

        fill!(local_source, -zero(eltype(local_source)))
        source!(
            balance_law,
            Vars{vars_state(balance_law, Prognostic(), FT)}(local_source),
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                state_1,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                aux_1,
            ),
        )

        @unroll for s in 1:num_state
            local_tendency[s] += α * local_source[s]
        end

        M_ijk = vgeo[ijk, _M, e]
        ξ1x1_ijk = M_ijk * vgeo[ijk, _ξ1x1, e]
        ξ1x2_ijk = M_ijk * vgeo[ijk, _ξ1x2, e]
        ξ1x3_ijk = M_ijk * vgeo[ijk, _ξ1x3, e]
        
        ξ2x1_ijk = M_ijk * vgeo[ijk, _ξ2x1, e]
        ξ2x2_ijk = M_ijk * vgeo[ijk, _ξ2x2, e]
        ξ2x3_ijk = M_ijk * vgeo[ijk, _ξ2x3, e]
        
        ξ3x1_ijk = M_ijk * vgeo[ijk, _ξ3x1, e]
        ξ3x2_ijk = M_ijk * vgeo[ijk, _ξ3x2, e]
        ξ3x3_ijk = M_ijk * vgeo[ijk, _ξ3x3, e]
        
        # Build ode scaling into mass matrix (so it doesn't show up later)
        local_MI = α / M_ijk

        @unroll for l in 1:Nq
            ljk = l + Nq * ((j - 1) + Nq * (k - 1))
            ilk = i + Nq * ((l - 1) + Nq * (k - 1))
            ijl = i + Nq * ((j - 1) + Nq * (l - 1))
        
            M_ljk = vgeo[ljk, _M, e]
            ξ1x1_ljk = M_ljk * vgeo[ljk, _ξ1x1, e]
            ξ1x2_ljk = M_ljk * vgeo[ljk, _ξ1x2, e]
            ξ1x3_ljk = M_ljk * vgeo[ljk, _ξ1x3, e]
            
            M_ilk = vgeo[ilk, _M, e]
            ξ2x1_ilk = M_ilk * vgeo[ilk, _ξ2x1, e]
            ξ2x2_ilk = M_ilk * vgeo[ilk, _ξ2x2, e]
            ξ2x3_ilk = M_ilk * vgeo[ilk, _ξ2x3, e]
            
            M_ijl = vgeo[ijl, _M, e]
            ξ3x1_ijl = M_ijl * vgeo[ijl, _ξ3x1, e]
            ξ3x2_ijl = M_ijl * vgeo[ijl, _ξ3x2, e]
            ξ3x3_ijl = M_ijl * vgeo[ijl, _ξ3x3, e]

            # Compute derivatives wrt ξ1
            # ( G_11 (Q_1 ∘ H_1) - (H_1 ∘ Q_1^T) G_11) 1 +
            # ( G_12 (Q_1 ∘ H_2) - (H_2 ∘ Q_1^T) G_12) 1 +
            # ( G_13 (Q_1 ∘ H_3) - (H_3 ∘ Q_1^T) G_13) 1 +
            @unroll for s in 1:num_state
                state_2[s] = state_prognostic[ljk, s, e]
            end
            @unroll for s in 1:num_aux
                aux_2[s] = state_auxiliary[ljk, s, e]
            end
            fill!(local_H, -zero(FT))
            numerical_volume_flux_first_order!(
                volume_numerical_flux_first_order,
                balance_law,
                local_H,
                state_1,
                aux_1,
                state_2,
                aux_2,
            )
            @unroll for s in 1:num_state
                # G_11 (Q_1 ∘ H_1) 1 +
                # G_12 (Q_1 ∘ H_2) 1 +
                # G_13 (Q_1 ∘ H_3) 1
                local_tendency[s] -=
                    local_MI *
                    D[i, l] *
                    (
                        ξ1x1_ijk * local_H[1, s] +
                        ξ1x2_ijk * local_H[2, s] +
                        (
                            dim == 3 ?
                            ξ1x3_ijk * local_H[3, s] :
                            -zero(FT)
                        )
                    )
                #  (H_1 ∘ Q_1^T) G_11 1 +
                #  (H_2 ∘ Q_1^T) G_12 1 +
                #  (H_3 ∘ Q_1^T) G_13 1
                local_tendency[s] +=
                    local_MI *
                    (
                        local_H[1, s] * ξ1x1_ljk +
                        local_H[2, s] * ξ1x2_ljk +
                        (
                            dim == 3 ?
                            local_H[3, s] * ξ1x3_ljk :
                            -zero(FT)
                        )
                    ) *
                    D[l, i]
            end

            # Compute derivatives wrt ξ2
            # ( G_21 (Q_2 ∘ H_1) - (H_1 ∘ Q_2^T) G_21) 1 +
            # ( G_22 (Q_2 ∘ H_2) - (H_2 ∘ Q_2^T) G_22) 1 +
            # ( G_23 (Q_2 ∘ H_3) - (H_3 ∘ Q_2^T) G_23) 1 +
            @unroll for s in 1:num_state
                state_2[s] = state_prognostic[ilk, s, e]
            end
            @unroll for s in 1:num_aux
                aux_2[s] = state_auxiliary[ilk, s, e] 
            end
            fill!(local_H, -zero(FT))
            numerical_volume_flux_first_order!(
                volume_numerical_flux_first_order,
                balance_law,
                local_H,
                state_1,
                aux_1,
                state_2,
                aux_2,
            )
            @unroll for s in 1:num_state
                # G_21 (Q_2 ∘ H_1) 1 +
                # G_22 (Q_2 ∘ H_2) 1 +
                # G_23 (Q_2 ∘ H_3) 1
                local_tendency[s] -=
                    local_MI *
                    D[j, l] *
                    (
                        ξ2x1_ijk * local_H[1, s] +
                        ξ2x2_ijk * local_H[2, s] +
                        (
                            dim == 3 ?
                            ξ2x3_ijk * local_H[3, s] :
                            -zero(FT)
                        )
                    )
                #  (H_1 ∘ Q_2^T) G_21 1 +
                #  (H_2 ∘ Q_2^T) G_22 1 +
                #  (H_3 ∘ Q_2^T) G_23 1
                local_tendency[s] +=
                    local_MI *
                    (
                        local_H[1, s] * ξ2x1_ilk +
                        local_H[2, s] * ξ2x2_ilk +
                        (
                            dim == 3 ?
                            local_H[3, s] * ξ2x3_ilk :
                            -zero(FT)
                        )
                    ) *
                    D[l, j]
            end

            # Compute derivatives wrt ξ3
            # ( G_31 (Q_3 ∘ H_1) - (H_1 ∘ Q_3^T) G_31) 1 +
            # ( G_32 (Q_3 ∘ H_2) - (H_2 ∘ Q_3^T) G_32) 1 +
            # ( G_33 (Q_3 ∘ H_3) - (H_3 ∘ Q_3^T) G_33) 1 +
            if dim == 3
                @unroll for s in 1:num_state
                    state_2[s] = state_prognostic[ijl, s, e]
                end
                @unroll for s in 1:num_aux
                    aux_2[s] = state_auxiliary[ijl, s, e]
                end
                fill!(local_H, -zero(FT))
                numerical_volume_flux_first_order!(
                    volume_numerical_flux_first_order,
                    balance_law,
                    local_H,
                    state_1,
                    aux_1,
                    state_2,
                    aux_2,
                )
                @unroll for s in 1:num_state
                    # G_21 (Q_2 ∘ H_1) 1 +
                    # G_22 (Q_2 ∘ H_2) 1 +
                    # G_23 (Q_2 ∘ H_3) 1
                    local_tendency[s] -=
                        local_MI *
                        D[k, l] *
                        (
                            ξ3x1_ijk * local_H[1, s] +
                            ξ3x2_ijk * local_H[2, s] +
                            ξ3x3_ijk * local_H[3, s]
                        )
                    #  (H_1 ∘ Q_2^T) G_21 1 +
                    #  (H_2 ∘ Q_2^T) G_22 1 +
                    #  (H_3 ∘ Q_2^T) G_23 1
                    local_tendency[s] +=
                        local_MI *
                        (
                            local_H[1, s] * ξ3x1_ijl +
                            local_H[2, s] * ξ3x2_ijl +
                            local_H[3, s] * ξ3x3_ijl
                        ) *
                        D[l, k]
                end
            end

            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            @unroll for s in 1:num_state
                tendency[ijk, s, e] = local_tendency[s]
            end
        end
    end
end

# For PBL experiment
@kernel function kernel_drag_source!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    ::Val{nvertelem},
    tendency,
    state_prognostic,
    state_auxiliary,
    elems,
) where {dim, N, nvertelem}
    @uniform begin
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_auxiliary = number_states(balance_law, Auxiliary())

        # Number of Gauss-Lobatto quadrature points in each direction
        Nq = N + 1
        Nq1 = Nq
        Nq2 = dim == 2 ? 1 : Nq
        Nq3 = Nq

        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic_bot = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_source = MArray{Tuple{num_state_prognostic}, FT}(undef)
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]

        # Loop up the stack of elements
        for ev in 1:nvertelem
            e = ev + (eh - 1) * nvertelem

            if ev == 1
              ij = i + Nq1 * (j - 1)
              @unroll for s in 1:num_state_prognostic
                  local_state_prognostic_bot[s] = state_prognostic[ij, s, e]
              end
            end

            @unroll for k in 1:Nq3
                ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
                @unroll for s in 1:num_state_prognostic
                    local_state_prognostic[s] = state_prognostic[ijk, s, e]
                end

                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
                end

                fill!(local_source, -zero(eltype(local_source)))
                drag_source!(
                    balance_law,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_source,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic_bot,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary,
                    ),
                )
                
                @unroll for s in 1:num_state_prognostic
                    tendency[ijk, s, e] += local_source[s]
                end
            end
        end
    end
end
