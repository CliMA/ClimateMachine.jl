@doc """
kernel_fvm_tendency!(
        balance_law::BalanceLaw,
        ::Val{dim},
        ::Val{N},
        ::Val{nvertelem},
        state_prognostic,
        state_auxiliary,
        vgeo,
        Imat,
        elems
    ) where {dim, N, nvertelem}

    Computational kernel: compute first order flux tendency.
See [`BalanceLaw`](@ref) for usage.
""" kernel_fvm_tendency!
@kernel function kernel_fvm_tendency!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    numerical_flux_first_order,
    numerical_flux_second_order,
    tendency,
    state_prognostic,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    sgeo,
    t,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
    α,
) where {dim, N, nvertelem}
    @uniform begin
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        nout = number_states(balance_law, UpwardIntegrals())
        vsp = Vars{vars_state(balance_law, Prognostic(), FT)}
        vsa = Vars{vars_state(balance_law, Auxiliary(), FT)}
        Nq = N + 1
        Nqj = dim == 2 ? 1 : Nq

        # Is this correct?
        face_direction = VerticalDirection()

        flux_bot = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic⁺ = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic⁻ = MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_primitive = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_primitive⁺ = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_primitive⁻ = MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_primitive_bot = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_primitive_top = MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_prognostic_bot = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic_top = MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_auxiliary_top⁻ = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_auxiliary_bot = MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_kernel = MArray{Tuple{nout, Nq}, FT}(undef)
    end

    local_integral = @private FT (nout, Nq)
    s_I = @localmem FT (Nq, Nq)

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        # @unroll for n in 1:Nq
        #     s_I[i, n] = Imat[i, n]
        # end
        # @synchronize

        eh = elems[_eh]

        # Loop up the stack of elements
        # ----> z (r) direction

        # Element/DOF naming convention:
        #    | --------------- | --------------- | --------------- | -----> z
        #    | bot⁻   e⁻  top⁻ | bot    e    top | bot⁺  e⁺   top⁺ |
        
        for ev in 1:nvertelem
            e = ev + (eh - 1) * nvertelem
            e⁺ = e + 1
            e⁻ = e - 1

            # TODO: write correct expression for cell_weights:
            cell_weights = SVector(1,1,1)

            # Step 1: reconstruct states on bottom/top faces
            ijk = i + Nq * ((j - 1))
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[ijk, s, e]
            end
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic⁻[s] = state_prognostic[ijk, s, e⁻]
            end
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic⁺[s] = state_prognostic[ijk, s, e⁺]
            end

            prognostic_to_primitive!(
                balance_law,
                Vars{vsp}(local_state_primitive),
                Vars{vsp}(local_state_prognostic)
            )
            prognostic_to_primitive!(
                balance_law,
                Vars{vsp}(local_state_primitive⁺),
                Vars{vsp}(local_state_prognostic⁺)
            )
            prognostic_to_primitive!(
                balance_law,
                Vars{vsp}(local_state_primitive⁻),
                Vars{vsp}(local_state_prognostic⁻)
            )

            cell_states_primitive = (local_state_primitive⁻,  local_state_primitive, local_state_primitive⁺)

            fv_reconstruction!(local_state_primitive_bot, local_state_primitive_top, cell_states_primitive, cell_weights)
            
            primitive_to_prognostic!(
                balance_law,
                Vars{vsp}(local_state_prognostic_bot),
                Vars{vsp}(local_state_primitive_bot),
            )
            primitive_to_prognostic!(
                balance_law,
                Vars{vsp}(local_state_prognostic_top),
                Vars{vsp}(local_state_primitive_top),
            )            

            # Step 2: compute flux on the bottom face
            
            # compute flux_bot from
            #   local_state_prognostic_top⁻
            #   local_state_prognostic_bot
            numerical_flux_first_order!(
                numerical_flux_first_order,
                balance_law,
                Vars{vsp}(flux_bot),
                SVector(normal_vector),
                Vars{vsp}(local_state_prognostic_top⁻),
                Vars{vsa}(local_state_auxiliary_top⁻),
                Vars{vsp}(local_state_prognostic_bot),
                Vars{vsa}(local_state_auxiliary_bot),
                t,
                face_direction,
            )

            # Update RHS (in outer face loop): M⁻¹ Mfᵀ(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆))
            # TODO: This isn't correct:
            # FIXME: Should we pretch these?
            @unroll for s in 1:num_state_prognostic
                tendency[vid⁻, s, e⁻] -= α * vMI * sM * flux_bot[s]
            end
            @unroll for s in 1:num_state_prognostic
                tendency[vid⁻, s, e] -= α * vMI * sM * flux_bot[s]
            end
            # Need to wait after even faces to avoid race conditions
            # @synchronize(f % 2 == 0)

            # Handle ragged end
            if ev == nvertelem
            end


            local_state_prognostic_top⁻ = local_state_prognostic_top
            
        end
        # Step 3: compute flux on the top face (bc condition)
    end
end
