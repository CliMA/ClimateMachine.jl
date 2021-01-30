import .FVReconstructions: FVConstant, FVLinear
import ..BalanceLaws:
    prognostic_to_primitive!, primitive_to_prognostic!, Primitive
import .FVReconstructions: width
import StaticArrays: SUnitRange

@doc """
    function vert_fvm_interface_tendency!(
        balance_law::BalanceLaw,
        ::Val{info},
        ::Val{nvertelem},
        ::Val{periodicstack},
        ::VerticalDirection,
        numerical_flux_first_order,
        tendency,
        state_prognostic,
        state_auxiliary,
        vgeo,
        sgeo,
        t,
        vmap⁻,
        vmap⁺,
        elemtobndy,
        elems,
        α,
    )

Compute kernel for evaluating the interface tendencies using vertical FVM
reconstructions with a DG method of the form:

∫ₑ ψ⋅ ∂q/∂t dx - ∫ₑ ∇ψ⋅(Fⁱⁿᵛ + Fᵛⁱˢᶜ) dx + ∮ₑ n̂ ψ⋅(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆) dS,

or equivalently in matrix form:

dQ/dt = M⁻¹(MS + DᵀM(Fⁱⁿᵛ + Fᵛⁱˢᶜ) + ∑ᶠ LᵀMf(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆)).

This kernel computes the surface terms: M⁻¹ ∑ᶠ LᵀMf(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆)), where M
is the mass matrix, Mf is the face mass matrix, L is an interpolator from
volume to face, and Fⁱⁿᵛ⋆, Fᵛⁱˢᶜ⋆ are the numerical fluxes for the inviscid
and viscous fluxes, respectively.

A finite volume reconstruction is used to construction `Fⁱⁿᵛ⋆`
""" vert_fvm_interface_tendency!
@kernel function vert_fvm_interface_tendency!(
    balance_law::BalanceLaw,
    ::Val{info},
    ::Val{nvertelem},
    ::Val{periodicstack},
    ::VerticalDirection,
    reconstruction!,
    numerical_flux_first_order,
    numerical_flux_second_order,
    tendency,
    state_prognostic,
    state_gradient_flux,
    state_auxiliary,
    vgeo,
    sgeo,
    t,
    elemtobndy,
    elems,
    α,
    β,
    increment,
    add_source,
) where {info, nvertelem, periodicstack}
    @uniform begin
        dim = info.dim
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_primitive = number_states(balance_law, Primitive())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        num_state_hyperdiffusive = number_states(balance_law, Hyperdiffusive())
        @assert num_state_hyperdiffusive == 0

        nface = info.nface
        Np = info.Np
        Nqk = info.Nqk # Can only be 1 for the FVM method!
        @assert Nqk == 1

        # We only have the vertical faces
        faces = (nface - 1):nface

        stencil_width = width(reconstruction!)

        # If this fails the `@nif` below needs to change
        @assert stencil_width < 4

        # In the case of stencil_width = 0 we still need two values to evaluate
        # the fluxes, so the minimum stencil diameter is 2
        stencil_diameter = max(2, 2stencil_width + 1)

        # Value in the stencil that corresponds to the top face with respect to
        # face being updated
        stencil_center = max(stencil_width, 1) + 1

        # 1 → element i, face i - 1/2 (bottom face)
        # 2 → element i, face i + 1/2 (top face)
        local_state_face_prognostic = ntuple(Val(2)) do _
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        end

        local_cell_weights = MArray{Tuple{stencil_diameter}, FT}(undef)

        # Two mass matrix inverse corresponding to +/- cells
        vMI = MArray{Tuple{2}, FT}(undef)

        # Storing the value below element when walking up the stack
        # cell i-1, face i - 1/2
        local_state_face_prognostic_neighbor =
            MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_face_primitive = ntuple(Val(2)) do _
            MArray{Tuple{num_state_primitive}, FT}(undef)
        end

        # Storage for all the values in the stencil
        local_state_prognostic = ntuple(Val(stencil_diameter)) do _
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        end


        # Need to wrap in SVector so we can use views later
        local_state_primitive = SVector(ntuple(Val(stencil_diameter)) do _
            MArray{Tuple{num_state_primitive}, FT}(undef)
        end...)

        local_state_auxiliary = ntuple(Val(stencil_diameter)) do _
            MArray{Tuple{num_state_auxiliary}, FT}(undef)
        end

        # FIXME: These two arrays could be smaller
        # (only 2 elements not stencil_diameter)
        local_state_gradient_flux = ntuple(Val(stencil_diameter)) do _
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        end

        local_state_hyperdiffusive = ntuple(Val(stencil_diameter)) do _
            MArray{Tuple{num_state_hyperdiffusive}, FT}(undef)
        end

        # Storage for the numerical flux
        local_flux = MArray{Tuple{num_state_prognostic}, FT}(undef)

        # Storage for the extra boundary points for some BCs
        local_state_prognostic_bottom1 =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_gradient_flux_bottom1 =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_auxiliary_bottom1 =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)

        # Storage for the tendency
        local_tendency = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_source = MArray{Tuple{num_state_prognostic}, FT}(undef)

        # XXX: will revisit this later for FVM
        fill!(local_state_prognostic_bottom1, NaN)
        fill!(local_state_gradient_flux_bottom1, NaN)
        fill!(local_state_auxiliary_bottom1, NaN)

        # The remainder model needs to know which direction of face the model is
        # being evaluated for. In this case we only have `VerticalDirection()`
        # faces
        face_direction = (VerticalDirection(),)
    end

    # Optimization ideas:
    #  - More than 1 thread per stack
    #  - shift pointers not data
    #  - Don't keep wide stencil for all variables / data
    #  - Don't load periodic data when domain is not periodic

    # Get the horizontal group IDs
    grp_H = @index(Group, Linear)

    # Determine the index for the element at the bottom of the stack
    eHI = (grp_H - 1) * nvertelem + 1

    # Compute bottom stack element index minus one (so we can add vert element
    # number directly)
    eH = elems[eHI] - 1

    # Which degree of freedom do we handle in the element
    n = @index(Local, Linear)


    # Loads the data for a given element
    function load_data!(
        local_state_prognostic,
        local_state_auxiliary,
        local_state_gradient_flux,
        e,
    )
        @unroll for s in 1:num_state_prognostic
            @inbounds local_state_prognostic[s] = state_prognostic[n, s, e]
        end

        @unroll for s in 1:num_state_auxiliary
            @inbounds local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end

        @unroll for s in 1:num_state_gradient_flux
            @inbounds local_state_gradient_flux[s] =
                state_gradient_flux[n, s, e]
        end
    end

    # To update the first element we either need to apply a BCS on the bottom
    # face in the nonperiodic case, or we need to reconstruct the periodic value
    # for the bottom face
    @inbounds begin
        # If periodic, we are doing the reconstruction in element periodically
        # below the first element, otherwise we are reconstructing the first
        # element
        eV = periodicstack ? nvertelem : 1

        # Figure out the data we need
        els = ntuple(Val(stencil_diameter)) do k
            eH + mod1(eV - 1 + k - (stencil_center - 1), nvertelem)
        end

        # Load all the stencil data
        @unroll for k in 1:stencil_diameter
            load_data!(
                local_state_prognostic[k],
                local_state_auxiliary[k],
                local_state_gradient_flux[k],
                els[k],
            )
            # If local cell weights are NOT _M we need to load _vMI out of sgeo
            local_cell_weights[k] = 2 * vgeo[n, _JcV, els[k]]
        end

        # Transform all the data into primitive variables
        @unroll for k in 1:stencil_diameter
            prognostic_to_primitive!(
                balance_law,
                local_state_primitive[k],
                local_state_prognostic[k],
                local_state_auxiliary[k],
            )
        end
        vMI[2] = sgeo[_vMI, n, faces[2], els[stencil_center]]

        # If we are periodic we reconstruct the top and bottom values for eV
        # then start with eV update in loop below
        if periodicstack
            # Reconstruct the top and bottom values
            rng1, rng2 = stencil_center .+ (-stencil_width, stencil_width)
            rng = SUnitRange(rng1, rng2)
            reconstruction!(
                local_state_face_primitive[1],
                local_state_face_primitive[2],
                local_state_primitive[rng],
                local_cell_weights[rng],
            )

            # Transform the values back to prognostic state
            @unroll for f in 1:2
                primitive_to_prognostic!(
                    balance_law,
                    local_state_face_prognostic[f],
                    local_state_face_primitive[f],
                    # Use the cell auxiliary data
                    local_state_auxiliary[stencil_center],
                )
            end

            # Initialize local tendency
            @unroll for s in 1:num_state_prognostic
                local_tendency[s] = -zero(FT)
            end
        else

            bctag = elemtobndy[faces[1], eH + eV]

            sM = sgeo[_sM, n, faces[1], eH + eV]
            normal = SVector(
                sgeo[_n1, n, faces[1], eH + eV],
                sgeo[_n2, n, faces[1], eH + eV],
                sgeo[_n3, n, faces[1], eH + eV],
            )

            # Reconstruction using only eVs cell value
            rng = SUnitRange(stencil_center, stencil_center)
            reconstruction!(
                local_state_face_primitive[1],
                local_state_face_primitive[2],
                local_state_primitive[rng],
                local_cell_weights[rng],
            )

            # Transform the values back to prognostic state
            @unroll for k in 1:2
                primitive_to_prognostic!(
                    balance_law,
                    local_state_face_prognostic[k],
                    local_state_face_primitive[k],
                    # Use the cell auxiliary data
                    local_state_auxiliary[stencil_center],
                )
            end

            # Fill ghost cell data
            fill!(local_flux, -zero(FT))
            local_state_face_prognostic_neighbor .=
                local_state_face_prognostic[1]
            local_state_auxiliary[stencil_center - 1] .=
                local_state_auxiliary[stencil_center]

            numerical_boundary_flux_first_order!(
                numerical_flux_first_order,
                bctag,
                balance_law,
                local_flux,
                normal,
                local_state_face_prognostic[1],
                local_state_auxiliary[stencil_center],
                local_state_face_prognostic_neighbor,
                local_state_auxiliary[stencil_center - 1],
                t,
                face_direction,
                local_state_prognostic_bottom1,
                local_state_auxiliary_bottom1,
            )

            # Fill / reset ghost cell data
            local_state_prognostic[stencil_center - 1] .=
                local_state_prognostic[stencil_center]
            local_state_gradient_flux[stencil_center - 1] .=
                local_state_gradient_flux[stencil_center]
            local_state_hyperdiffusive[stencil_center - 1] .=
                local_state_hyperdiffusive[stencil_center]
            local_state_auxiliary[stencil_center - 1] .=
                local_state_auxiliary[stencil_center]

            numerical_boundary_flux_second_order!(
                numerical_flux_second_order,
                bctag,
                balance_law,
                local_flux,
                normal,
                local_state_prognostic[stencil_center],
                local_state_gradient_flux[stencil_center],
                local_state_hyperdiffusive[stencil_center],
                local_state_auxiliary[stencil_center],
                local_state_prognostic[stencil_center - 1],
                local_state_gradient_flux[stencil_center - 1],
                local_state_hyperdiffusive[stencil_center - 1],
                local_state_auxiliary[stencil_center - 1],
                t,
                local_state_prognostic_bottom1,
                local_state_gradient_flux_bottom1,
                local_state_auxiliary_bottom1,
            )

            # Compute boundary flux and add it in bottom element of the mesh
            @unroll for s in 1:num_state_prognostic
                local_tendency[s] = -α * sM * vMI[2] * local_flux[s]
            end
        end

        # The rest of the elements in the stack
        # Compute flux and update for face between elements eV and eV - 1
        #    top face of eV - 1
        #    bottom face of eV
        # For the reconstruction arrays `stencil_center - 1` corresponds to `eV
        # - 1` and `stencil_center` corresponds to `eV`
        #
        # Loop for periodic case has to go beyond the top element so we compute
        # the flux through the top face of vertical element `nvertelem` and
        # bottom face of vertical element 1
        for eV_up in (periodicstack ? (1:nvertelem) : (2:nvertelem))
            # mod1 handles periodicity
            eV_dn = mod1(eV_up - 1, nvertelem)

            # Shift data in storage in order to load new upper element for
            # reconstruction
            # FIXME: shift pointers not data?
            @unroll for k in 1:(stencil_diameter - 1)
                local_state_prognostic[k] .= local_state_prognostic[k + 1]
                local_state_primitive[k] .= local_state_primitive[k + 1]
                local_state_auxiliary[k] .= local_state_auxiliary[k + 1]
                local_state_gradient_flux[k] .= local_state_gradient_flux[k + 1]
                local_cell_weights[k] = local_cell_weights[k + 1]
            end

            # Update volume mass inverse as we move up the stack of elements
            vMI[1] = vMI[2]

            # Load surface metrics for the face we will update (bottom face of
            # `eV_up`)
            sM = sgeo[_sM, n, faces[1], eH + eV_up]
            normal = SVector(
                sgeo[_n1, n, faces[1], eH + eV_up],
                sgeo[_n2, n, faces[1], eH + eV_up],
                sgeo[_n3, n, faces[1], eH + eV_up],
            )

            # Reconstruction for eV_dn was computed in last time through the
            # loop, so we need to store the upper reconstructed values to
            # compute flux for this face
            local_state_face_prognostic_neighbor .=
                local_state_face_prognostic[2]

            # Next data we need to load (assume periodic, mod1, for now  will
            # mask out below as needed for boundary conditions)
            eV_load = mod1(eV_up + stencil_width, nvertelem)

            # Get element number
            e_load = eH + eV_load

            # Load the next cell into the end of the element arrays
            load_data!(
                local_state_prognostic[stencil_diameter],
                local_state_auxiliary[stencil_diameter],
                local_state_gradient_flux[stencil_diameter],
                e_load,
            )

            # Get local volume mass matrix inverse
            local_cell_weights[stencil_diameter] = 2vgeo[n, _JcV, e_load]
            vMI[2] = sgeo[_vMI, n, faces[2], eH + eV_up]

            # Tranform the prognostic data to primitive data
            prognostic_to_primitive!(
                balance_law,
                local_state_primitive[stencil_diameter],
                local_state_prognostic[stencil_diameter],
                local_state_auxiliary[stencil_diameter],
            )

            # Do the reconstruction! for this cell and compute the values at the
            # bottom (1) and top (2) faces of element `eV_up`
            if periodicstack ||
               stencil_width < eV_up < nvertelem - stencil_width + 1
                # If we are in the interior or periodic just use the
                # reconstruction
                rng1, rng2 = stencil_center .+ (-stencil_width, stencil_width)
                rng = SUnitRange(rng1, rng2)
                reconstruction!(
                    local_state_face_primitive[1],
                    local_state_face_primitive[2],
                    local_state_primitive[rng],
                    local_cell_weights[rng],
                )
            elseif eV_up <= stencil_width
                # Bottom of the element stack requires reconstruct using a
                # subset of the elements
                # Values around stencil center that we need for this
                # reconstruction
                Base.Cartesian.@nif 4 w -> (eV_up == w) w -> begin
                    rng1, rng2 = stencil_center .+ (1 - w, w - 1)
                    rng = SUnitRange(rng1, rng2)
                    reconstruction!(
                        local_state_face_primitive[1],
                        local_state_face_primitive[2],
                        local_state_primitive[rng],
                        local_cell_weights[rng],
                    )
                end w -> throw(BoundsError(local_state_primitive, w))
            elseif eV_up >= nvertelem - stencil_width + 1
                # Top of the element stack requires reconstruct using a
                # subset of the elements
                Base.Cartesian.@nif 4 w -> (w == (nvertelem - eV_up + 1)) w ->
                    begin
                        rng1, rng2 = stencil_center .+ (1 - w, w - 1)
                        rng = SUnitRange(rng1, rng2)
                        reconstruction!(
                            local_state_face_primitive[1],
                            local_state_face_primitive[2],
                            local_state_primitive[rng],
                            local_cell_weights[rng],
                        )
                    end w -> throw(BoundsError(local_state_primitive, w))
            end

            # Transform reconstructed primitive values to prognostic
            @unroll for k in 1:2
                primitive_to_prognostic!(
                    balance_law,
                    local_state_face_prognostic[k],
                    local_state_face_primitive[k],
                    # Use the cell auxiliary data
                    local_state_auxiliary[stencil_center],
                )
            end

            # Compute the flux for the bottom face of the element we are
            # considering
            fill!(local_flux, -zero(FT))
            numerical_flux_first_order!(
                numerical_flux_first_order,
                balance_law,
                local_flux,
                normal,
                local_state_face_prognostic[1],
                local_state_auxiliary[stencil_center],
                local_state_face_prognostic_neighbor,
                local_state_auxiliary[stencil_center - 1],
                t,
                face_direction,
            )

            numerical_flux_second_order!(
                numerical_flux_second_order,
                balance_law,
                local_flux,
                normal,
                local_state_prognostic[stencil_center],
                local_state_gradient_flux[stencil_center],
                local_state_hyperdiffusive[stencil_center],
                local_state_auxiliary[stencil_center],
                local_state_prognostic[stencil_center - 1],
                local_state_gradient_flux[stencil_center - 1],
                local_state_hyperdiffusive[stencil_center - 1],
                local_state_auxiliary[stencil_center - 1],
                t,
            )

            if add_source
                fill!(local_source, -zero(eltype(local_source)))
                source_arr!(
                    balance_law,
                    local_source,
                    local_state_prognostic[stencil_center - 1],
                    local_state_gradient_flux[stencil_center - 1],
                    local_state_auxiliary[stencil_center - 1],
                    t,
                    (VerticalDirection(),),
                )
                @unroll for s in 1:num_state_prognostic
                    local_tendency[s] += local_source[s]
                end
            end


            # Update the bottom element:
            # numerical flux is computed with respect to the top element, so
            # `+=` is used to reverse the flux
            @unroll for s in 1:num_state_prognostic
                local_tendency[s] += α * sM * vMI[1] * local_flux[s]

                if increment
                    tendency[n, s, eH + eV_dn] += local_tendency[s]
                else
                    if β != 0
                        T = local_tendency[s] + β * tendency[n, s, eH + eV_dn]
                    else
                        T = local_tendency[s]
                    end
                    tendency[n, s, eH + eV_dn] = T
                end

                # Store contribution to the top element tendency
                local_tendency[s] = -α * sM * vMI[2] * local_flux[s]
            end

            # Update top element of the stack
            if eV_up == nvertelem
                if add_source
                    fill!(local_source, -zero(eltype(local_source)))
                    source_arr!(
                        balance_law,
                        local_source,
                        local_state_prognostic[stencil_center],
                        local_state_gradient_flux[stencil_center],
                        local_state_auxiliary[stencil_center],
                        t,
                        (VerticalDirection(),),
                    )
                    @unroll for s in 1:num_state_prognostic
                        local_tendency[s] += local_source[s]
                    end
                end
                # If periodic just add in the tendency that we just computed
                if periodicstack
                    @unroll for s in 1:num_state_prognostic
                        if increment
                            tendency[n, s, eH + eV_up] += local_tendency[s]
                        else
                            # This tendency has already been updated in the first element (eV_up=1)
                            tendency[n, s, eH + eV_up] += local_tendency[s]
                        end
                    end
                else
                    # Load surface metrics for the face we will update
                    # (top face of `eV_up`)
                    bctag = elemtobndy[faces[2], eH + eV_up]
                    sM = sgeo[_sM, n, faces[2], eH + eV_up]
                    normal = SVector(
                        sgeo[_n1, n, faces[2], eH + eV_up],
                        sgeo[_n2, n, faces[2], eH + eV_up],
                        sgeo[_n3, n, faces[2], eH + eV_up],
                    )

                    # Since we are at the last element to update, we can safely use
                    # `stencil_center - 1` to store the ghost data

                    fill!(local_flux, -zero(FT))

                    # Fill ghost cell data
                    # Use the top reconstruction (since handling top face)
                    local_state_face_prognostic_neighbor .=
                        local_state_face_prognostic[2]
                    local_state_auxiliary[stencil_center - 1] .=
                        local_state_auxiliary[stencil_center]
                    numerical_boundary_flux_first_order!(
                        numerical_flux_first_order,
                        bctag,
                        balance_law,
                        local_flux,
                        normal,
                        # Use the top reconstruction (since handling top face)
                        local_state_face_prognostic[2],
                        local_state_auxiliary[stencil_center],
                        local_state_face_prognostic_neighbor,
                        local_state_auxiliary[stencil_center - 1],
                        t,
                        face_direction,
                        local_state_prognostic_bottom1,
                        local_state_auxiliary_bottom1,
                    )

                    # Fill / reset ghost cell data
                    local_state_prognostic[stencil_center - 1] .=
                        local_state_prognostic[stencil_center]
                    local_state_gradient_flux[stencil_center - 1] .=
                        local_state_gradient_flux[stencil_center]
                    local_state_hyperdiffusive[stencil_center - 1] .=
                        local_state_hyperdiffusive[stencil_center]
                    local_state_auxiliary[stencil_center - 1] .=
                        local_state_auxiliary[stencil_center]
                    numerical_boundary_flux_second_order!(
                        numerical_flux_second_order,
                        bctag,
                        balance_law,
                        local_flux,
                        normal,
                        local_state_prognostic[stencil_center],
                        local_state_gradient_flux[stencil_center],
                        local_state_hyperdiffusive[stencil_center],
                        local_state_auxiliary[stencil_center],
                        local_state_prognostic[stencil_center - 1],
                        local_state_gradient_flux[stencil_center - 1],
                        local_state_hyperdiffusive[stencil_center - 1],
                        local_state_auxiliary[stencil_center - 1],
                        t,
                        local_state_prognostic_bottom1,
                        local_state_gradient_flux_bottom1,
                        local_state_auxiliary_bottom1,
                    )

                    @unroll for s in 1:num_state_prognostic
                        local_tendency[s] -= α * sM * vMI[2] * local_flux[s]
                        if increment
                            tendency[n, s, eH + eV_up] += local_tendency[s]
                        else
                            if β != 0
                                T =
                                    local_tendency[s] +
                                    β * tendency[n, s, eH + eV_up]
                            else
                                T = local_tendency[s]
                            end
                            tendency[n, s, eH + eV_up] = T
                        end
                    end
                end
            end
        end
    end
end

@kernel function vert_fvm_interface_gradients!(
    balance_law::BalanceLaw,
    ::Val{info},
    ::Val{nvertelem},
    ::Val{periodicstack},
    ::VerticalDirection,
    state_prognostic,
    state_gradient_flux,
    state_auxiliary,
    vgeo,
    sgeo,
    t,
    elemtobndy,
    elems,
    increment,
) where {info, nvertelem, periodicstack}
    @uniform begin

        dim = info.dim
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        ngradstate = number_states(balance_law, Gradient())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        nface = info.nface
        Np = info.Np
        faces = (nface - 1, nface)

        # Storage for the prognostic state for e-1, e, e+1
        local_state_prognostic =
            ntuple(_ -> MArray{Tuple{num_state_prognostic}, FT}(undef), Val(3))

        # Storage for the auxiliary state for e-1, e, e+1
        local_state_auxiliary =
            ntuple(_ -> MArray{Tuple{num_state_auxiliary}, FT}(undef), Val(3))

        # Storage for the transform state for e-1, e, e+1
        # e.g., the state we take the gradient of
        l_grad_arg = ntuple(_ -> MArray{Tuple{ngradstate}, FT}(undef), Val(3))

        # Storage for the contribution to the gradient from these faces
        l_nG = MArray{Tuple{3, ngradstate}, FT}(undef)

        l_nG_bc = MArray{Tuple{3, ngradstate}, FT}(undef)

        # Storage for the state gradient flux locally
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)

        local_state_prognostic_bottom1 =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary_bottom1 =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)

        # XXX: will revisit this later for FVM
        fill!(local_state_prognostic_bottom1, NaN)
        fill!(local_state_auxiliary_bottom1, NaN)
    end

    # Element index
    eI = @index(Group, Linear)
    # Index of a quadrature point on a face
    n = @index(Local, Linear)

    @inbounds begin
        e = elems[eI]
        eV = mod1(e, nvertelem)

        # Figure out the element above and below e
        e_dn, bc_dn = if eV > 1
            e - 1, 0
        elseif periodicstack
            e + nvertelem - 1, 0
        else
            e, elemtobndy[faces[1], e]
        end

        e_up, bc_up = if eV < nvertelem
            e + 1, 0
        elseif periodicstack
            e - nvertelem + 1, 0
        else
            e, elemtobndy[faces[2], e]
        end

        bctag = (bc_dn, bc_up)

        els = (e_dn, e, e_up)

        # Load the normal vectors and surface mass matrix on the faces
        normal_vector = ntuple(
            k -> SVector(
                sgeo[_n1, n, faces[k], e],
                sgeo[_n2, n, faces[k], e],
                sgeo[_n3, n, faces[k], e],
            ),
            Val(2),
        )
        sM = ntuple(k -> sgeo[_sM, n, faces[k], e], Val(2))

        # Volume mass same on both faces
        vMI = sgeo[_vMI, n, faces[1], e]

        # Get the mass matrix for each of the elements
        M = ntuple(k -> vgeo[n, _M, els[k]], Val(3))

        # Load prognostic data
        @unroll for k in 1:3
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[k][s] = state_prognostic[n, s, els[k]]
            end
        end

        # Load auxiliary data
        @unroll for k in 1:3
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[k][s] = state_auxiliary[n, s, els[k]]
            end
        end

        # Transform to the gradient argument (i.e., the values we take the
        # gradient of)
        @unroll for k in 1:3
            fill!(l_grad_arg[k], -zero(eltype(l_grad_arg[k])))
            compute_gradient_argument_arr!(
                balance_law,
                l_grad_arg[k],
                local_state_prognostic[k],
                local_state_auxiliary[k],
                t,
            )
        end

        # Compute the surface integral contribution from these two faces:
        #   M⁻¹ Lᵀ Mf n̂ G*
        # Since this is FVM we do not subtract the interior state
        fill!(l_nG, -zero(eltype(l_nG)))
        @unroll for f in 1:2
            if bctag[f] == 0
                @unroll for s in 1:ngradstate
                    # Interpolate to the face (using the mass matrix for the
                    # interpolation weights) -- "the gradient numerical flux"
                    G =
                        (
                            M[f] * l_grad_arg[f + 1][s] +
                            M[f + 1] * l_grad_arg[f][s]
                        ) / (M[f] + M[f + 1])

                    # Compute the surface integral for this component and face
                    # multiplied by the normal to get the rotation to the
                    # physical space
                    @unroll for i in 1:3
                        l_nG[i, s] += vMI * sM[f] * normal_vector[f][i] * G
                    end
                end
            else
                # Computes G* incorporating boundary conditions
                numerical_boundary_flux_gradient!(
                    CentralNumericalFluxGradient(),
                    bctag[f],
                    balance_law,
                    l_nG_bc,
                    normal_vector[f],
                    l_grad_arg[2],
                    local_state_prognostic[2],
                    local_state_auxiliary[2],
                    l_grad_arg[2f - 1],
                    local_state_prognostic[2f - 1],
                    local_state_auxiliary[2f - 1],
                    t,
                    local_state_prognostic_bottom1,
                    local_state_auxiliary_bottom1,
                )

                @unroll for s in 1:ngradstate
                    @unroll for i in 1:3
                        l_nG[i, s] += vMI * sM[f] * l_nG_bc[i, s]
                    end
                end
            end
        end

        # Applies linear transformation of gradients to the diffusive variables
        # for storage
        compute_gradient_flux_arr!(
            balance_law,
            local_state_gradient_flux,
            l_nG,
            local_state_prognostic[2],
            local_state_auxiliary[2],
            t,
        )

        # This is the surface integral evaluated discretely
        # M^(-1) Mf G*
        @unroll for s in 1:num_state_gradient_flux
            if increment
                state_gradient_flux[n, s, e] += local_state_gradient_flux[s]
            else
                state_gradient_flux[n, s, e] = local_state_gradient_flux[s]
            end
        end
    end
end

@kernel function vert_fvm_auxiliary_field_gradient!(
    balance_law::BalanceLaw,
    ::Val{info},
    ∇state,
    state,
    vgeo,
    sgeo,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    ::Val{I},
    ::Val{O},
    increment,
) where {info, I, O}
    @uniform begin
        FT = eltype(state)
        ngradstate = length(I)

        dim = info.dim
        nface = info.nface
        Np = info.Np

        # We only have the vertical faces
        faces = (nface - 1):nface

        local_state_bottom = fill!(MArray{Tuple{ngradstate}, FT}(undef), NaN)
        local_state = fill!(MArray{Tuple{ngradstate}, FT}(undef), NaN)
        local_state_top = fill!(MArray{Tuple{ngradstate}, FT}(undef), NaN)
    end

    e = @index(Group, Linear)
    n = @index(Local, Linear)

    @inbounds begin
        face_bottom = faces[1]
        face_top = faces[2]

        bctag_bottom = elemtobndy[face_bottom, e]
        bctag_top = elemtobndy[face_top, e]

        # TODO: exploit structured grid
        id = vmap⁻[n, face_bottom, e]
        id_bottom = vmap⁺[n, face_bottom, e]
        id_top = vmap⁺[n, face_top, e]

        e_bottom = ((id_bottom - 1) ÷ Np) + 1
        e_top = ((id_top - 1) ÷ Np) + 1

        vid = ((id - 1) % Np) + 1
        vid_bottom = ((id_bottom - 1) % Np) + 1
        vid_top = ((id_top - 1) % Np) + 1

        if bctag_bottom != 0
            e_bottom = e
            vid_bottom = vid
        end
        if bctag_top != 0
            e_top = e
            vid_top = vid
        end

        dzh_bottom = vgeo[vid_bottom, _JcV, e_bottom]
        dzh = vgeo[vid, _JcV, e]
        dzh_top = vgeo[vid_top, _JcV, e_top]

        if dim == 2
            ξvx1 = vgeo[vid, _ξ2x1, e]
            ξvx2 = vgeo[vid, _ξ2x2, e]
            ξvx3 = vgeo[vid, _ξ2x3, e]
        elseif dim == 3
            ξvx1 = vgeo[vid, _ξ3x1, e]
            ξvx2 = vgeo[vid, _ξ3x2, e]
            ξvx3 = vgeo[vid, _ξ3x3, e]
        end

        @unroll for s in 1:ngradstate
            local_state_bottom[s] = state[vid_bottom, I[s], e_bottom]
            local_state_top[s] = state[vid_top, I[s], e_top]
        end

        # only need the middle state near the boundaries
        if bctag_bottom != 0 || bctag_top != 0
            @unroll for s in 1:ngradstate
                local_state[s] = state[vid, I[s], e]
            end
        end

        # extrapolation at the boundaries equivalent to one-sided differencing
        if bctag_bottom != 0
            @unroll for s in 1:ngradstate
                local_state_bottom[s] = 2 * local_state[s] - local_state_top[s]
            end
            dzh_bottom = dzh_top
        end

        if bctag_top != 0
            @unroll for s in 1:ngradstate
                local_state_top[s] = 2 * local_state[s] - local_state_bottom[s]
            end
            dzh_top = dzh_bottom
        end

        @unroll for s in 1:ngradstate
            dz = dzh_top + 2dzh + dzh_bottom
            ∇s_v = (local_state_top[s] - local_state_bottom[s]) / dz
            # rotate back to Cartesian
            if increment
                ∇state[vid, O[3 * (s - 1) + 1], e] += ξvx1 * dzh * ∇s_v
                ∇state[vid, O[3 * (s - 1) + 2], e] += ξvx2 * dzh * ∇s_v
                ∇state[vid, O[3 * (s - 1) + 3], e] += ξvx3 * dzh * ∇s_v
            else
                ∇state[vid, O[3 * (s - 1) + 1], e] = ξvx1 * dzh * ∇s_v
                ∇state[vid, O[3 * (s - 1) + 2], e] = ξvx2 * dzh * ∇s_v
                ∇state[vid, O[3 * (s - 1) + 3], e] = ξvx3 * dzh * ∇s_v
            end
        end
    end
end

"""
    kernel_fvm_balance!(f!, balance_law::BalanceLaw, ::Val{nvertelem}, state_auxiliary, vgeo, elems)
    
    To avoid the oscillation in the reference temperature
    ρ[nvertelem] = ρ[nvertelem] * (1 - ϵ)
    to ensure 
    T[nvertelem - 1] ≥ T[nvertelem]

    pᵢ₋₁ - pᵢ =  g ρᵢ Δzᵢ/2 + g ρᵢ₋₁ Δzᵢ₋₁/2
    for i = nvertelem-1:-1:1 
        ρᵢ₋₁  = (pᵢ₋₁ - pᵢ - g ρᵢ Δzᵢ/2) / (g  Δzᵢ₋₁/2)
    end

 - `f!`: update function
 - `balance_law`: atmosphere model
 - `state_auxiliary`: auxiliary variables, ρ is updated 
 - `vgeo`: 2*vgeo[ijk ,_JcV , e] is Δz
 - `elems`: horizontal element list
"""
@kernel function kernel_fvm_balance!(
    f!,
    balance_law::BalanceLaw,
    ::Val{nvertelem},
    state_auxiliary,
    vgeo,
    elems,
) where {nvertelem}
    @uniform begin
        FT = eltype(state_auxiliary)
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        # ρᵢ pᵢ
        local_state_auxiliary_top =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)
        # ρᵢ₋₁ pᵢ₋₁
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

        Δz = MArray{Tuple{2}, FT}(undef)
    end

    _eH = @index(Group, Linear)
    n = @index(Local, Linear)

    @inbounds begin
        eH = elems[_eH]

        # handle top element
        eV = nvertelem
        e = eV + (eH - 1) * nvertelem
        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary_top[s] = state_auxiliary[n, s, e]
        end
        # Δzᵢ
        Δz[2] = 2 * vgeo[n, _JcV, e]

        # Loop up the stack of elements
        for eV in (nvertelem - 1):-1:1
            e = eV + (eH - 1) * nvertelem

            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[n, s, e]
            end
            # Δzᵢ₋₁
            Δz[1] = 2 * vgeo[n, _JcV, e]

            f!(
                balance_law,
                # ρᵢ₋₁ pᵢ₋₁
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary,
                ),
                # ρᵢ pᵢ
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary_top,
                ),
                Δz,
            )

            # update to the global array
            @unroll for s in 1:num_state_auxiliary
                state_auxiliary[n, s, e] = local_state_auxiliary[s]
            end

            # (ρᵢ₋₁ pᵢ₋₁ Δzᵢ₋₁) -> (ρᵢ pᵢ Δzᵢ) 
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary_top[s] = local_state_auxiliary[s]
            end
            Δz[2] = Δz[1]
        end
    end
end
