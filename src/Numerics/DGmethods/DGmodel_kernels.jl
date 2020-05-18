using .NumericalFluxes:
    NumericalFluxGradient,
    NumericalFluxFirstOrder,
    NumericalFluxSecondOrder,
    numerical_flux_gradient!,
    numerical_flux_first_order!,
    numerical_flux_second_order!,
    numerical_flux_divergence!,
    numerical_flux_higher_order!,
    numerical_boundary_flux_gradient!,
    numerical_boundary_flux_first_order!,
    numerical_boundary_flux_second_order!,
    numerical_boundary_flux_divergence!,
    numerical_boundary_flux_higher_order!

using ..Mesh.Geometry

# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _ξ1x1, _ξ2x1, _ξ3x1 = Grids._ξ1x1, Grids._ξ2x1, Grids._ξ3x1
const _ξ1x2, _ξ2x2, _ξ3x2 = Grids._ξ1x2, Grids._ξ2x2, Grids._ξ3x2
const _ξ1x3, _ξ2x3, _ξ3x3 = Grids._ξ1x3, Grids._ξ2x3, Grids._ξ3x3
const _M, _MI = Grids._M, Grids._MI
const _x1, _x2, _x3 = Grids._x1, Grids._x2, Grids._x3
const _JcV = Grids._JcV

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
    direction,
    tendency,
    state_conservative,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    t,
    ω,
    D,
    elems,
    increment,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(state_conservative)
        num_state_conservative = number_state_conservative(balance_law, FT)
        num_state_gradient_flux = number_state_gradient_flux(balance_law, FT)
        num_state_auxiliary = number_state_auxiliary(balance_law, FT)

        ngradlapstate = num_gradient_laplacian(balance_law, FT)
        nhyperviscstate = num_hyperdiffusive(balance_law, FT)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq

        local_source = MArray{Tuple{num_state_conservative}, FT}(undef)
        local_state_conservative =
            MArray{Tuple{num_state_conservative}, FT}(undef)
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_hyperdiffusion = MArray{Tuple{nhyperviscstate}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_flux = MArray{Tuple{3, num_state_conservative}, FT}(undef)
        local_flux_3 = MArray{Tuple{num_state_conservative}, FT}(undef)
    end

    shared_flux = @localmem FT (2, Nq, Nq, num_state_conservative)
    s_D = @localmem FT (Nq, Nq)

    local_tendency = @private FT (Nqk, num_state_conservative)
    local_MI = @private FT (Nqk,)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        s_D[i, j] = D[i, j]
        @unroll for k in 1:Nqk
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            @unroll for s in 1:num_state_conservative
                local_tendency[k, s] =
                    increment ? tendency[ijk, s, e] : zero(FT)
            end
            local_MI[k] = vgeo[ijk, _MI, e]
        end

        @unroll for k in 1:Nqk
            @synchronize
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))

            M = vgeo[ijk, _M, e]
            ξ1x1 = vgeo[ijk, _ξ1x1, e]
            ξ1x2 = vgeo[ijk, _ξ1x2, e]
            ξ1x3 = vgeo[ijk, _ξ1x3, e]
            if dim == 3 || (dim == 2 && direction isa EveryDirection)
                ξ2x1 = vgeo[ijk, _ξ2x1, e]
                ξ2x2 = vgeo[ijk, _ξ2x2, e]
                ξ2x3 = vgeo[ijk, _ξ2x3, e]
            end
            if dim == 3 && direction isa EveryDirection
                ξ3x1 = vgeo[ijk, _ξ3x1, e]
                ξ3x2 = vgeo[ijk, _ξ3x2, e]
                ξ3x3 = vgeo[ijk, _ξ3x3, e]
            end

            @unroll for s in 1:num_state_conservative
                local_state_conservative[s] = state_conservative[ijk, s, e]
            end

            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
            end

            @unroll for s in 1:num_state_gradient_flux
                local_state_gradient_flux[s] = state_gradient_flux[ijk, s, e]
            end

            @unroll for s in 1:nhyperviscstate
                local_state_hyperdiffusion[s] = Qhypervisc_grad[ijk, s, e]
            end

            fill!(local_flux, -zero(eltype(local_flux)))
            flux_first_order!(
                balance_law,
                Grad{vars_state_conservative(balance_law, FT)}(local_flux),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary,
                ),
                t,
            )

            @unroll for s in 1:num_state_conservative
                shared_flux[1, i, j, s] = local_flux[1, s]
                shared_flux[2, i, j, s] = local_flux[2, s]
                local_flux_3[s] = local_flux[3, s]
            end

            fill!(local_flux, -zero(eltype(local_flux)))
            flux_second_order!(
                balance_law,
                Grad{vars_state_conservative(balance_law, FT)}(local_flux),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative,
                ),
                Vars{vars_state_gradient_flux(balance_law, FT)}(
                    local_state_gradient_flux,
                ),
                Vars{vars_hyperdiffusive(balance_law, FT)}(
                    local_state_hyperdiffusion,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary,
                ),
                t,
            )

            @unroll for s in 1:num_state_conservative
                shared_flux[1, i, j, s] += local_flux[1, s]
                shared_flux[2, i, j, s] += local_flux[2, s]
                local_flux_3[s] += local_flux[3, s]
            end

            # Build "inside metrics" flux
            @unroll for s in 1:num_state_conservative
                F1, F2, F3 = shared_flux[1, i, j, s],
                shared_flux[2, i, j, s],
                local_flux_3[s]

                shared_flux[1, i, j, s] =
                    M * (ξ1x1 * F1 + ξ1x2 * F2 + ξ1x3 * F3)
                if dim == 3 || (dim == 2 && direction isa EveryDirection)
                    shared_flux[2, i, j, s] =
                        M * (ξ2x1 * F1 + ξ2x2 * F2 + ξ2x3 * F3)
                end
                if dim == 3 && direction isa EveryDirection
                    local_flux_3[s] = M * (ξ3x1 * F1 + ξ3x2 * F2 + ξ3x3 * F3)
                end
            end

            if dim == 3 && direction isa EveryDirection
                @unroll for n in 1:Nqk
                    MI = local_MI[n]
                    @unroll for s in 1:num_state_conservative
                        local_tendency[n, s] += MI * s_D[k, n] * local_flux_3[s]
                    end
                end
            end

            fill!(local_source, -zero(eltype(local_source)))
            source!(
                balance_law,
                Vars{vars_state_conservative(balance_law, FT)}(local_source),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative,
                ),
                Vars{vars_state_gradient_flux(balance_law, FT)}(
                    local_state_gradient_flux,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary,
                ),
                t,
                direction,
            )

            @unroll for s in 1:num_state_conservative
                local_tendency[k, s] += local_source[s]
            end
            @synchronize

            # Weak "inside metrics" derivative
            MI = local_MI[k]
            @unroll for s in 1:num_state_conservative
                @unroll for n in 1:Nq
                    # ξ1-grid lines
                    local_tendency[k, s] +=
                        MI * s_D[n, i] * shared_flux[1, n, j, s]

                    # ξ2-grid lines
                    if dim == 3 || (dim == 2 && direction isa EveryDirection)
                        local_tendency[k, s] +=
                            MI * s_D[n, j] * shared_flux[2, i, n, s]
                    end
                end
            end
        end

        @unroll for k in 1:Nqk
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            @unroll for s in 1:num_state_conservative
                tendency[ijk, s, e] = local_tendency[k, s]
            end
        end
    end
end

@kernel function volume_tendency!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction::VerticalDirection,
    tendency,
    state_conservative,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    t,
    ω,
    D,
    elems,
    increment,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(state_conservative)
        num_state_conservative = number_state_conservative(balance_law, FT)
        num_state_gradient_flux = number_state_gradient_flux(balance_law, FT)
        num_state_auxiliary = number_state_auxiliary(balance_law, FT)

        ngradlapstate = num_gradient_laplacian(balance_law, FT)
        nhyperviscstate = num_hyperdiffusive(balance_law, FT)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq

        local_source = MArray{Tuple{num_state_conservative}, FT}(undef)
        local_state_conservative =
            MArray{Tuple{num_state_conservative}, FT}(undef)
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_hyperdiffusion = MArray{Tuple{nhyperviscstate}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_flux = MArray{Tuple{3, num_state_conservative}, FT}(undef)
        local_flux_total = MArray{Tuple{3, num_state_conservative}, FT}(undef)

        _ζx1 = dim == 2 ? _ξ2x1 : _ξ3x1
        _ζx2 = dim == 2 ? _ξ2x2 : _ξ3x2
        _ζx3 = dim == 2 ? _ξ2x3 : _ξ3x3

        shared_flux_size =
            dim == 2 ? (Nq, Nq, num_state_conservative) : (0, 0, 0)
    end

    local_tendency = @private FT (Nqk, num_state_conservative)
    local_MI = @private FT (Nqk,)

    shared_flux = @localmem FT shared_flux_size
    s_D = @localmem FT (Nq, Nq)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        s_D[i, j] = D[i, j]

        @unroll for k in 1:Nqk
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            @unroll for s in 1:num_state_conservative
                local_tendency[k, s] =
                    increment ? tendency[ijk, s, e] : zero(FT)
            end
            local_MI[k] = vgeo[ijk, _MI, e]
        end

        @unroll for k in 1:Nqk
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))

            M = vgeo[ijk, _M, e]
            ζx1 = vgeo[ijk, _ζx1, e]
            ζx2 = vgeo[ijk, _ζx2, e]
            ζx3 = vgeo[ijk, _ζx3, e]

            @unroll for s in 1:num_state_conservative
                local_state_conservative[s] = state_conservative[ijk, s, e]
            end

            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
            end

            @unroll for s in 1:num_state_gradient_flux
                local_state_gradient_flux[s] = state_gradient_flux[ijk, s, e]
            end

            @unroll for s in 1:nhyperviscstate
                local_state_hyperdiffusion[s] = Qhypervisc_grad[ijk, s, e]
            end

            fill!(local_flux, -zero(eltype(local_flux)))
            flux_first_order!(
                balance_law,
                Grad{vars_state_conservative(balance_law, FT)}(local_flux),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary,
                ),
                t,
            )

            @unroll for s in 1:num_state_conservative
                local_flux_total[1, s] = local_flux[1, s]
                local_flux_total[2, s] = local_flux[2, s]
                local_flux_total[3, s] = local_flux[3, s]
            end

            fill!(local_flux, -zero(eltype(local_flux)))
            flux_second_order!(
                balance_law,
                Grad{vars_state_conservative(balance_law, FT)}(local_flux),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative,
                ),
                Vars{vars_state_gradient_flux(balance_law, FT)}(
                    local_state_gradient_flux,
                ),
                Vars{vars_hyperdiffusive(balance_law, FT)}(
                    local_state_hyperdiffusion,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary,
                ),
                t,
            )

            @unroll for s in 1:num_state_conservative
                local_flux_total[1, s] += local_flux[1, s]
                local_flux_total[2, s] += local_flux[2, s]
                local_flux_total[3, s] += local_flux[3, s]
            end

            # Build "inside metrics" flux
            @unroll for s in 1:num_state_conservative
                F1, F2, F3 = local_flux_total[1, s],
                local_flux_total[2, s],
                local_flux_total[3, s]
                Fv = M * (ζx1 * F1 + ζx2 * F2 + ζx3 * F3)
                if dim == 2
                    shared_flux[i, j, s] = Fv
                else
                    local_flux_total[1, s] = Fv
                end
            end

            if dim == 3
                @unroll for n in 1:Nqk
                    MI = local_MI[n]
                    @unroll for s in 1:num_state_conservative
                        local_tendency[n, s] +=
                            MI * s_D[k, n] * local_flux_total[1, s]
                    end
                end
            end

            fill!(local_source, -zero(eltype(local_source)))
            source!(
                balance_law,
                Vars{vars_state_conservative(balance_law, FT)}(local_source),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative,
                ),
                Vars{vars_state_gradient_flux(balance_law, FT)}(
                    local_state_gradient_flux,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary,
                ),
                t,
                direction,
            )

            @unroll for s in 1:num_state_conservative
                local_tendency[k, s] += local_source[s]
            end

            @synchronize(dim == 2)
            if dim == 2
                MI = local_MI[k]
                @unroll for n in 1:Nq
                    @unroll for s in 1:num_state_conservative
                        local_tendency[k, s] +=
                            MI * s_D[n, j] * shared_flux[i, n, s]
                    end
                end
            end
        end

        @unroll for k in 1:Nqk
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            @unroll for s in 1:num_state_conservative
                tendency[ijk, s, e] = local_tendency[k, s]
            end
        end
    end
end

@doc """
    interface_tendency!(balance_law::BalanceLaw, Val(polyorder),
            numerical_flux_first_order::NumericalFluxFirstOrder,
            numerical_flux_second_order::NumericalFluxSecondOrder,
            tendency, state_conservative, state_gradient_flux, state_auxiliary,
            vgeo, sgeo, t, vmap⁻, vmap⁺, elemtobndy,
            elems)

Computational kernel: Evaluate the surface integrals on right-hand side of a
`BalanceLaw` semi-discretization.
""" interface_tendency!
@kernel function interface_tendency!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    numerical_flux_first_order::NumericalFluxFirstOrder,
    numerical_flux_second_order::NumericalFluxSecondOrder,
    tendency,
    state_conservative,
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
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(state_conservative)
        num_state_conservative = number_state_conservative(balance_law, FT)
        num_state_gradient_flux = number_state_gradient_flux(balance_law, FT)
        nhyperviscstate = num_hyperdiffusive(balance_law, FT)
        num_state_auxiliary = number_state_auxiliary(balance_law, FT)
        ngradlapstate = num_gradient_laplacian(balance_law, FT)

        if dim == 1
            Np = (N + 1)
            Nfp = 1
            nface = 2
        elseif dim == 2
            Np = (N + 1) * (N + 1)
            Nfp = (N + 1)
            nface = 4
        elseif dim == 3
            Np = (N + 1) * (N + 1) * (N + 1)
            Nfp = (N + 1) * (N + 1)
            nface = 6
        end

        faces = 1:nface
        if direction isa VerticalDirection
            faces = (nface - 1):nface
        elseif direction isa HorizontalDirection
            faces = 1:(nface - 2)
        end

        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq

        local_state_conservative⁻ =
            MArray{Tuple{num_state_conservative}, FT}(undef)
        local_state_gradient_flux⁻ =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_hyperdiffusion⁻ = MArray{Tuple{nhyperviscstate}, FT}(undef)
        local_state_auxiliary⁻ = MArray{Tuple{num_state_auxiliary}, FT}(undef)

        # Need two copies since numerical_flux_first_order! can modify state_conservative⁺
        local_state_conservative⁺nondiff =
            MArray{Tuple{num_state_conservative}, FT}(undef)
        local_state_conservative⁺diff =
            MArray{Tuple{num_state_conservative}, FT}(undef)

        # Need two copies since numerical_flux_first_order! can modify state_auxiliary⁺
        local_state_auxiliary⁺nondiff =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_auxiliary⁺diff =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_state_gradient_flux⁺ =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_hyperdiffusion⁺ = MArray{Tuple{nhyperviscstate}, FT}(undef)

        local_state_conservative_bottom1 =
            MArray{Tuple{num_state_conservative}, FT}(undef)
        local_state_gradient_flux_bottom1 =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_auxiliary_bottom1 =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_flux = MArray{Tuple{num_state_conservative}, FT}(undef)
    end

    eI = @index(Group, Linear)
    n = @index(Local, Linear)

    e = @private Int (1,)
    @inbounds e[1] = elems[eI]

    @inbounds for f in faces
        e⁻ = e[1]
        normal_vector = SVector(
            sgeo[_n1, n, f, e⁻],
            sgeo[_n2, n, f, e⁻],
            sgeo[_n3, n, f, e⁻],
        )
        sM, vMI = sgeo[_sM, n, f, e⁻], sgeo[_vMI, n, f, e⁻]
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1

        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1

        # Load minus side data
        @unroll for s in 1:num_state_conservative
            local_state_conservative⁻[s] = state_conservative[vid⁻, s, e⁻]
        end

        @unroll for s in 1:num_state_gradient_flux
            local_state_gradient_flux⁻[s] = state_gradient_flux[vid⁻, s, e⁻]
        end

        @unroll for s in 1:nhyperviscstate
            local_state_hyperdiffusion⁻[s] = Qhypervisc_grad[vid⁻, s, e⁻]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁻[s] = state_auxiliary[vid⁻, s, e⁻]
        end

        # Load plus side data
        @unroll for s in 1:num_state_conservative
            local_state_conservative⁺diff[s] =
                local_state_conservative⁺nondiff[s] =
                    state_conservative[vid⁺, s, e⁺]
        end

        @unroll for s in 1:num_state_gradient_flux
            local_state_gradient_flux⁺[s] = state_gradient_flux[vid⁺, s, e⁺]
        end

        @unroll for s in 1:nhyperviscstate
            local_state_hyperdiffusion⁺[s] = Qhypervisc_grad[vid⁺, s, e⁺]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁺diff[s] =
                local_state_auxiliary⁺nondiff[s] = state_auxiliary[vid⁺, s, e⁺]
        end

        bctype = elemtobndy[f, e⁻]
        fill!(local_flux, -zero(eltype(local_flux)))
        if bctype == 0
            numerical_flux_first_order!(
                numerical_flux_first_order,
                balance_law,
                Vars{vars_state_conservative(balance_law, FT)}(local_flux),
                SVector(normal_vector),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁻,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁺nondiff,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁺nondiff,
                ),
                t,
            )
            numerical_flux_second_order!(
                numerical_flux_second_order,
                balance_law,
                Vars{vars_state_conservative(balance_law, FT)}(local_flux),
                normal_vector,
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁻,
                ),
                Vars{vars_state_gradient_flux(balance_law, FT)}(
                    local_state_gradient_flux⁻,
                ),
                Vars{vars_hyperdiffusive(balance_law, FT)}(
                    local_state_hyperdiffusion⁻,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁺diff,
                ),
                Vars{vars_state_gradient_flux(balance_law, FT)}(
                    local_state_gradient_flux⁺,
                ),
                Vars{vars_hyperdiffusive(balance_law, FT)}(
                    local_state_hyperdiffusion⁺,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁺diff,
                ),
                t,
            )
        else
            if (dim == 2 && f == 3) || (dim == 3 && f == 5)
                # Loop up the first element along all horizontal elements
                @unroll for s in 1:num_state_conservative
                    local_state_conservative_bottom1[s] =
                        state_conservative[n + Nqk^2, s, e⁻]
                end
                @unroll for s in 1:num_state_gradient_flux
                    local_state_gradient_flux_bottom1[s] =
                        state_gradient_flux[n + Nqk^2, s, e⁻]
                end
                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary_bottom1[s] =
                        state_auxiliary[n + Nqk^2, s, e⁻]
                end
            end
            numerical_boundary_flux_first_order!(
                numerical_flux_first_order,
                balance_law,
                Vars{vars_state_conservative(balance_law, FT)}(local_flux),
                SVector(normal_vector),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁻,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁺nondiff,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁺nondiff,
                ),
                bctype,
                t,
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative_bottom1,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary_bottom1,
                ),
            )
            numerical_boundary_flux_second_order!(
                numerical_flux_second_order,
                balance_law,
                Vars{vars_state_conservative(balance_law, FT)}(local_flux),
                normal_vector,
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁻,
                ),
                Vars{vars_state_gradient_flux(balance_law, FT)}(
                    local_state_gradient_flux⁻,
                ),
                Vars{vars_hyperdiffusive(balance_law, FT)}(
                    local_state_hyperdiffusion⁻,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁺diff,
                ),
                Vars{vars_state_gradient_flux(balance_law, FT)}(
                    local_state_gradient_flux⁺,
                ),
                Vars{vars_hyperdiffusive(balance_law, FT)}(
                    local_state_hyperdiffusion⁺,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁺diff,
                ),
                bctype,
                t,
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative_bottom1,
                ),
                Vars{vars_state_gradient_flux(balance_law, FT)}(
                    local_state_gradient_flux_bottom1,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary_bottom1,
                ),
            )
        end

        #Update RHS
        @unroll for s in 1:num_state_conservative
            # FIXME: Should we pretch these?
            tendency[vid⁻, s, e⁻] -= vMI * sM * local_flux[s]
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

@kernel function volume_gradients!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    state_conservative,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    t,
    D,
    hypervisc_indexmap,
    elems,
) where {dim, polyorder}
    @uniform begin
        N = polyorder

        FT = eltype(state_conservative)
        num_state_conservative = number_state_conservative(balance_law, FT)
        ngradstate = number_state_gradient(balance_law, FT)
        ngradlapstate = num_gradient_laplacian(balance_law, FT)
        num_state_gradient_flux = number_state_gradient_flux(balance_law, FT)
        num_state_auxiliary = number_state_auxiliary(balance_law, FT)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq

        ngradtransformstate = num_state_conservative

        local_transform = MArray{Tuple{ngradstate}, FT}(undef)
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
    end

    shared_transform = @localmem FT (Nq, Nq, ngradstate)
    s_D = @localmem FT (Nq, Nq)

    local_state_conservative = @private FT (ngradtransformstate, Nqk)
    local_state_auxiliary = @private FT (num_state_auxiliary, Nqk)
    local_transform_gradient = @private FT (3, ngradstate, Nqk)
    Gξ3 = @private FT (ngradstate, Nqk)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds @views begin
        s_D[i, j] = D[i, j]

        @unroll for k in 1:Nqk
            @unroll for s in 1:ngradstate
                local_transform_gradient[1, s, k] = -zero(FT)
                local_transform_gradient[2, s, k] = -zero(FT)
                local_transform_gradient[3, s, k] = -zero(FT)
                Gξ3[s, k] = -zero(FT)
            end
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            @unroll for s in 1:ngradtransformstate
                local_state_conservative[s, k] = state_conservative[ijk, s, e]
            end
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s, k] = state_auxiliary[ijk, s, e]
            end
        end

        @unroll for k in 1:Nqk
            fill!(local_transform, -zero(eltype(local_transform)))
            compute_gradient_argument!(
                balance_law,
                Vars{vars_state_gradient(balance_law, FT)}(local_transform),
                Vars{vars_state_conservative(balance_law, FT)}(local_state_conservative[
                    :,
                    k,
                ]),
                Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary[
                    :,
                    k,
                ]),
                t,
            )
            @unroll for s in 1:ngradstate
                shared_transform[i, j, s] = local_transform[s]
            end
            @synchronize

            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            ξ1x1, ξ1x2, ξ1x3 =
                vgeo[ijk, _ξ1x1, e], vgeo[ijk, _ξ1x2, e], vgeo[ijk, _ξ1x3, e]

            # Compute gradient of each state
            @unroll for s in 1:ngradstate
                Gξ1 = Gξ2 = zero(FT)

                @unroll for n in 1:Nq
                    Gξ1 += s_D[i, n] * shared_transform[n, j, s]
                    if dim == 3 || (dim == 2 && direction isa EveryDirection)
                        Gξ2 += s_D[j, n] * shared_transform[i, n, s]
                    end
                    if dim == 3 && direction isa EveryDirection
                        Gξ3[s, n] += s_D[n, k] * shared_transform[i, j, s]
                    end
                end

                local_transform_gradient[1, s, k] += ξ1x1 * Gξ1
                local_transform_gradient[2, s, k] += ξ1x2 * Gξ1
                local_transform_gradient[3, s, k] += ξ1x3 * Gξ1

                if dim == 3 || (dim == 2 && direction isa EveryDirection)
                    ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e],
                    vgeo[ijk, _ξ2x2, e],
                    vgeo[ijk, _ξ2x3, e]
                    local_transform_gradient[1, s, k] += ξ2x1 * Gξ2
                    local_transform_gradient[2, s, k] += ξ2x2 * Gξ2
                    local_transform_gradient[3, s, k] += ξ2x3 * Gξ2
                end
            end
            @synchronize
        end

        @unroll for k in 1:Nqk
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))

            if dim == 3 && direction isa EveryDirection
                ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e],
                vgeo[ijk, _ξ3x2, e],
                vgeo[ijk, _ξ3x3, e]
                @unroll for s in 1:ngradstate
                    local_transform_gradient[1, s, k] += ξ3x1 * Gξ3[s, k]
                    local_transform_gradient[2, s, k] += ξ3x2 * Gξ3[s, k]
                    local_transform_gradient[3, s, k] += ξ3x3 * Gξ3[s, k]
                end
            end

            @unroll for s in 1:ngradlapstate
                Qhypervisc_grad[ijk, 3 * (s - 1) + 1, e] =
                    local_transform_gradient[1, hypervisc_indexmap[s], k]
                Qhypervisc_grad[ijk, 3 * (s - 1) + 2, e] =
                    local_transform_gradient[2, hypervisc_indexmap[s], k]
                Qhypervisc_grad[ijk, 3 * (s - 1) + 3, e] =
                    local_transform_gradient[3, hypervisc_indexmap[s], k]
            end

            if num_state_gradient_flux > 0
                fill!(
                    local_state_gradient_flux,
                    -zero(eltype(local_state_gradient_flux)),
                )
                compute_gradient_flux!(
                    balance_law,
                    Vars{vars_state_gradient_flux(balance_law, FT)}(
                        local_state_gradient_flux,
                    ),
                    Grad{vars_state_gradient(balance_law, FT)}(local_transform_gradient[
                        :,
                        :,
                        k,
                    ]),
                    Vars{vars_state_conservative(balance_law, FT)}(local_state_conservative[
                        :,
                        k,
                    ]),
                    Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary[
                        :,
                        k,
                    ]),
                    t,
                )

                @unroll for s in 1:num_state_gradient_flux
                    state_gradient_flux[ijk, s, e] =
                        local_state_gradient_flux[s]
                end
            end
        end
    end
end

@kernel function volume_gradients!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    ::VerticalDirection,
    state_conservative,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    t,
    D,
    hypervisc_indexmap,
    elems,
) where {dim, polyorder}
    @uniform begin
        N = polyorder

        FT = eltype(state_conservative)
        num_state_conservative = number_state_conservative(balance_law, FT)
        ngradstate = number_state_gradient(balance_law, FT)
        ngradlapstate = num_gradient_laplacian(balance_law, FT)
        num_state_gradient_flux = number_state_gradient_flux(balance_law, FT)
        num_state_auxiliary = number_state_auxiliary(balance_law, FT)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq

        ngradtransformstate = num_state_conservative

        local_transform = MArray{Tuple{ngradstate}, FT}(undef)
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)

        _ζx1 = dim == 2 ? _ξ2x1 : _ξ3x1
        _ζx2 = dim == 2 ? _ξ2x2 : _ξ3x2
        _ζx3 = dim == 2 ? _ξ2x3 : _ξ3x3

        Gζ_size = dim == 3 ? (ngradstate, Nqk) : (0, 0)
    end

    shared_transform = @localmem FT (Nq, Nq, ngradstate)
    s_D = @localmem FT (Nq, Nq)

    local_state_conservative = @private FT (ngradtransformstate, Nqk)
    local_state_auxiliary = @private FT (num_state_auxiliary, Nqk)
    local_transform_gradient = @private FT (3, ngradstate, Nqk)

    local_ζ = @private FT (3, Nqk)

    Gζ = @private FT Gζ_size

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds @views begin
        s_D[i, j] = D[i, j]

        @unroll for k in 1:Nqk
            @unroll for s in 1:ngradstate
                local_transform_gradient[1, s, k] = -zero(FT)
                local_transform_gradient[2, s, k] = -zero(FT)
                local_transform_gradient[3, s, k] = -zero(FT)
                if dim == 3
                    Gζ[s, k] = -zero(FT)
                end
            end
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            @unroll for s in 1:ngradtransformstate
                local_state_conservative[s, k] = state_conservative[ijk, s, e]
            end
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s, k] = state_auxiliary[ijk, s, e]
            end
            local_ζ[1, k] = vgeo[ijk, _ζx1, e]
            local_ζ[2, k] = vgeo[ijk, _ζx2, e]
            local_ζ[3, k] = vgeo[ijk, _ζx3, e]
        end

        @unroll for k in 1:Nqk
            fill!(local_transform, -zero(eltype(local_transform)))
            compute_gradient_argument!(
                balance_law,
                Vars{vars_state_gradient(balance_law, FT)}(local_transform),
                Vars{vars_state_conservative(balance_law, FT)}(local_state_conservative[
                    :,
                    k,
                ]),
                Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary[
                    :,
                    k,
                ]),
                t,
            )
            @unroll for s in 1:ngradstate
                shared_transform[i, j, s] = local_transform[s]
            end
            @synchronize

            # Compute gradient of each state
            @unroll for s in 1:ngradstate
                if dim == 2
                    Gζ = zero(FT)
                    @unroll for n in 1:Nq
                        Gζ += s_D[j, n] * shared_transform[i, n, s]
                    end
                    local_transform_gradient[1, s, k] += local_ζ[1, k] * Gζ
                    local_transform_gradient[2, s, k] += local_ζ[2, k] * Gζ
                    local_transform_gradient[3, s, k] += local_ζ[3, k] * Gζ
                else
                    @unroll for n in 1:Nq
                        Gζ[s, n] += s_D[n, k] * shared_transform[i, j, s]
                    end
                end
            end
            @synchronize
        end

        @unroll for k in 1:Nqk
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))

            if dim == 3
                @unroll for s in 1:ngradstate
                    local_transform_gradient[1, s, k] +=
                        local_ζ[1, k] * Gζ[s, k]
                    local_transform_gradient[2, s, k] +=
                        local_ζ[2, k] * Gζ[s, k]
                    local_transform_gradient[3, s, k] +=
                        local_ζ[3, k] * Gζ[s, k]
                end
            end

            @unroll for s in 1:ngradlapstate
                Qhypervisc_grad[ijk, 3 * (s - 1) + 1, e] =
                    local_transform_gradient[1, hypervisc_indexmap[s], k]
                Qhypervisc_grad[ijk, 3 * (s - 1) + 2, e] =
                    local_transform_gradient[2, hypervisc_indexmap[s], k]
                Qhypervisc_grad[ijk, 3 * (s - 1) + 3, e] =
                    local_transform_gradient[3, hypervisc_indexmap[s], k]
            end

            if num_state_gradient_flux > 0
                fill!(
                    local_state_gradient_flux,
                    -zero(eltype(local_state_gradient_flux)),
                )
                compute_gradient_flux!(
                    balance_law,
                    Vars{vars_state_gradient_flux(balance_law, FT)}(
                        local_state_gradient_flux,
                    ),
                    Grad{vars_state_gradient(balance_law, FT)}(local_transform_gradient[
                        :,
                        :,
                        k,
                    ]),
                    Vars{vars_state_conservative(balance_law, FT)}(local_state_conservative[
                        :,
                        k,
                    ]),
                    Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary[
                        :,
                        k,
                    ]),
                    t,
                )

                @unroll for s in 1:num_state_gradient_flux
                    state_gradient_flux[ijk, s, e] =
                        local_state_gradient_flux[s]
                end
            end
        end
    end
end

@kernel function interface_gradients!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    numerical_flux_gradient::NumericalFluxGradient,
    state_conservative,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    sgeo,
    t,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    hypervisc_indexmap,
    elems,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(state_conservative)
        num_state_conservative = number_state_conservative(balance_law, FT)
        ngradstate = number_state_gradient(balance_law, FT)
        ngradlapstate = num_gradient_laplacian(balance_law, FT)
        num_state_gradient_flux = number_state_gradient_flux(balance_law, FT)
        num_state_auxiliary = number_state_auxiliary(balance_law, FT)

        if dim == 1
            Np = (N + 1)
            Nfp = 1
            nface = 2
        elseif dim == 2
            Np = (N + 1) * (N + 1)
            Nfp = (N + 1)
            nface = 4
        elseif dim == 3
            Np = (N + 1) * (N + 1) * (N + 1)
            Nfp = (N + 1) * (N + 1)
            nface = 6
        end

        faces = 1:nface
        if direction isa VerticalDirection
            faces = (nface - 1):nface
        elseif direction isa HorizontalDirection
            faces = 1:(nface - 2)
        end

        Nqk = dim == 2 ? 1 : N + 1

        ngradtransformstate = num_state_conservative

        local_state_conservative⁻ =
            MArray{Tuple{ngradtransformstate}, FT}(undef)
        local_state_auxiliary⁻ = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_transform⁻ = MArray{Tuple{ngradstate}, FT}(undef)
        l_nG⁻ = MArray{Tuple{3, ngradstate}, FT}(undef)

        local_state_conservative⁺ =
            MArray{Tuple{ngradtransformstate}, FT}(undef)
        local_state_auxiliary⁺ = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_transform⁺ = MArray{Tuple{ngradstate}, FT}(undef)

        # FIXME state_gradient_flux is sort of a terrible name...
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_transform_gradient = MArray{Tuple{3, ngradstate}, FT}(undef)
        local_state_conservative⁻visc =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)

        local_state_conservative_bottom1 =
            MArray{Tuple{num_state_conservative}, FT}(undef)
        local_state_auxiliary_bottom1 =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)
    end

    eI = @index(Group, Linear)
    n = @index(Local, Linear)

    e = @private Int (1,)
    @inbounds e[1] = elems[eI]

    @inbounds for f in faces
        e⁻ = e[1]
        normal_vector = SVector(
            sgeo[_n1, n, f, e⁻],
            sgeo[_n2, n, f, e⁻],
            sgeo[_n3, n, f, e⁻],
        )
        sM, vMI = sgeo[_sM, n, f, e⁻], sgeo[_vMI, n, f, e⁻]
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1

        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1

        # Load minus side data
        @unroll for s in 1:ngradtransformstate
            local_state_conservative⁻[s] = state_conservative[vid⁻, s, e⁻]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁻[s] = state_auxiliary[vid⁻, s, e⁻]
        end

        fill!(local_transform⁻, -zero(eltype(local_transform⁻)))
        compute_gradient_argument!(
            balance_law,
            Vars{vars_state_gradient(balance_law, FT)}(local_transform⁻),
            Vars{vars_state_conservative(balance_law, FT)}(
                local_state_conservative⁻,
            ),
            Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary⁻),
            t,
        )

        # Load plus side data
        @unroll for s in 1:ngradtransformstate
            local_state_conservative⁺[s] = state_conservative[vid⁺, s, e⁺]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁺[s] = state_auxiliary[vid⁺, s, e⁺]
        end

        fill!(local_transform⁺, -zero(eltype(local_transform⁺)))
        compute_gradient_argument!(
            balance_law,
            Vars{vars_state_gradient(balance_law, FT)}(local_transform⁺),
            Vars{vars_state_conservative(balance_law, FT)}(
                local_state_conservative⁺,
            ),
            Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary⁺),
            t,
        )

        bctype = elemtobndy[f, e⁻]
        fill!(
            local_state_gradient_flux,
            -zero(eltype(local_state_gradient_flux)),
        )
        if bctype == 0
            numerical_flux_gradient!(
                numerical_flux_gradient,
                balance_law,
                local_transform_gradient,
                SVector(normal_vector),
                Vars{vars_state_gradient(balance_law, FT)}(local_transform⁻),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁻,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_state_gradient(balance_law, FT)}(local_transform⁺),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁺,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁺,
                ),
                t,
            )
            if num_state_gradient_flux > 0
                compute_gradient_flux!(
                    balance_law,
                    Vars{vars_state_gradient_flux(balance_law, FT)}(
                        local_state_gradient_flux,
                    ),
                    Grad{vars_state_gradient(balance_law, FT)}(
                        local_transform_gradient,
                    ),
                    Vars{vars_state_conservative(balance_law, FT)}(
                        local_state_conservative⁻,
                    ),
                    Vars{vars_state_auxiliary(balance_law, FT)}(
                        local_state_auxiliary⁻,
                    ),
                    t,
                )
            end
        else
            if (dim == 2 && f == 3) || (dim == 3 && f == 5)
                # Loop up the first element along all horizontal elements
                @unroll for s in 1:num_state_conservative
                    local_state_conservative_bottom1[s] =
                        state_conservative[n + Nqk^2, s, e⁻]
                end
                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary_bottom1[s] =
                        state_auxiliary[n + Nqk^2, s, e⁻]
                end
            end
            numerical_boundary_flux_gradient!(
                numerical_flux_gradient,
                balance_law,
                local_transform_gradient,
                SVector(normal_vector),
                Vars{vars_state_gradient(balance_law, FT)}(local_transform⁻),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁻,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_state_gradient(balance_law, FT)}(local_transform⁺),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁺,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁺,
                ),
                bctype,
                t,
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative_bottom1,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary_bottom1,
                ),
            )
            if num_state_gradient_flux > 0
                compute_gradient_flux!(
                    balance_law,
                    Vars{vars_state_gradient_flux(balance_law, FT)}(
                        local_state_gradient_flux,
                    ),
                    Grad{vars_state_gradient(balance_law, FT)}(
                        local_transform_gradient,
                    ),
                    Vars{vars_state_conservative(balance_law, FT)}(
                        local_state_conservative⁻,
                    ),
                    Vars{vars_state_auxiliary(balance_law, FT)}(
                        local_state_auxiliary⁻,
                    ),
                    t,
                )
            end
        end

        @unroll for j in 1:ngradstate
            @unroll for i in 1:3
                l_nG⁻[i, j] = normal_vector[i] * local_transform⁻[j]
            end
        end

        @unroll for s in 1:ngradlapstate
            j = hypervisc_indexmap[s]
            Qhypervisc_grad[vid⁻, 3 * (s - 1) + 1, e⁻] +=
                vMI * sM * (local_transform_gradient[1, j] - l_nG⁻[1, j])
            Qhypervisc_grad[vid⁻, 3 * (s - 1) + 2, e⁻] +=
                vMI * sM * (local_transform_gradient[2, j] - l_nG⁻[2, j])
            Qhypervisc_grad[vid⁻, 3 * (s - 1) + 3, e⁻] +=
                vMI * sM * (local_transform_gradient[3, j] - l_nG⁻[3, j])
        end

        compute_gradient_flux!(
            balance_law,
            Vars{vars_state_gradient_flux(balance_law, FT)}(
                local_state_conservative⁻visc,
            ),
            Grad{vars_state_gradient(balance_law, FT)}(l_nG⁻),
            Vars{vars_state_conservative(balance_law, FT)}(
                local_state_conservative⁻,
            ),
            Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary⁻),
            t,
        )


        @unroll for s in 1:num_state_gradient_flux
            state_gradient_flux[vid⁻, s, e⁻] +=
                vMI *
                sM *
                (
                    local_state_gradient_flux[s] -
                    local_state_conservative⁻visc[s]
                )
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

@kernel function kernel_init_state_conservative!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    state,
    state_auxiliary,
    vgeo,
    elems,
    args...,
) where {dim, polyorder}
    N = polyorder
    FT = eltype(state_auxiliary)
    num_state_auxiliary = number_state_auxiliary(balance_law, FT)
    num_state_conservative = number_state_conservative(balance_law, FT)

    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Np = Nq * Nq * Nqk

    l_state = MArray{Tuple{num_state_conservative}, FT}(undef)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

    I = @index(Global, Linear)
    e = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        coords = SVector(vgeo[n, _x1, e], vgeo[n, _x2, e], vgeo[n, _x3, e])
        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end
        @unroll for s in 1:num_state_conservative
            l_state[s] = state[n, s, e]
        end
        init_state_conservative!(
            balance_law,
            Vars{vars_state_conservative(balance_law, FT)}(l_state),
            Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary),
            coords,
            args...,
        )
        @unroll for s in 1:num_state_conservative
            state[n, s, e] = l_state[s]
        end
    end
end


@doc """
    kernel_init_state_auxiliary!(balance_law::BalanceLaw, Val(polyorder), state_auxiliary, vgeo, elems)

Computational kernel: Initialize the auxiliary state

See [`BalanceLaw`](@ref) for usage.
""" kernel_init_state_auxiliary!
@kernel function kernel_init_state_auxiliary!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    state_auxiliary,
    vgeo,
    elems,
) where {dim, polyorder}
    N = polyorder
    FT = eltype(state_auxiliary)
    num_state_auxiliary = number_state_auxiliary(balance_law, FT)

    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Np = Nq * Nq * Nqk

    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

    I = @index(Global, Linear)
    e = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end

        init_state_auxiliary!(
            balance_law,
            Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary),
            LocalGeometry(Val(polyorder), vgeo, n, e),
        )

        @unroll for s in 1:num_state_auxiliary
            state_auxiliary[n, s, e] = local_state_auxiliary[s]
        end
    end
end

@doc """
    kernel_nodal_update_auxiliary_state!(balance_law::BalanceLaw, ::Val{dim}, ::Val{N}, f!, state_conservative, state_auxiliary, [state_gradient_flux,]
                          t, elems, activedofs) where {dim, N}

Update the auxiliary state array
""" kernel_nodal_update_auxiliary_state!
@kernel function kernel_nodal_update_auxiliary_state!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    f!,
    state_conservative,
    state_auxiliary,
    t,
    elems,
    activedofs,
) where {dim, N}
    FT = eltype(state_conservative)
    num_state_conservative = number_state_conservative(balance_law, FT)
    num_state_auxiliary = number_state_auxiliary(balance_law, FT)

    Nq = N + 1

    Nqk = dim == 2 ? 1 : Nq

    Np = Nq * Nq * Nqk

    local_state_conservative = MArray{Tuple{num_state_conservative}, FT}(undef)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

    I = @index(Global, Linear)
    eI = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        e = elems[eI]

        active = activedofs[n + (e - 1) * Np]

        if active
            @unroll for s in 1:num_state_conservative
                local_state_conservative[s] = state_conservative[n, s, e]
            end

            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[n, s, e]
            end

            f!(
                balance_law,
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary,
                ),
                t,
            )

            @unroll for s in 1:num_state_auxiliary
                state_auxiliary[n, s, e] = local_state_auxiliary[s]
            end
        end
    end
end

@kernel function kernel_nodal_update_auxiliary_state!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    f!,
    state_conservative,
    state_auxiliary,
    state_gradient_flux,
    t,
    elems,
    activedofs,
) where {dim, N}
    FT = eltype(state_conservative)
    num_state_conservative = number_state_conservative(balance_law, FT)
    num_state_gradient_flux = number_state_gradient_flux(balance_law, FT)
    num_state_auxiliary = number_state_auxiliary(balance_law, FT)

    Nq = N + 1

    Nqk = dim == 2 ? 1 : Nq

    Np = Nq * Nq * Nqk

    local_state_conservative = MArray{Tuple{num_state_conservative}, FT}(undef)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
    local_state_gradient_flux =
        MArray{Tuple{num_state_gradient_flux}, FT}(undef)

    I = @index(Global, Linear)
    eI = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        e = elems[eI]

        active = activedofs[n + (e - 1) * Np]

        if active
            @unroll for s in 1:num_state_conservative
                local_state_conservative[s] = state_conservative[n, s, e]
            end

            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[n, s, e]
            end

            @unroll for s in 1:num_state_gradient_flux
                local_state_gradient_flux[s] = state_gradient_flux[n, s, e]
            end

            f!(
                balance_law,
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary,
                ),
                Vars{vars_state_gradient_flux(balance_law, FT)}(
                    local_state_gradient_flux,
                ),
                t,
            )

            @unroll for s in 1:num_state_auxiliary
                state_auxiliary[n, s, e] = local_state_auxiliary[s]
            end
        end
    end
end

@doc """
    kernel_indefinite_stack_integral!(balance_law::BalanceLaw, ::Val{dim}, ::Val{N},
                                  ::Val{nvertelem}, state_conservative, state_auxiliary, vgeo,
                                  Imat, elems) where {dim, N, nvertelem}
Computational kernel: compute indefinite integral along the vertical stack
See [`BalanceLaw`](@ref) for usage.
""" kernel_indefinite_stack_integral!
@kernel function kernel_indefinite_stack_integral!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    ::Val{nvertelem},
    state_conservative,
    state_auxiliary,
    vgeo,
    Imat,
    elems,
) where {dim, N, nvertelem}
    @uniform begin
        FT = eltype(state_conservative)
        num_state_conservative = number_state_conservative(balance_law, FT)
        num_state_auxiliary = number_state_auxiliary(balance_law, FT)
        nout = num_integrals(balance_law, FT)

        Nq = N + 1
        Nqj = dim == 2 ? 1 : Nq

        local_state_conservative =
            MArray{Tuple{num_state_conservative}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_kernel = MArray{Tuple{nout, Nq}, FT}(undef)
    end

    local_integral = @private FT (nout, Nq)
    s_I = @localmem FT (Nq, Nq)

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        @unroll for n in 1:Nq
            s_I[i, n] = Imat[i, n]
        end
        @synchronize

        # Initialize the constant state at zero
        @unroll for k in 1:Nq
            @unroll for s in 1:nout
                local_integral[s, k] = 0
            end
        end

        eh = elems[_eh]

        # Loop up the stack of elements
        for ev in 1:nvertelem
            e = ev + (eh - 1) * nvertelem

            # Evaluate the integral kernel at each DOF in the slabk
            # loop up the pencil
            @unroll for k in 1:Nq
                ijk = i + Nq * ((j - 1) + Nqj * (k - 1))
                Jc = vgeo[ijk, _JcV, e]
                @unroll for s in 1:num_state_conservative
                    local_state_conservative[s] = state_conservative[ijk, s, e]
                end

                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
                end

                integral_load_auxiliary_state!(
                    balance_law,
                    Vars{vars_integrals(balance_law, FT)}(view(
                        local_kernel,
                        :,
                        k,
                    )),
                    Vars{vars_state_conservative(balance_law, FT)}(
                        local_state_conservative,
                    ),
                    Vars{vars_state_auxiliary(balance_law, FT)}(
                        local_state_auxiliary,
                    ),
                )

                # multiply in the curve jacobian
                @unroll for s in 1:nout
                    local_kernel[s, k] *= Jc
                end
            end

            # Evaluate the integral up the element
            @unroll for s in 1:nout
                @unroll for k in 1:Nq
                    @unroll for n in 1:Nq
                        local_integral[s, k] += s_I[k, n] * local_kernel[s, n]
                    end
                end
            end

            # Store out to memory and reset the background value for next element
            @unroll for k in 1:Nq
                @unroll for s in 1:nout
                    local_kernel[s, k] = local_integral[s, k]
                end
                ijk = i + Nq * ((j - 1) + Nqj * (k - 1))
                integral_set_auxiliary_state!(
                    balance_law,
                    Vars{vars_state_auxiliary(balance_law, FT)}(view(
                        state_auxiliary,
                        ijk,
                        :,
                        e,
                    )),
                    Vars{vars_integrals(balance_law, FT)}(view(
                        local_kernel,
                        :,
                        k,
                    )),
                )
                @unroll for ind_out in 1:nout
                    local_integral[ind_out, k] = local_integral[ind_out, Nq]
                end
            end
        end
    end
end

@kernel function kernel_reverse_indefinite_stack_integral!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    ::Val{nvertelem},
    state,
    state_auxiliary,
    elems,
) where {dim, N, nvertelem}
    @uniform begin
        FT = eltype(state_auxiliary)

        Nq = N + 1
        Nqj = dim == 2 ? 1 : Nq
        nout = num_reverse_integrals(balance_law, FT)

        # note that k is the second not 4th index (since this is scratch memory and k
        # needs to be persistent across threads)
        l_T = MArray{Tuple{nout}, FT}(undef)
        l_V = MArray{Tuple{nout}, FT}(undef)
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]

        # Initialize the constant state at zero
        ijk = i + Nq * ((j - 1) + Nqj * (Nq - 1))
        et = nvertelem + (eh - 1) * nvertelem
        reverse_integral_load_auxiliary_state!(
            balance_law,
            Vars{vars_reverse_integrals(balance_law, FT)}(l_T),
            Vars{vars_state_conservative(balance_law, FT)}(view(
                state,
                ijk,
                :,
                et,
            )),
            Vars{vars_state_auxiliary(balance_law, FT)}(view(
                state_auxiliary,
                ijk,
                :,
                et,
            )),
        )

        # Loop up the stack of elements
        for ev in 1:nvertelem
            e = ev + (eh - 1) * nvertelem
            @unroll for k in 1:Nq
                ijk = i + Nq * ((j - 1) + Nqj * (k - 1))
                reverse_integral_load_auxiliary_state!(
                    balance_law,
                    Vars{vars_reverse_integrals(balance_law, FT)}(l_V),
                    Vars{vars_state_conservative(balance_law, FT)}(view(
                        state,
                        ijk,
                        :,
                        e,
                    )),
                    Vars{vars_state_auxiliary(balance_law, FT)}(view(
                        state_auxiliary,
                        ijk,
                        :,
                        e,
                    )),
                )
                l_V .= l_T .- l_V
                reverse_integral_set_auxiliary_state!(
                    balance_law,
                    Vars{vars_state_auxiliary(balance_law, FT)}(view(
                        state_auxiliary,
                        ijk,
                        :,
                        e,
                    )),
                    Vars{vars_reverse_integrals(balance_law, FT)}(l_V),
                )
            end
        end
    end
end

# TODO: Generalize to more than one field?
@kernel function kernel_copy_stack_field_down!(
    ::Val{dim},
    ::Val{N},
    ::Val{nvertelem},
    state_auxiliary,
    elems,
    ::Val{fldin},
    ::Val{fldout},
) where {dim, N, nvertelem, fldin, fldout}
    DFloat = eltype(state_auxiliary)

    Nq = N + 1
    Nqj = dim == 2 ? 1 : Nq

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    # note that k is the second not 4th index (since this is scratch memory and k
    # needs to be persistent across threads)
    @inbounds begin
        # Initialize the constant state at zero
        ijk = i + Nq * ((j - 1) + Nqj * (Nq - 1))
        eh = elems[_eh]
        et = nvertelem + (eh - 1) * nvertelem
        val = state_auxiliary[ijk, fldin, et]

        # Loop up the stack of elements
        for ev in 1:nvertelem
            e = ev + (eh - 1) * nvertelem
            @unroll for k in 1:Nq
                ijk = i + Nq * ((j - 1) + Nqj * (k - 1))
                state_auxiliary[ijk, fldout, e] = val
            end
        end
    end
end

@kernel function volume_divergence_of_gradients!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    Qhypervisc_grad,
    Qhypervisc_div,
    vgeo,
    D,
    elems,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(Qhypervisc_grad)
        ngradlapstate = num_gradient_laplacian(balance_law, FT)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq

        l_div = MArray{Tuple{ngradlapstate}, FT}(undef)
    end

    s_grad = @localmem FT (Nq, Nq, Nqk, ngradlapstate, 3)
    s_D = @localmem FT (Nq, Nq)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)
    ijk = @index(Local, Linear)

    @inbounds begin
        s_D[i, j] = D[i, j]

        @unroll for s in 1:ngradlapstate
            s_grad[i, j, k, s, 1] = Qhypervisc_grad[ijk, 3 * (s - 1) + 1, e]
            s_grad[i, j, k, s, 2] = Qhypervisc_grad[ijk, 3 * (s - 1) + 2, e]
            s_grad[i, j, k, s, 3] = Qhypervisc_grad[ijk, 3 * (s - 1) + 3, e]
        end
        @synchronize

        ξ1x1, ξ1x2, ξ1x3 =
            vgeo[ijk, _ξ1x1, e], vgeo[ijk, _ξ1x2, e], vgeo[ijk, _ξ1x3, e]
        if dim == 3 || (dim == 2 && direction isa EveryDirection)
            ξ2x1, ξ2x2, ξ2x3 =
                vgeo[ijk, _ξ2x1, e], vgeo[ijk, _ξ2x2, e], vgeo[ijk, _ξ2x3, e]
        end
        if dim == 3 && direction isa EveryDirection
            ξ3x1, ξ3x2, ξ3x3 =
                vgeo[ijk, _ξ3x1, e], vgeo[ijk, _ξ3x2, e], vgeo[ijk, _ξ3x3, e]
        end

        @unroll for s in 1:ngradlapstate
            g1ξ1 = g1ξ2 = g1ξ3 = zero(FT)
            g2ξ1 = g2ξ2 = g2ξ3 = zero(FT)
            g3ξ1 = g3ξ2 = g3ξ3 = zero(FT)
            @unroll for n in 1:Nq
                Din = s_D[i, n]
                g1ξ1 += Din * s_grad[n, j, k, s, 1]
                g2ξ1 += Din * s_grad[n, j, k, s, 2]
                g3ξ1 += Din * s_grad[n, j, k, s, 3]
                if dim == 3 || (dim == 2 && direction isa EveryDirection)
                    Djn = s_D[j, n]
                    g1ξ2 += Djn * s_grad[i, n, k, s, 1]
                    g2ξ2 += Djn * s_grad[i, n, k, s, 2]
                    g3ξ2 += Djn * s_grad[i, n, k, s, 3]
                end
                if dim == 3 && direction isa EveryDirection
                    Dkn = s_D[k, n]
                    g1ξ3 += Dkn * s_grad[i, j, n, s, 1]
                    g2ξ3 += Dkn * s_grad[i, j, n, s, 2]
                    g3ξ3 += Dkn * s_grad[i, j, n, s, 3]
                end
            end
            l_div[s] = ξ1x1 * g1ξ1 + ξ1x2 * g2ξ1 + ξ1x3 * g3ξ1

            if dim == 3 || (dim == 2 && direction isa EveryDirection)
                l_div[s] += ξ2x1 * g1ξ2 + ξ2x2 * g2ξ2 + ξ2x3 * g3ξ2
            end

            if dim == 3 && direction isa EveryDirection
                l_div[s] += ξ3x1 * g1ξ3 + ξ3x2 * g2ξ3 + ξ3x3 * g3ξ3
            end
        end

        @unroll for s in 1:ngradlapstate
            Qhypervisc_div[ijk, s, e] = l_div[s]
        end
        @synchronize
    end
end

@kernel function volume_divergence_of_gradients!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    ::VerticalDirection,
    Qhypervisc_grad,
    Qhypervisc_div,
    vgeo,
    D,
    elems,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(Qhypervisc_grad)
        ngradlapstate = num_gradient_laplacian(balance_law, FT)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq

        l_div = MArray{Tuple{ngradlapstate}, FT}(undef)
    end

    s_grad = @localmem FT (Nq, Nq, Nqk, ngradlapstate, 3)
    s_D = @localmem FT (Nq, Nq)

    e = @index(Group, Linear)
    ijk = @index(Local, Linear)
    i, j, k = @index(Local, NTuple)
    @inbounds begin
        s_D[i, j] = D[i, j]

        @unroll for s in 1:ngradlapstate
            s_grad[i, j, k, s, 1] = Qhypervisc_grad[ijk, 3 * (s - 1) + 1, e]
            s_grad[i, j, k, s, 2] = Qhypervisc_grad[ijk, 3 * (s - 1) + 2, e]
            s_grad[i, j, k, s, 3] = Qhypervisc_grad[ijk, 3 * (s - 1) + 3, e]
        end
        @synchronize

        if dim == 2
            ξ2x1, ξ2x2, ξ2x3 =
                vgeo[ijk, _ξ2x1, e], vgeo[ijk, _ξ2x2, e], vgeo[ijk, _ξ2x3, e]
        else
            ξ3x1, ξ3x2, ξ3x3 =
                vgeo[ijk, _ξ3x1, e], vgeo[ijk, _ξ3x2, e], vgeo[ijk, _ξ3x3, e]
        end

        @unroll for s in 1:ngradlapstate
            g1ξv = g2ξv = g3ξv = zero(FT)
            @unroll for n in 1:Nq
                if dim == 2
                    Djn = s_D[j, n]
                    g1ξv += Djn * s_grad[i, n, k, s, 1]
                    g2ξv += Djn * s_grad[i, n, k, s, 2]
                    g3ξv += Djn * s_grad[i, n, k, s, 3]
                else
                    Dkn = s_D[k, n]
                    g1ξv += Dkn * s_grad[i, j, n, s, 1]
                    g2ξv += Dkn * s_grad[i, j, n, s, 2]
                    g3ξv += Dkn * s_grad[i, j, n, s, 3]
                end
            end

            if dim == 2
                l_div[s] = ξ2x1 * g1ξv + ξ2x2 * g2ξv + ξ2x3 * g3ξv
            else
                l_div[s] = ξ3x1 * g1ξv + ξ3x2 * g2ξv + ξ3x3 * g3ξv
            end
        end

        @unroll for s in 1:ngradlapstate
            Qhypervisc_div[ijk, s, e] = l_div[s]
        end
        @synchronize
    end
end

@kernel function interface_divergence_of_gradients!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    divgradnumpenalty,
    Qhypervisc_grad,
    Qhypervisc_div,
    vgeo,
    sgeo,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(Qhypervisc_grad)
        ngradlapstate = num_gradient_laplacian(balance_law, FT)

        if dim == 1
            Np = (N + 1)
            Nfp = 1
            nface = 2
        elseif dim == 2
            Np = (N + 1) * (N + 1)
            Nfp = (N + 1)
            nface = 4
        elseif dim == 3
            Np = (N + 1) * (N + 1) * (N + 1)
            Nfp = (N + 1) * (N + 1)
            nface = 6
        end

        faces = 1:nface
        if direction isa VerticalDirection
            faces = (nface - 1):nface
        elseif direction isa HorizontalDirection
            faces = 1:(nface - 2)
        end

        Nqk = dim == 2 ? 1 : N + 1

        l_grad⁻ = MArray{Tuple{3, ngradlapstate}, FT}(undef)
        l_grad⁺ = MArray{Tuple{3, ngradlapstate}, FT}(undef)
        l_div = MArray{Tuple{ngradlapstate}, FT}(undef)
    end

    eI = @index(Group, Linear)
    n = @index(Local, Linear)

    e = @private Int (1,)
    @inbounds e[1] = elems[eI]

    @inbounds for f in faces
        e⁻ = e[1]
        normal_vector = SVector(
            sgeo[_n1, n, f, e⁻],
            sgeo[_n2, n, f, e⁻],
            sgeo[_n3, n, f, e⁻],
        )
        sM, vMI = sgeo[_sM, n, f, e⁻], sgeo[_vMI, n, f, e⁻]
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1

        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1

        # Load minus side data
        @unroll for s in 1:ngradlapstate
            l_grad⁻[1, s] = Qhypervisc_grad[vid⁻, 3 * (s - 1) + 1, e⁻]
            l_grad⁻[2, s] = Qhypervisc_grad[vid⁻, 3 * (s - 1) + 2, e⁻]
            l_grad⁻[3, s] = Qhypervisc_grad[vid⁻, 3 * (s - 1) + 3, e⁻]
        end

        # Load plus side data
        @unroll for s in 1:ngradlapstate
            l_grad⁺[1, s] = Qhypervisc_grad[vid⁺, 3 * (s - 1) + 1, e⁺]
            l_grad⁺[2, s] = Qhypervisc_grad[vid⁺, 3 * (s - 1) + 2, e⁺]
            l_grad⁺[3, s] = Qhypervisc_grad[vid⁺, 3 * (s - 1) + 3, e⁺]
        end

        bctype = elemtobndy[f, e⁻]
        if bctype == 0
            numerical_flux_divergence!(
                divgradnumpenalty,
                balance_law,
                Vars{vars_gradient_laplacian(balance_law, FT)}(l_div),
                normal_vector,
                Grad{vars_gradient_laplacian(balance_law, FT)}(l_grad⁻),
                Grad{vars_gradient_laplacian(balance_law, FT)}(l_grad⁺),
            )
        else
            numerical_boundary_flux_divergence!(
                divgradnumpenalty,
                balance_law,
                Vars{vars_gradient_laplacian(balance_law, FT)}(l_div),
                normal_vector,
                Grad{vars_gradient_laplacian(balance_law, FT)}(l_grad⁻),
                Grad{vars_gradient_laplacian(balance_law, FT)}(l_grad⁺),
                bctype,
            )
        end

        @unroll for s in 1:ngradlapstate
            Qhypervisc_div[vid⁻, s, e⁻] += vMI * sM * l_div[s]
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

@kernel function volume_gradients_of_laplacians!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    Qhypervisc_grad,
    Qhypervisc_div,
    state_conservative,
    state_auxiliary,
    vgeo,
    ω,
    D,
    elems,
    t,
) where {dim, polyorder}
    @uniform begin
        N = polyorder

        FT = eltype(Qhypervisc_grad)
        num_state_conservative = number_state_conservative(balance_law, FT)
        ngradlapstate = num_gradient_laplacian(balance_law, FT)
        nhyperviscstate = num_hyperdiffusive(balance_law, FT)
        num_state_auxiliary = number_state_auxiliary(balance_law, FT)
        ngradtransformstate = num_state_conservative

        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq

        l_grad_lap = MArray{Tuple{3, ngradlapstate}, FT}(undef)
        local_state_hyperdiffusion = MArray{Tuple{nhyperviscstate}, FT}(undef)
    end

    s_lap = @localmem FT (Nq, Nq, Nqk, ngradlapstate)
    s_D = @localmem FT (Nq, Nq)
    s_ω = @localmem FT (Nq,)
    local_state_conservative = @private FT (ngradtransformstate,)
    local_state_auxiliary = @private FT (num_state_auxiliary,)

    e = @index(Group, Linear)
    ijk = @index(Local, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds @views begin
        s_ω[j] = ω[j]
        s_D[i, j] = D[i, j]

        @unroll for s in 1:ngradtransformstate
            local_state_conservative[s] = state_conservative[ijk, s, e]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
        end

        @unroll for s in 1:ngradlapstate
            s_lap[i, j, k, s] = Qhypervisc_div[ijk, s, e]
        end
        @synchronize

        ξ1x1, ξ1x2, ξ1x3 =
            vgeo[ijk, _ξ1x1, e], vgeo[ijk, _ξ1x2, e], vgeo[ijk, _ξ1x3, e]
        if dim == 3 || (dim == 2 && direction isa EveryDirection)
            ξ2x1, ξ2x2, ξ2x3 =
                vgeo[ijk, _ξ2x1, e], vgeo[ijk, _ξ2x2, e], vgeo[ijk, _ξ2x3, e]
        end
        if dim == 3 && direction isa EveryDirection
            ξ3x1, ξ3x2, ξ3x3 =
                vgeo[ijk, _ξ3x1, e], vgeo[ijk, _ξ3x2, e], vgeo[ijk, _ξ3x3, e]
        end
        @unroll for s in 1:ngradlapstate
            lap_ξ1 = lap_ξ2 = lap_ξ3 = zero(FT)
            @unroll for n in 1:Nq
                njk = n + Nq * ((j - 1) + Nq * (k - 1))
                Dni = s_D[n, i] * s_ω[n] / s_ω[i]
                lap_njk = s_lap[n, j, k, s]
                lap_ξ1 += Dni * lap_njk
                if dim == 3 || (dim == 2 && direction isa EveryDirection)
                    ink = i + Nq * ((n - 1) + Nq * (k - 1))
                    Dnj = s_D[n, j] * s_ω[n] / s_ω[j]
                    lap_ink = s_lap[i, n, k, s]
                    lap_ξ2 += Dnj * lap_ink
                end
                if dim == 3 && direction isa EveryDirection
                    ijn = i + Nq * ((j - 1) + Nq * (n - 1))
                    Dnk = s_D[n, k] * s_ω[n] / s_ω[k]
                    lap_ijn = s_lap[i, j, n, s]
                    lap_ξ3 += Dnk * lap_ijn
                end
            end

            l_grad_lap[1, s] = -ξ1x1 * lap_ξ1
            l_grad_lap[2, s] = -ξ1x2 * lap_ξ1
            l_grad_lap[3, s] = -ξ1x3 * lap_ξ1

            if dim == 3 || (dim == 2 && direction isa EveryDirection)
                l_grad_lap[1, s] -= ξ2x1 * lap_ξ2
                l_grad_lap[2, s] -= ξ2x2 * lap_ξ2
                l_grad_lap[3, s] -= ξ2x3 * lap_ξ2
            end

            if dim == 3 && direction isa EveryDirection
                l_grad_lap[1, s] -= ξ3x1 * lap_ξ3
                l_grad_lap[2, s] -= ξ3x2 * lap_ξ3
                l_grad_lap[3, s] -= ξ3x3 * lap_ξ3
            end
        end

        fill!(
            local_state_hyperdiffusion,
            -zero(eltype(local_state_hyperdiffusion)),
        )
        transform_post_gradient_laplacian!(
            balance_law,
            Vars{vars_hyperdiffusive(balance_law, FT)}(
                local_state_hyperdiffusion,
            ),
            Grad{vars_gradient_laplacian(balance_law, FT)}(l_grad_lap),
            Vars{vars_state_conservative(balance_law, FT)}(local_state_conservative[:]),
            Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary[:]),
            t,
        )
        @unroll for s in 1:nhyperviscstate
            Qhypervisc_grad[ijk, s, e] = local_state_hyperdiffusion[s]
        end
        @synchronize
    end
end

@kernel function volume_gradients_of_laplacians!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    ::VerticalDirection,
    Qhypervisc_grad,
    Qhypervisc_div,
    state_conservative,
    state_auxiliary,
    vgeo,
    ω,
    D,
    elems,
    t,
) where {dim, polyorder}
    @uniform begin
        N = polyorder

        FT = eltype(Qhypervisc_grad)
        num_state_conservative = number_state_conservative(balance_law, FT)
        ngradlapstate = num_gradient_laplacian(balance_law, FT)
        nhyperviscstate = num_hyperdiffusive(balance_law, FT)
        num_state_auxiliary = number_state_auxiliary(balance_law, FT)
        ngradtransformstate = num_state_conservative

        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq

        l_grad_lap = MArray{Tuple{3, ngradlapstate}, FT}(undef)
        local_state_hyperdiffusion = MArray{Tuple{nhyperviscstate}, FT}(undef)
    end

    s_lap = @localmem FT (Nq, Nq, Nqk, ngradlapstate)
    s_D = @localmem FT (Nq, Nq)
    s_ω = @localmem FT (Nq,)
    local_state_conservative = @private FT (ngradtransformstate,)
    local_state_auxiliary = @private FT (num_state_auxiliary,)

    e = @index(Group, Linear)
    ijk = @index(Local, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds @views begin
        s_ω[j] = ω[j]
        s_D[i, j] = D[i, j]

        @unroll for s in 1:ngradtransformstate
            local_state_conservative[s] = state_conservative[ijk, s, e]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
        end

        @unroll for s in 1:ngradlapstate
            s_lap[i, j, k, s] = Qhypervisc_div[ijk, s, e]
        end
        @synchronize

        if dim == 2
            ξvx1, ξvx2, ξvx3 =
                vgeo[ijk, _ξ2x1, e], vgeo[ijk, _ξ2x2, e], vgeo[ijk, _ξ2x3, e]
        else
            ξvx1, ξvx2, ξvx3 =
                vgeo[ijk, _ξ3x1, e], vgeo[ijk, _ξ3x2, e], vgeo[ijk, _ξ3x3, e]
        end
        @unroll for s in 1:ngradlapstate
            lap_ξv = zero(FT)
            @unroll for n in 1:Nq
                if dim == 2
                    ink = i + Nq * ((n - 1) + Nq * (k - 1))
                    Dnj = s_D[n, j] * s_ω[n] / s_ω[j]
                    lap_ink = s_lap[i, n, k, s]
                    lap_ξv += Dnj * lap_ink
                else
                    ijn = i + Nq * ((j - 1) + Nq * (n - 1))
                    Dnk = s_D[n, k] * s_ω[n] / s_ω[k]
                    lap_ijn = s_lap[i, j, n, s]
                    lap_ξv += Dnk * lap_ijn
                end
            end

            l_grad_lap[1, s] = -ξvx1 * lap_ξv
            l_grad_lap[2, s] = -ξvx2 * lap_ξv
            l_grad_lap[3, s] = -ξvx3 * lap_ξv
        end

        fill!(
            local_state_hyperdiffusion,
            -zero(eltype(local_state_hyperdiffusion)),
        )
        transform_post_gradient_laplacian!(
            balance_law,
            Vars{vars_hyperdiffusive(balance_law, FT)}(
                local_state_hyperdiffusion,
            ),
            Grad{vars_gradient_laplacian(balance_law, FT)}(l_grad_lap),
            Vars{vars_state_conservative(balance_law, FT)}(local_state_conservative[:]),
            Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary[:]),
            t,
        )
        @unroll for s in 1:nhyperviscstate
            Qhypervisc_grad[ijk, s, e] = local_state_hyperdiffusion[s]
        end
        @synchronize
    end
end

@kernel function interface_gradients_of_laplacians!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    hyperviscnumflux,
    Qhypervisc_grad,
    Qhypervisc_div,
    state_conservative,
    state_auxiliary,
    vgeo,
    sgeo,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
    t,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(Qhypervisc_grad)
        num_state_conservative = number_state_conservative(balance_law, FT)
        ngradlapstate = num_gradient_laplacian(balance_law, FT)
        nhyperviscstate = num_hyperdiffusive(balance_law, FT)
        num_state_auxiliary = number_state_auxiliary(balance_law, FT)
        ngradtransformstate = num_state_conservative

        if dim == 1
            Np = (N + 1)
            Nfp = 1
            nface = 2
        elseif dim == 2
            Np = (N + 1) * (N + 1)
            Nfp = (N + 1)
            nface = 4
        elseif dim == 3
            Np = (N + 1) * (N + 1) * (N + 1)
            Nfp = (N + 1) * (N + 1)
            nface = 6
        end

        faces = 1:nface
        if direction isa VerticalDirection
            faces = (nface - 1):nface
        elseif direction isa HorizontalDirection
            faces = 1:(nface - 2)
        end

        Nqk = dim == 2 ? 1 : N + 1

        l_lap⁻ = MArray{Tuple{ngradlapstate}, FT}(undef)
        l_lap⁺ = MArray{Tuple{ngradlapstate}, FT}(undef)
        local_state_hyperdiffusion = MArray{Tuple{nhyperviscstate}, FT}(undef)

        local_state_conservative⁻ =
            MArray{Tuple{ngradtransformstate}, FT}(undef)
        local_state_auxiliary⁻ = MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_state_conservative⁺ =
            MArray{Tuple{ngradtransformstate}, FT}(undef)
        local_state_auxiliary⁺ = MArray{Tuple{num_state_auxiliary}, FT}(undef)
    end

    eI = @index(Group, Linear)
    n = @index(Local, Linear)

    e = @private Int (1,)
    @inbounds e[1] = elems[eI]

    @inbounds for f in faces
        e⁻ = e[1]
        normal_vector = SVector(
            sgeo[_n1, n, f, e⁻],
            sgeo[_n2, n, f, e⁻],
            sgeo[_n3, n, f, e⁻],
        )
        sM, vMI = sgeo[_sM, n, f, e⁻], sgeo[_vMI, n, f, e⁻]
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1

        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1

        # Load minus side data
        @unroll for s in 1:ngradtransformstate
            local_state_conservative⁻[s] = state_conservative[vid⁻, s, e⁻]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁻[s] = state_auxiliary[vid⁻, s, e⁻]
        end

        @unroll for s in 1:ngradlapstate
            l_lap⁻[s] = Qhypervisc_div[vid⁻, s, e⁻]
        end

        # Load plus side data
        @unroll for s in 1:ngradtransformstate
            local_state_conservative⁺[s] = state_conservative[vid⁺, s, e⁺]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁺[s] = state_auxiliary[vid⁺, s, e⁺]
        end

        @unroll for s in 1:ngradlapstate
            l_lap⁺[s] = Qhypervisc_div[vid⁺, s, e⁺]
        end

        bctype = elemtobndy[f, e⁻]
        if bctype == 0
            numerical_flux_higher_order!(
                hyperviscnumflux,
                balance_law,
                Vars{vars_hyperdiffusive(balance_law, FT)}(
                    local_state_hyperdiffusion,
                ),
                normal_vector,
                Vars{vars_gradient_laplacian(balance_law, FT)}(l_lap⁻),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁻,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_gradient_laplacian(balance_law, FT)}(l_lap⁺),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁺,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁺,
                ),
                t,
            )
        else
            numerical_boundary_flux_higher_order!(
                hyperviscnumflux,
                balance_law,
                Vars{vars_hyperdiffusive(balance_law, FT)}(
                    local_state_hyperdiffusion,
                ),
                normal_vector,
                Vars{vars_gradient_laplacian(balance_law, FT)}(l_lap⁻),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁻,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_gradient_laplacian(balance_law, FT)}(l_lap⁺),
                Vars{vars_state_conservative(balance_law, FT)}(
                    local_state_conservative⁺,
                ),
                Vars{vars_state_auxiliary(balance_law, FT)}(
                    local_state_auxiliary⁺,
                ),
                bctype,
                t,
            )
        end

        @unroll for s in 1:nhyperviscstate
            Qhypervisc_grad[vid⁻, s, e⁻] +=
                vMI * sM * local_state_hyperdiffusion[s]
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

@kernel function kernel_local_courant!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    pointwise_courant,
    local_courant,
    state_conservative,
    state_auxiliary,
    state_gradient_flux,
    elems,
    Δt,
    simtime,
    direction,
) where {dim, N}
    @uniform begin
        FT = eltype(state_conservative)
        num_state_conservative = number_state_conservative(balance_law, FT)
        num_state_gradient_flux = number_state_gradient_flux(balance_law, FT)
        num_state_auxiliary = number_state_auxiliary(balance_law, FT)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq

        Np = Nq * Nq * Nqk

        local_state_conservative =
            MArray{Tuple{num_state_conservative}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
    end

    I = @index(Global, Linear)
    e = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        @unroll for s in 1:num_state_conservative
            local_state_conservative[s] = state_conservative[n, s, e]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end

        @unroll for s in 1:num_state_gradient_flux
            local_state_gradient_flux[s] = state_gradient_flux[n, s, e]
        end

        Δx = pointwise_courant[n, e]
        c = local_courant(
            balance_law,
            Vars{vars_state_conservative(balance_law, FT)}(
                local_state_conservative,
            ),
            Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary),
            Vars{vars_state_gradient_flux(balance_law, FT)}(
                local_state_gradient_flux,
            ),
            Δx,
            Δt,
            simtime,
            direction,
        )

        pointwise_courant[n, e] = c
    end
end
