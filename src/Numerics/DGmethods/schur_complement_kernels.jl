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

@kernel function kernel_init_schur_state!(
    schur_complement::SchurComplement,
    balance_law::BalanceLaw,
    schur_state,
    schur_state_auxiliary,
    state_rhs,
    state_auxiliary,
    vgeo,
    ::Val{dim},
    ::Val{polyorder}
) where {dim, polyorder}
    N = polyorder
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Np = Nq * Nq * Nqk
    FT = eltype(schur_state)
    schur_num_state = schur_number_state(schur_complement, FT)
    schur_num_state_auxiliary = schur_number_state_auxiliary(schur_complement, FT)
    num_state_rhs = number_state_conservative(balance_law, FT)
    num_state_auxiliary = number_state_auxiliary(balance_law, FT)

    local_schur_state = MArray{Tuple{1,}, FT}(undef)
    local_schur_state_auxiliary = MArray{Tuple{schur_num_state_auxiliary}, FT}(undef)
    local_state_rhs = MArray{Tuple{num_state_rhs}, FT}(undef)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

    I = @index(Global, Linear)
    e = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        local_schur_state[1] = schur_state[n, 1, e]
        
        @unroll for s in 1:schur_num_state_auxiliary
            local_schur_state_auxiliary[s] = schur_state_auxiliary[n, s, e]
        end

        @unroll for s in 1:num_state_rhs
            local_state_rhs[s] = state_rhs[n, s, e]
        end
        
        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end

        schur_init_state!(
            schur_complement,
            balance_law,
            Vars{schur_vars_state(schur_complement, FT)}(local_schur_state),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary),
            Vars{vars_state_conservative(balance_law, FT)}(local_state_rhs),
            Vars{vars_state_auxiliary(balance_law, FT)}(local_state_auxiliary),
        )

        schur_state[n, 1, e] = local_schur_state[1]
    end
end

@kernel function schur_init_auxiliary_state!(
    sc::SchurComplement,
    bl::BalanceLaw,
    schur_auxstate,
    auxstate,
    vgeo,
    elems,
    ::Val{dim},
    ::Val{polyorder}
) where {dim, polyorder}
    N = polyorder
    FT = eltype(schur_auxstate)
    nauxstate = number_state_auxiliary(bl, FT)
    nschurauxstate = schur_number_state_auxiliary(sc, FT)

    l_aux = MArray{Tuple{nauxstate}, FT}(undef)
    l_schur_aux = MArray{Tuple{nschurauxstate}, FT}(undef)

    e = @index(Group, Linear)
    n = @index(Local, Linear)

    @inbounds begin
        @unroll for s in 1:nauxstate
            l_aux[s] = auxstate[n, s, e]
        end

        schur_init_aux!(
            sc,
            bl,
            Vars{schur_vars_state_auxiliary(sc, FT)}(l_schur_aux),
            Vars{vars_state_auxiliary(bl, FT)}(l_aux),
            LocalGeometry(Val(polyorder), vgeo, n, e),
        )

        @unroll for s in 1:nschurauxstate
            schur_auxstate[n, s, e] = l_schur_aux[s]
        end
    end
end

@kernel function schur_auxiliary_gradients!(
    schur_complement::SchurComplement,
    balance_law::BalanceLaw,
    schur_state_auxiliary,
    vgeo,
    D,
    schur_indexmap,
    direction,
    ::Val{dim},
    ::Val{polyorder},
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(schur_state_auxiliary)
        schur_num_state_auxiliary = schur_number_state_auxiliary(schur_complement, FT)
        schur_num_gradient_auxiliary = schur_number_gradient_auxiliary(schur_complement, FT)
        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq
    end
    
    s_D = @localmem FT (Nq, Nq)
    shared_auxiliary = @localmem FT (Nq, Nq, Nqk)
    
    e = @index(Group, Linear)
    ijk = @index(Local, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds @views begin
      s_D[i, j] = D[i, j]

      for s in 1:schur_num_gradient_auxiliary
        shared_auxiliary[i, j, k] = schur_state_auxiliary[ijk, schur_indexmap[s], e]
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

        Gξ1 = Gξ2 = Gξ3 = zero(FT)
        @unroll for n in 1:Nq
            Gξ1 += s_D[i, n] * shared_auxiliary[n, j, k]
            if dim == 3 || (dim == 2 && direction isa EveryDirection)
                Gξ2 += s_D[j, n] * shared_auxiliary[i, n, k]
            end
            if dim == 3 && direction isa EveryDirection
                Gξ3 += s_D[k, n] * shared_auxiliary[i, j, n]
            end
        end
        gradient_auxiliary_1 = ξ1x1 * Gξ1
        gradient_auxiliary_2 = ξ1x2 * Gξ1
        gradient_auxiliary_3 = ξ1x3 * Gξ1

        if dim == 3 || (dim == 2 && direction isa EveryDirection)
          gradient_auxiliary_1 += ξ2x1 * Gξ2
          gradient_auxiliary_2 += ξ2x2 * Gξ2
          gradient_auxiliary_3 += ξ2x3 * Gξ2
        end

        if dim == 3 && direction isa EveryDirection
          gradient_auxiliary_1 += ξ3x1 * Gξ3
          gradient_auxiliary_2 += ξ3x2 * Gξ3
          gradient_auxiliary_3 += ξ3x3 * Gξ3
        end

        base_index = schur_num_state_auxiliary - 3schur_num_gradient_auxiliary + 3(s - 1) 
        schur_state_auxiliary[ijk, base_index + 1, e] = gradient_auxiliary_1
        schur_state_auxiliary[ijk, base_index + 2, e] = gradient_auxiliary_2
        schur_state_auxiliary[ijk, base_index + 3, e] = gradient_auxiliary_3
        @synchronize
      end
    end
end

@kernel function schur_volume_gradients!(
    schur_complement::SchurComplement,
    balance_law::BalanceLaw,
    schur_state_gradient,
    schur_state,
    schur_state_auxiliary,
    vgeo,
    D,
    direction,
    ::Val{dim},
    ::Val{polyorder},
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(schur_state_auxiliary)
        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq
    end
    
    s_D = @localmem FT (Nq, Nq)
    shared_state = @localmem FT (Nq, Nq, Nqk)
    
    e = @index(Group, Linear)
    ijk = @index(Local, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds @views begin
      s_D[i, j] = D[i, j]

      shared_state[i, j, k] = schur_state[ijk, 1, e]
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

      Gξ1 = Gξ2 = Gξ3 = zero(FT)
      @unroll for n in 1:Nq
          Gξ1 += s_D[i, n] * shared_state[n, j, k]
          if dim == 3 || (dim == 2 && direction isa EveryDirection)
              Gξ2 += s_D[j, n] * shared_state[i, n, k]
          end
          if dim == 3 && direction isa EveryDirection
              Gξ3 += s_D[k, n] * shared_state[i, j, n]
          end
      end
      state_gradient_x = ξ1x1 * Gξ1
      state_gradient_y = ξ1x2 * Gξ1
      state_gradient_z = ξ1x3 * Gξ1

      if dim == 3 || (dim == 2 && direction isa EveryDirection)
        state_gradient_x += ξ2x1 * Gξ2
        state_gradient_y += ξ2x2 * Gξ2
        state_gradient_z += ξ2x3 * Gξ2
      end

      if dim == 3 && direction isa EveryDirection
        state_gradient_x += ξ3x1 * Gξ3
        state_gradient_y += ξ3x2 * Gξ3
        state_gradient_z += ξ3x3 * Gξ3
      end

      schur_state_gradient[ijk, 1, e] = state_gradient_x
      schur_state_gradient[ijk, 2, e] = state_gradient_y
      schur_state_gradient[ijk, 3, e] = state_gradient_z
    end
end

@kernel function schur_interface_gradients!(
    schur_complement::SchurComplement,
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    schur_state_gradient,
    schur_state,
    vgeo,
    sgeo,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(schur_state)

        schur_num_state_gradient = schur_number_state_gradient(schur_complement, FT)

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

        local_penalty = MArray{Tuple{3}, FT}(undef)
        local_schur_state⁻ = MArray{Tuple{1}, FT}(undef)
        local_schur_state⁺ = MArray{Tuple{1}, FT}(undef)
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
        local_schur_state⁻[1] = schur_state[vid⁻, 1, e⁻]
        # Load plus side data
        local_schur_state⁺[1] = schur_state[vid⁺, 1, e⁺]

        bctype = elemtobndy[f, e⁻]
        if bctype != 0
          schur_gradient_boundary_state!(
            schur_complement,
            balance_law,
            Vars{schur_vars_state(schur_complement, FT)}(local_schur_state⁺),
            normal_vector,
            Vars{schur_vars_state(schur_complement, FT)}(local_schur_state⁻),
            bctype
          )
        end
        @unroll for d in 1:3
          local_penalty[d] = normal_vector[d] *
            (local_schur_state⁺[1] .- local_schur_state⁻[1]) / 2
        end

        @unroll for d in 1:3
          schur_state_gradient[vid⁻, d, e⁻] += vMI * sM * local_penalty[d]
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

@kernel function schur_volume_lhs!(
    schur_complement::SchurComplement,
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    schur_lhs,
    schur_state,
    schur_state_auxiliary,
    schur_state_gradient,
    vgeo,
    D,
    α
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(schur_state)
        
        schur_num_state_auxiliary = schur_number_state_auxiliary(schur_complement, FT)
        schur_num_state_gradient = schur_number_state_gradient(schur_complement, FT)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq

        local_source = MArray{Tuple{1,}, FT}(undef)
        local_schur_state =
            MArray{Tuple{1,}, FT}(undef)
        local_schur_state_gradient =
            MArray{Tuple{schur_num_state_gradient}, FT}(undef)
        local_schur_state_auxiliary = MArray{Tuple{schur_num_state_auxiliary}, FT}(undef)
        local_flux = MArray{Tuple{3, 1}, FT}(undef)
    end

    shared_flux = @localmem FT (3, Nq, Nq, Nqk)
    s_D = @localmem FT (Nq, Nq)
    
    local_schur_lhs = @private FT (1,)

    e = @index(Group, Linear)
    ijk = @index(Local, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        s_D[i, j] = D[i, j]

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

        local_schur_lhs[1] = zero(FT)
        local_schur_state[1] = schur_state[ijk, 1, e]

        @unroll for s in 1:schur_num_state_auxiliary
            local_schur_state_auxiliary[s] = schur_state_auxiliary[ijk, s, e]
        end

        @unroll for s in 1:schur_num_state_gradient
            local_schur_state_gradient[s] = schur_state_gradient[ijk, s, e]
        end

        fill!(local_flux, -zero(eltype(local_flux)))
        schur_lhs_conservative!(
            schur_complement,
            balance_law,
            Grad{schur_vars_state(schur_complement, FT)}(local_flux),
            Vars{schur_vars_state(schur_complement, FT)}(
                local_schur_state,
            ),
            Vars{schur_vars_state_gradient(schur_complement, FT)}(local_schur_state_gradient),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary),
            α
        )

        shared_flux[1, i, j, k] = local_flux[1]
        shared_flux[2, i, j, k] = local_flux[2]
        shared_flux[3, i, j, k] = local_flux[3]

        # Build "inside metrics" flux
        F1, F2, F3 = shared_flux[1, i, j, k],
                     shared_flux[2, i, j, k],
                     shared_flux[3, i, j, k]

        shared_flux[1, i, j, k] = M * (ξ1x1 * F1 + ξ1x2 * F2 + ξ1x3 * F3)
        if dim == 3 || (dim == 2 && direction isa EveryDirection)
            shared_flux[2, i, j, k] =
                M * (ξ2x1 * F1 + ξ2x2 * F2 + ξ2x3 * F3)
        end
        if dim == 3 && direction isa EveryDirection
            shared_flux[3, i, j, k] =
                M * (ξ3x1 * F1 + ξ3x2 * F2 + ξ3x3 * F3)
        end

        fill!(local_source, -zero(eltype(local_source)))
        schur_lhs_nonconservative!(
            schur_complement,
            balance_law,
            Vars{schur_vars_state(schur_complement, FT)}(local_source),
            Vars{schur_vars_state(schur_complement, FT)}(
                local_schur_state,
            ),
            Vars{schur_vars_state_gradient(schur_complement, FT)}(
                local_schur_state_gradient,
            ),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary),
            α
        )

        local_schur_lhs[1] += local_source[1]
        @synchronize

        # Weak "inside metrics" derivative
        MI = vgeo[ijk, _MI, e]
        @unroll for n in 1:Nq
            # ξ1-grid lines
            local_schur_lhs[1] -= MI * s_D[n, i] * shared_flux[1, n, j, k]

            # ξ2-grid lines
            if dim == 3 || (dim == 2 && direction isa EveryDirection)
                local_schur_lhs[1] -=
                    MI * s_D[n, j] * shared_flux[2, i, n, k]
            end

            # ξ3-grid lines
            if dim == 3 && direction isa EveryDirection
                local_schur_lhs[1] -=
                    MI * s_D[n, k] * shared_flux[3, i, j, n]
            end
        end
        
        ijk = i + Nq * ((j - 1) + Nq * (k - 1))
        schur_lhs[ijk, 1, e] = local_schur_lhs[1]
    end
end

@kernel function schur_interface_lhs!(
    schur_complement::SchurComplement,
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    schur_lhs,
    schur_state,
    schur_state_auxiliary,
    schur_state_gradient,
    vgeo,
    sgeo,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
    α
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(schur_state)

        schur_num_state = schur_number_state(schur_complement, FT)
        schur_num_state_auxiliary = schur_number_state_auxiliary(schur_complement, FT)
        schur_num_state_gradient = schur_number_state_gradient(schur_complement, FT)

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

        local_schur_state⁻ =
            MArray{Tuple{1,}, FT}(undef)
        local_schur_state_gradient⁻ =
            MArray{Tuple{schur_num_state_gradient}, FT}(undef)
        local_schur_state_auxiliary⁻ = MArray{Tuple{schur_num_state_auxiliary}, FT}(undef)
        local_flux⁻ = MArray{Tuple{3, 1}, FT}(undef)
        
        local_schur_state⁺ =
            MArray{Tuple{1,}, FT}(undef)
        local_schur_state_gradient⁺ =
            MArray{Tuple{schur_num_state_gradient}, FT}(undef)
        local_schur_state_auxiliary⁺ = MArray{Tuple{schur_num_state_auxiliary}, FT}(undef)
        local_flux⁺ = MArray{Tuple{3, 1}, FT}(undef)
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
        local_schur_state⁻[1] = schur_state[vid⁻, 1, e⁻]

        @unroll for s in 1:schur_num_state_gradient
            local_schur_state_gradient⁻[s] = schur_state_gradient[vid⁻, s, e⁻]
        end

        @unroll for s in 1:schur_num_state_auxiliary
            local_schur_state_auxiliary⁻[s] = schur_state_auxiliary[vid⁻, s, e⁻]
        end
        
        # Load plus side data
        local_schur_state⁺[1] = schur_state[vid⁺, 1, e⁺]

        @unroll for s in 1:schur_num_state_gradient
            local_schur_state_gradient⁺[s] = schur_state_gradient[vid⁺, s, e⁺]
        end

        @unroll for s in 1:schur_num_state_auxiliary
            local_schur_state_auxiliary⁺[s] = schur_state_auxiliary[vid⁺, s, e⁺]
        end

        bctype = elemtobndy[f, e⁻]
        if bctype != 0
          schur_lhs_boundary_state!(
            schur_complement,
            balance_law,
            Vars{schur_vars_state(schur_complement, FT)}(
                local_schur_state⁺,
            ),
            Vars{schur_vars_state_gradient(schur_complement, FT)}(local_schur_state_gradient⁺),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁺),
            normal_vector,
            Vars{schur_vars_state(schur_complement, FT)}(
                local_schur_state⁻,
            ),
            Vars{schur_vars_state_gradient(schur_complement, FT)}(local_schur_state_gradient⁻),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁻),
            bctype
          )
        end

        fill!(local_flux⁻, -zero(eltype(local_flux⁻)))
        schur_lhs_conservative!(
            schur_complement,
            balance_law,
            Grad{schur_vars_state(schur_complement, FT)}(local_flux⁻),
            Vars{schur_vars_state(schur_complement, FT)}(
                local_schur_state⁻,
            ),
            Vars{schur_vars_state_gradient(schur_complement, FT)}(local_schur_state_gradient⁻),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁻),
            α
        )
        
        fill!(local_flux⁺, -zero(eltype(local_flux⁺)))
        schur_lhs_conservative!(
            schur_complement,
            balance_law,
            Grad{schur_vars_state(schur_complement, FT)}(local_flux⁺),
            Vars{schur_vars_state(schur_complement, FT)}(
                local_schur_state⁺,
            ),
            Vars{schur_vars_state_gradient(schur_complement, FT)}(local_schur_state_gradient⁺),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁺),
            α
        )
        local_schur_lhs = normal_vector' * SVector(local_flux⁻ + local_flux⁺) / 2

        #Update RHS
        schur_lhs[vid⁻, 1, e⁻] += vMI * sM * local_schur_lhs
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

@kernel function schur_volume_rhs!(
    schur_complement::SchurComplement,
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    schur_rhs,
    state_conservative,
    schur_state_auxiliary,
    vgeo,
    D,
    α
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(state_conservative)
        
        num_state_conservative = number_state_conservative(balance_law, FT)
        schur_num_state_auxiliary = schur_number_state_auxiliary(schur_complement, FT)

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq

        local_source = MArray{Tuple{1,}, FT}(undef)
        local_state_conservative =
            MArray{Tuple{num_state_conservative}, FT}(undef)
        local_schur_state_auxiliary = MArray{Tuple{schur_num_state_auxiliary}, FT}(undef)
        local_flux = MArray{Tuple{3, 1}, FT}(undef)
    end

    shared_flux = @localmem FT (3, Nq, Nq, Nqk)
    s_D = @localmem FT (Nq, Nq)
    
    local_schur_rhs = @private FT (1,)

    e = @index(Group, Linear)
    ijk = @index(Local, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        s_D[i, j] = D[i, j]

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

        local_schur_rhs[1] = zero(FT)
        
        @unroll for s in 1:num_state_conservative
          local_state_conservative[s] = state_conservative[ijk, s, e]
        end

        @unroll for s in 1:schur_num_state_auxiliary
            local_schur_state_auxiliary[s] = schur_state_auxiliary[ijk, s, e]
        end

        fill!(local_flux, -zero(eltype(local_flux)))
        schur_rhs_conservative!(
            schur_complement,
            balance_law,
            Grad{schur_vars_state(schur_complement, FT)}(local_flux),
            Vars{vars_state_conservative(balance_law, FT)}(
                local_state_conservative,
            ),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary),
            α
        )

        shared_flux[1, i, j, k] = local_flux[1]
        shared_flux[2, i, j, k] = local_flux[2]
        shared_flux[3, i, j, k] = local_flux[3]

        # Build "inside metrics" flux
        F1, F2, F3 = shared_flux[1, i, j, k],
                     shared_flux[2, i, j, k],
                     shared_flux[3, i, j, k]

        shared_flux[1, i, j, k] = M * (ξ1x1 * F1 + ξ1x2 * F2 + ξ1x3 * F3)
        if dim == 3 || (dim == 2 && direction isa EveryDirection)
            shared_flux[2, i, j, k] =
                M * (ξ2x1 * F1 + ξ2x2 * F2 + ξ2x3 * F3)
        end
        if dim == 3 && direction isa EveryDirection
            shared_flux[3, i, j, k] =
                M * (ξ3x1 * F1 + ξ3x2 * F2 + ξ3x3 * F3)
        end

        fill!(local_source, -zero(eltype(local_source)))
        schur_rhs_nonconservative!(
            schur_complement,
            balance_law,
            Vars{schur_vars_state(schur_complement, FT)}(local_source),
            Vars{vars_state_conservative(balance_law, FT)}(
                local_state_conservative,
            ),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary),
            α,
        )

        local_schur_rhs[1] += local_source[1]
        @synchronize

        # Weak "inside metrics" derivative
        MI = vgeo[ijk, _MI, e]
        @unroll for n in 1:Nq
            # ξ1-grid lines
            local_schur_rhs[1] -= MI * s_D[n, i] * shared_flux[1, n, j, k]

            # ξ2-grid lines
            if dim == 3 || (dim == 2 && direction isa EveryDirection)
                local_schur_rhs[1] -=
                    MI * s_D[n, j] * shared_flux[2, i, n, k]
            end

            # ξ3-grid lines
            if dim == 3 && direction isa EveryDirection
                local_schur_rhs[1] -=
                    MI * s_D[n, k] * shared_flux[3, i, j, n]
            end
        end
        
        ijk = i + Nq * ((j - 1) + Nq * (k - 1))
        schur_rhs[ijk, 1, e] = local_schur_rhs[1]
    end
end

@kernel function schur_interface_rhs!(
    schur_complement::SchurComplement,
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    schur_rhs,
    state_conservative,
    schur_state_auxiliary,
    vgeo,
    sgeo,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
    α
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(state_conservative)

        num_state_conservative = number_state_conservative(balance_law, FT)
        schur_num_state_auxiliary = schur_number_state_auxiliary(schur_complement, FT)

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
        local_schur_state_auxiliary⁻ = MArray{Tuple{schur_num_state_auxiliary}, FT}(undef)
        local_flux⁻ = MArray{Tuple{3, 1}, FT}(undef)
        
        local_state_conservative⁺ =
            MArray{Tuple{num_state_conservative}, FT}(undef)
        local_schur_state_auxiliary⁺ = MArray{Tuple{schur_num_state_auxiliary}, FT}(undef)
        local_flux⁺ = MArray{Tuple{3, 1}, FT}(undef)
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

        @unroll for s in 1:schur_num_state_auxiliary
            local_schur_state_auxiliary⁻[s] = schur_state_auxiliary[vid⁻, s, e⁻]
        end
        
        # Load plus side data
        @unroll for s in 1:num_state_conservative
          local_state_conservative⁺[s] = state_conservative[vid⁺, s, e⁺]
        end

        @unroll for s in 1:schur_num_state_auxiliary
            local_schur_state_auxiliary⁺[s] = schur_state_auxiliary[vid⁺, s, e⁺]
        end

        bctype = elemtobndy[f, e⁻]
        if bctype != 0
           schur_rhs_boundary_state!(
             schur_complement,
             balance_law,
             Vars{vars_state_conservative(balance_law, FT)}(
                 local_state_conservative⁺,
             ),
             Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁺),
             normal_vector,
             Vars{vars_state_conservative(balance_law, FT)}(
                 local_state_conservative⁻,
             ),
             Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁻),
             bctype
           )
        end
        fill!(local_flux⁻, -zero(eltype(local_flux⁻)))
        schur_rhs_conservative!(
            schur_complement,
            balance_law,
            Grad{schur_vars_state(schur_complement, FT)}(local_flux⁻),
            Vars{vars_state_conservative(balance_law, FT)}(
                local_state_conservative⁻,
            ),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁻),
            α
        )
        
        fill!(local_flux⁺, -zero(eltype(local_flux⁺)))
        schur_rhs_conservative!(
            schur_complement,
            balance_law,
            Grad{schur_vars_state(schur_complement, FT)}(local_flux⁺),
            Vars{vars_state_conservative(balance_law, FT)}(
                local_state_conservative⁺,
            ),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁺),
            α
        )
        local_schur_rhs = normal_vector' * SVector(local_flux⁻ + local_flux⁺) / 2

        #Update RHS
        schur_rhs[vid⁻, 1, e⁻] += vMI * sM * local_schur_rhs
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

@kernel function schur_volume_update!(
    schur_complement::SchurComplement,
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    state_lhs,
    state_rhs,
    schur_state,
    schur_state_gradient,
    schur_state_auxiliary,
    vgeo,
    D,
    α
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(schur_state)
        
        num_state_conservative = number_state_conservative(balance_law, FT)
        schur_num_state_gradient = schur_number_state_gradient(schur_complement, FT)
        schur_num_state_auxiliary = schur_number_state_auxiliary(schur_complement, FT)

        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq

        local_source =
            MArray{Tuple{num_state_conservative}, FT}(undef)
        local_schur_state = MArray{Tuple{1,}, FT}(undef)
        local_schur_state_gradient = MArray{Tuple{schur_num_state_gradient}, FT}(undef)
        local_schur_state_auxiliary = MArray{Tuple{schur_num_state_auxiliary}, FT}(undef)
        local_state_rhs = MArray{Tuple{num_state_conservative}, FT}(undef)
        local_flux = MArray{Tuple{3, num_state_conservative}, FT}(undef)
    end

    shared_flux = @localmem FT (3, Nq, Nq, Nqk, num_state_conservative)
    s_D = @localmem FT (Nq, Nq)
    
    local_state_lhs = @private FT (num_state_conservative,)

    e = @index(Group, Linear)
    ijk = @index(Local, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        s_D[i, j] = D[i, j]

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
          local_state_lhs[s] = zero(FT)
        end

        local_schur_state[1] = schur_state[ijk, 1, e]

        @unroll for s in 1:schur_num_state_gradient
            local_schur_state_gradient[s] = schur_state_gradient[ijk, s, e]
        end

        @unroll for s in 1:schur_num_state_auxiliary
            local_schur_state_auxiliary[s] = schur_state_auxiliary[ijk, s, e]
        end
        
        @unroll for s in 1:num_state_conservative
          local_state_rhs[s] = state_rhs[ijk, s, e]
        end

        fill!(local_flux, -zero(eltype(local_flux)))
        schur_update_conservative!(
            schur_complement,
            balance_law,
            Grad{vars_state_conservative(balance_law, FT)}(local_flux),
            Vars{schur_vars_state(schur_complement, FT)}(local_schur_state),
            Vars{schur_vars_state_gradient(schur_complement, FT)}(local_schur_state_gradient),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary),
            Vars{vars_state_conservative(balance_law, FT)}(local_state_rhs),
            α
        )

        @unroll for s in 1:num_state_conservative
          shared_flux[1, i, j, k, s] = local_flux[1, s]
          shared_flux[2, i, j, k, s] = local_flux[2, s]
          shared_flux[3, i, j, k, s] = local_flux[3, s]
        end

        # Build "inside metrics" flux
        @unroll for s in 1:num_state_conservative
          F1, F2, F3 = shared_flux[1, i, j, k, s],
                       shared_flux[2, i, j, k, s],
                       shared_flux[3, i, j, k, s]

          shared_flux[1, i, j, k, s] = M * (ξ1x1 * F1 + ξ1x2 * F2 + ξ1x3 * F3)
          if dim == 3 || (dim == 2 && direction isa EveryDirection)
              shared_flux[2, i, j, k, s] =
                  M * (ξ2x1 * F1 + ξ2x2 * F2 + ξ2x3 * F3)
          end
          if dim == 3 && direction isa EveryDirection
              shared_flux[3, i, j, k, s] =
                  M * (ξ3x1 * F1 + ξ3x2 * F2 + ξ3x3 * F3)
          end
        end

        fill!(local_source, -zero(eltype(local_source)))
        schur_update_nonconservative!(
            schur_complement,
            balance_law,
            Vars{vars_state_conservative(balance_law, FT)}(local_source),
            Vars{schur_vars_state(schur_complement, FT)}(
                local_schur_state,
            ),
            Vars{schur_vars_state_gradient(schur_complement, FT)}(local_schur_state_gradient),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary),
            Vars{vars_state_conservative(balance_law, FT)}(local_state_rhs),
            α
        )

        @unroll for s in 1:num_state_conservative
          local_state_lhs[s] += local_source[s]
        end
        @synchronize

        # Weak "inside metrics" derivative
        MI = vgeo[ijk, _MI, e]
        @unroll for s in 1:num_state_conservative
          @unroll for n in 1:Nq
              # ξ1-grid lines
              local_state_lhs[s] -= MI * s_D[n, i] * shared_flux[1, n, j, k, s]

              # ξ2-grid lines
              if dim == 3 || (dim == 2 && direction isa EveryDirection)
                  local_state_lhs[s] -=
                      MI * s_D[n, j] * shared_flux[2, i, n, k, s]
              end

              # ξ3-grid lines
              if dim == 3 && direction isa EveryDirection
                  local_state_lhs[s] -=
                      MI * s_D[n, k] * shared_flux[3, i, j, n, s]
              end
          end
        end
        
        ijk = i + Nq * ((j - 1) + Nq * (k - 1))
        @unroll for s in 1:num_state_conservative
          state_lhs[ijk, s, e] = local_state_lhs[s]
        end
    end
end

@kernel function schur_interface_update!(
    schur_complement::SchurComplement,
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    direction,
    state_lhs,
    state_rhs,
    schur_state,
    schur_state_gradient,
    schur_state_auxiliary,
    vgeo,
    sgeo,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
    α
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(schur_state)

        num_state_conservative = number_state_conservative(balance_law, FT)
        schur_num_state_gradient = schur_number_state_gradient(schur_complement, FT)
        schur_num_state_auxiliary = schur_number_state_auxiliary(schur_complement, FT)

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

        local_state_lhs =
            MArray{Tuple{num_state_conservative}, FT}(undef)

        local_schur_state⁻ = MArray{Tuple{1}, FT}(undef)
        local_schur_state_gradient⁻ = MArray{Tuple{schur_num_state_gradient}, FT}(undef)
        local_schur_state_auxiliary⁻ = MArray{Tuple{schur_num_state_auxiliary}, FT}(undef)
        local_state_rhs⁻ = MArray{Tuple{num_state_conservative}, FT}(undef)
        local_flux⁻ = MArray{Tuple{3, num_state_conservative}, FT}(undef)
        
        local_schur_state⁺ = MArray{Tuple{1}, FT}(undef)
        local_schur_state_auxiliary⁺ = MArray{Tuple{schur_num_state_auxiliary}, FT}(undef)
        local_schur_state_gradient⁺ = MArray{Tuple{schur_num_state_gradient}, FT}(undef)
        local_state_rhs⁺ = MArray{Tuple{num_state_conservative}, FT}(undef)
        local_flux⁺ = MArray{Tuple{3, num_state_conservative}, FT}(undef)
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
        local_schur_state⁻[1] = schur_state[vid⁻, 1, e⁻]
        @unroll for s in 1:schur_num_state_gradient
            local_schur_state_gradient⁻[s] = schur_state_gradient[vid⁻, s, e⁻]
        end
        @unroll for s in 1:schur_num_state_auxiliary
            local_schur_state_auxiliary⁻[s] = schur_state_auxiliary[vid⁻, s, e⁻]
        end
        @unroll for s in 1:num_state_conservative
            local_state_rhs⁻[s] = state_rhs[vid⁻, s, e⁻]
        end
        
        # Load plus side data
        local_schur_state⁺[1] = schur_state[vid⁺, 1, e⁺]
        @unroll for s in 1:schur_num_state_gradient
            local_schur_state_gradient⁺[s] = schur_state_gradient[vid⁺, s, e⁺]
        end
        @unroll for s in 1:schur_num_state_auxiliary
            local_schur_state_auxiliary⁺[s] = schur_state_auxiliary[vid⁺, s, e⁺]
        end
        @unroll for s in 1:num_state_conservative
            local_state_rhs⁺[s] = state_rhs[vid⁺, s, e⁺]
        end

        bctype = elemtobndy[f, e⁻]
        if bctype != 0
           schur_update_boundary_state!(
             schur_complement,
             balance_law,
             Vars{schur_vars_state(schur_complement, FT)}(local_schur_state⁺),
             Vars{schur_vars_state_gradient(schur_complement, FT)}(local_schur_state_gradient⁺),
             Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁺),
             Vars{vars_state_conservative(balance_law, FT)}(local_state_rhs⁺),
             normal_vector,
             Vars{schur_vars_state(schur_complement, FT)}(local_schur_state⁻),
             Vars{schur_vars_state_gradient(schur_complement, FT)}(local_schur_state_gradient⁻),
             Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁻),
             Vars{vars_state_conservative(balance_law, FT)}(local_state_rhs⁻),
             bctype,
           )
        end
        fill!(local_flux⁻, -zero(eltype(local_flux⁻)))
        schur_update_conservative!(
            schur_complement,
            balance_law,
            Grad{vars_state_conservative(balance_law, FT)}(local_flux⁻),
            Vars{schur_vars_state(schur_complement, FT)}(local_schur_state⁻),
            Vars{schur_vars_state_gradient(schur_complement, FT)}(local_schur_state_gradient⁻),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁻),
            Vars{vars_state_conservative(balance_law, FT)}(local_state_rhs⁻),
            α
        )
        
        fill!(local_flux⁺, -zero(eltype(local_flux⁺)))
        schur_update_conservative!(
            schur_complement,
            balance_law,
            Grad{vars_state_conservative(balance_law, FT)}(local_flux⁺),
            Vars{schur_vars_state(schur_complement, FT)}(local_schur_state⁺),
            Vars{schur_vars_state_gradient(schur_complement, FT)}(local_schur_state_gradient⁺),
            Vars{schur_vars_state_auxiliary(schur_complement, FT)}(local_schur_state_auxiliary⁺),
            Vars{vars_state_conservative(balance_law, FT)}(local_state_rhs⁺),
            α
        )
        local_state_lhs .= (local_flux⁻ + local_flux⁺)' * normal_vector / 2
        

        #Update RHS
        @unroll for s in 1:num_state_conservative
          state_lhs[vid⁻, s, e⁻] += vMI * sM * local_state_lhs[s]
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end
