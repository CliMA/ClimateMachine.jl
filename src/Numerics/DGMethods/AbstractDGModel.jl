"""
    AbstractDGModel{BL, G, A}

Abstract supertype for defining DG method for a balance law of type `BL`,
grid of type `G`, with auxiliary data array of type `A`.

The following field access is assumed:

 - `balance_law` to access the balance law
 - `grid` to access the grid data
 - `state_auxiliary` to access the auxiliary data
"""
abstract type AbstractDGModel{BL, G, A} end

"""
    MPIStateArrays.MPIStateArray(dg::AbstractDGModel)

Create an unitialized `MPIStateArray` from the given `dg` model. The backing
array type is inferred from `dg.grid`.
"""
function MPIStateArrays.MPIStateArray(dg::AbstractDGModel)
    balance_law = dg.balance_law
    grid = dg.grid

    state_conservative = create_conservative_state(balance_law, grid)

    return state_conservative
end

"""
    init_ode_state(
        dg::AbstractDGModel{BL, G, A},
        args...;
        init_on_cpu = false,
    ) where {
        BL <: BalanceLaw,
        G <: DiscontinuousSpectralElementGrid,
        A <: MPIStateArray,
    }

Create an [`MPIStateArray`](@ref) from the given `dg` model which matches the
array type of `dg.grid`. The array is initialized by calling the
[`init_state_conservative!`](@ref) function for the underlying `dg.balance_law`.
The `args` are passed through to the `init_state_conservative!` function.

If `init_on_cpu == true` then the initialization is done using a host side
kernel, otherwise the kernel is launched on the device which supports the
underlying `dg.grid` data.
"""
function init_ode_state(
    dg::AbstractDGModel{BL, G, A},
    args...;
    init_on_cpu = false,
) where {
    BL <: BalanceLaw,
    G <: DiscontinuousSpectralElementGrid,
    A <: MPIStateArray,
}
    device = arraytype(dg.grid) <: Array ? CPU() : CUDADevice()

    balance_law = dg.balance_law
    grid = dg.grid

    state_conservative = MPIStateArray(dg)

    topology = grid.topology
    Np = dofs_per_element(grid)

    state_auxiliary = dg.state_auxiliary
    dim = dimensionality(grid)
    N = polynomialorder(grid)
    nrealelem = length(topology.realelems)

    if !init_on_cpu
        event = Event(device)
        event = kernel_init_state_conservative!(device, min(Np, 1024))(
            balance_law,
            Val(dim),
            Val(N),
            state_conservative.data,
            state_auxiliary.data,
            grid.vgeo,
            topology.realelems,
            args...;
            ndrange = Np * nrealelem,
            dependencies = (event,),
        )
        wait(device, event)
    else
        h_state_conservative = similar(state_conservative, Array)
        h_state_auxiliary = similar(state_auxiliary, Array)
        h_state_auxiliary .= state_auxiliary
        event = kernel_init_state_conservative!(CPU(), Np)(
            balance_law,
            Val(dim),
            Val(N),
            h_state_conservative.data,
            h_state_auxiliary.data,
            Array(grid.vgeo),
            topology.realelems,
            args...;
            ndrange = Np * nrealelem,
        )
        wait(event) # XXX: This could be `wait(device, event)` once KA supports that.
        state_conservative .= h_state_conservative
    end

    event = Event(device)
    event = MPIStateArrays.begin_ghost_exchange!(
        state_conservative;
        dependencies = event,
    )
    event = MPIStateArrays.end_ghost_exchange!(
        state_conservative;
        dependencies = event,
    )
    wait(device, event)

    return state_conservative
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

"""
    (dg::AbstractDGModel)(
        tendency,
        state_conservative,
        ::Nothing,
        param,
        t;
        increment = false,
    )

Wrapper for `dg(tendency, state_conservative, param, t, true, increment)`

This interface for the `AbstractDGModel` functor will be depreciated.
"""
function (dg::AbstractDGModel)(
    tendency,
    state_conservative,
    param,
    t;
    increment = false,
)
    dg(tendency, state_conservative, param, t, true, increment)
end

"""
    (dg::AbstractDGModel)(
        tendency,
        state_conservative,
        param,
        t,
        α = true,
        β = false,
    )

Each `AbstractDGModel` should implement a functor that computes

    tendency .= α .* dQdt(state_conservative, param, t) .+ β .* tendency

When called in 4-argument form it should compute

    tendency .= dQdt(state_conservative, param, t)
"""
function (dg::AbstractDGModel) end

@doc """
    interface_tendency!(balance_law::BalanceLaw, Val(polyorder),
            numerical_flux_first_order,
            numerical_flux_second_order,
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
    numerical_flux_first_order,
    numerical_flux_second_order,
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
    α,
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
        # The remainder model needs to know which direction of face the model is
        # being evaluated for. So faces 1:(nface - 2) are flagged as
        # `HorizontalDirection()` faces and the remaining two faces are
        # `VerticalDirection()` faces
        face_direction =
            f in 1:(nface - 2) ? (EveryDirection(), HorizontalDirection()) :
            (EveryDirection(), VerticalDirection())
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
                normal_vector,
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
                face_direction,
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
                normal_vector,
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
                face_direction,
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
            tendency[vid⁻, s, e⁻] -= α * vMI * sM * local_flux[s]
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

function MPIStateArrays.weightedsum(
    integrand::Function,
    dg::AbstractDGModel,
    state_conservative::MPIStateArray,
)
    tmp = similar(
        state_conservative,
        size(state_conservative, 1),
        1,
        size(state_conservative, 3),
    )
end

@kernel function integrand_compute_knl!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    tmp,
    integrand,
    state_conservative,
    state_auxiliary,
) where {dim, polynomialorder} end
