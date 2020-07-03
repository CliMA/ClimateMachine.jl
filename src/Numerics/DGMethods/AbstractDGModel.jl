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

function init_ode_state(
    dg::AbstractDGModel,
    args...;
    init_on_cpu = false,
    fill_nan = false,
)
    device = arraytype(dg.grid) <: Array ? CPU() : CUDADevice()

    balance_law = dg.balance_law
    grid = dg.grid

    state_prognostic =
        create_state(balance_law, grid, Prognostic(), fill_nan = fill_nan)

    topology = grid.topology
    Np = dofs_per_element(grid)

    state_auxiliary = dg.state_auxiliary
    dim = dimensionality(grid)
    # XXX: Needs updating for multiple polynomial orders
    N = polynomialorders(grid)
    # Currently only support single polynomial order
    @assert all(N[1] .== N)
    N = N[1]
    nrealelem = length(topology.realelems)

    if !init_on_cpu
        event = Event(device)
        event = kernel_init_state_prognostic!(device, min(Np, 1024))(
            balance_law,
            Val(dim),
            Val(N),
            state_prognostic.data,
            state_auxiliary.data,
            grid.vgeo,
            topology.realelems,
            args...;
            ndrange = Np * nrealelem,
            dependencies = (event,),
        )
        wait(device, event)
    else
        h_state_prognostic = similar(state_prognostic, Array)
        h_state_auxiliary = similar(state_auxiliary, Array)
        h_state_auxiliary .= state_auxiliary
        event = kernel_init_state_prognostic!(CPU(), Np)(
            balance_law,
            Val(dim),
            Val(N),
            h_state_prognostic.data,
            h_state_auxiliary.data,
            Array(grid.vgeo),
            topology.realelems,
            args...;
            ndrange = Np * nrealelem,
        )
        wait(event) # XXX: This could be `wait(device, event)` once KA supports that.
        state_prognostic .= h_state_prognostic
    end

    event = Event(device)
    event = MPIStateArrays.begin_ghost_exchange!(
        state_prognostic;
        dependencies = event,
    )
    event = MPIStateArrays.end_ghost_exchange!(
        state_prognostic;
        dependencies = event,
    )
    wait(device, event)

    return state_prognostic
end

@kernel function kernel_init_state_prognostic!(
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
    num_state_auxiliary = number_states(balance_law, Auxiliary())
    num_state_prognostic = number_states(balance_law, Prognostic())

    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Np = Nq * Nq * Nqk

    l_state = MArray{Tuple{num_state_prognostic}, FT}(undef)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

    I = @index(Global, Linear)
    e = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end
        @unroll for s in 1:num_state_prognostic
            l_state[s] = state[n, s, e]
        end
        init_state_prognostic!(
            balance_law,
            Vars{vars_state(balance_law, Prognostic(), FT)}(l_state),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                local_state_auxiliary,
            ),
            LocalGeometry{Np, N}(vgeo, n, e),
            args...,
        )
        @unroll for s in 1:num_state_prognostic
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
    function interface_tendency!(
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
    )
Compute kernel for evaluating the interface tendencies for the
DG form:
∫ₑ ψ⋅ ∂q/∂t dx - ∫ₑ ∇ψ⋅(Fⁱⁿᵛ + Fᵛⁱˢᶜ) dx + ∮ₑ n̂ ψ⋅(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆) dS,
or equivalently in matrix form:
dQ/dt = M⁻¹(MS + DᵀM(Fⁱⁿᵛ + Fᵛⁱˢᶜ) + ∑ᶠ LᵀMf(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆)).
This kernel computes the surface terms: M⁻¹ ∑ᶠ LᵀMf(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆)),
where M is the mass matrix, Mf is the face mass matrix, L is an interpolator
from volume to face, and Fⁱⁿᵛ⋆, Fᵛⁱˢᶜ⋆
are the numerical fluxes for the inviscid and viscous
fluxes, respectively.
""" interface_tendency!
@kernel function interface_tendency!(
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
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        nhyperviscstate = number_states(balance_law, Hyperdiffusive())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        ngradlapstate = number_states(balance_law, GradientLaplacian())

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

        local_state_prognostic⁻ = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_gradient_flux⁻ =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_hyperdiffusion⁻ = MArray{Tuple{nhyperviscstate}, FT}(undef)
        local_state_auxiliary⁻ = MArray{Tuple{num_state_auxiliary}, FT}(undef)

        # Need two copies since numerical_flux_first_order! can modify state_prognostic⁺
        local_state_prognostic⁺nondiff =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic⁺diff =
            MArray{Tuple{num_state_prognostic}, FT}(undef)

        # Need two copies since numerical_flux_first_order! can modify state_auxiliary⁺
        local_state_auxiliary⁺nondiff =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_auxiliary⁺diff =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_state_gradient_flux⁺ =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_hyperdiffusion⁺ = MArray{Tuple{nhyperviscstate}, FT}(undef)

        local_state_prognostic_bottom1 =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_gradient_flux_bottom1 =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_auxiliary_bottom1 =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_flux = MArray{Tuple{num_state_prognostic}, FT}(undef)
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
        # Get surface mass, volume mass inverse
        sM, vMI = sgeo[_sM, n, f, e⁻], sgeo[_vMI, n, f, e⁻]
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1

        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1

        # Load minus side data
        @unroll for s in 1:num_state_prognostic
            local_state_prognostic⁻[s] = state_prognostic[vid⁻, s, e⁻]
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
        @unroll for s in 1:num_state_prognostic
            local_state_prognostic⁺diff[s] =
                local_state_prognostic⁺nondiff[s] =
                    state_prognostic[vid⁺, s, e⁺]
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

        # Oh dang, it's boundary conditions
        bctag = elemtobndy[f, e⁻]
        fill!(local_flux, -zero(eltype(local_flux)))
        if bctag == 0
            numerical_flux_first_order!(
                numerical_flux_first_order,
                balance_law,
                Vars{vars_state(balance_law, Prognostic(), FT)}(local_flux),
                SVector(normal_vector),
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁻,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁺nondiff,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁺nondiff,
                ),
                t,
                face_direction,
            )
            numerical_flux_second_order!(
                numerical_flux_second_order,
                balance_law,
                Vars{vars_state(balance_law, Prognostic(), FT)}(local_flux),
                normal_vector,
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁻,
                ),
                Vars{vars_state(balance_law, GradientFlux(), FT)}(
                    local_state_gradient_flux⁻,
                ),
                Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                    local_state_hyperdiffusion⁻,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁺diff,
                ),
                Vars{vars_state(balance_law, GradientFlux(), FT)}(
                    local_state_gradient_flux⁺,
                ),
                Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                    local_state_hyperdiffusion⁺,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁺diff,
                ),
                t,
            )
        else
            if (dim == 2 && f == 3) || (dim == 3 && f == 5)
                # Loop up the first element along all horizontal elements
                @unroll for s in 1:num_state_prognostic
                    local_state_prognostic_bottom1[s] =
                        state_prognostic[n + Nqk^2, s, e⁻]
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

            bcs = boundary_conditions(balance_law)
            # TODO: there is probably a better way to unroll this loop
            Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
                bc = bcs[d]
                numerical_boundary_flux_first_order!(
                    numerical_flux_first_order,
                    bc,
                    balance_law,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(local_flux),
                    SVector(normal_vector),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁻,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁻,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁺nondiff,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁺nondiff,
                    ),
                    t,
                    face_direction,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic_bottom1,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary_bottom1,
                    ),
                )
                numerical_boundary_flux_second_order!(
                    numerical_flux_second_order,
                    bc,
                    balance_law,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(local_flux),
                    normal_vector,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁻,
                    ),
                    Vars{vars_state(balance_law, GradientFlux(), FT)}(
                        local_state_gradient_flux⁻,
                    ),
                    Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                        local_state_hyperdiffusion⁻,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁻,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁺diff,
                    ),
                    Vars{vars_state(balance_law, GradientFlux(), FT)}(
                        local_state_gradient_flux⁺,
                    ),
                    Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                        local_state_hyperdiffusion⁺,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁺diff,
                    ),
                    t,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic_bottom1,
                    ),
                    Vars{vars_state(balance_law, GradientFlux(), FT)}(
                        local_state_gradient_flux_bottom1,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary_bottom1,
                    ),
                )
            end d -> throw(BoundsError(bcs, bctag))
        end

        # Update RHS (in outer face loop): M⁻¹ Mfᵀ(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆))
        @unroll for s in 1:num_state_prognostic
            # FIXME: Should we pretch these?
            tendency[vid⁻, s, e⁻] -= α * vMI * sM * local_flux[s]
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end
