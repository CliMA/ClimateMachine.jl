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

    state_prognostic = create_state(balance_law, grid, Prognostic())

    return state_prognostic
end

"""
    init_ode_state(
        dg::AbstractDGModel
        args...;
        init_on_cpu = false,
        fill_nan = false,
    )

Create an [`MPIStateArray`](@ref) from the given `dg` model which matches the
array type of `dg.grid`. The array is initialized by calling the
[`init_state_conservative!`](@ref) function for the underlying `dg.balance_law`.
The `args` are passed through to the `init_state_conservative!` function.

If `init_on_cpu == true` then the initialization is done using a host side
kernel, otherwise the kernel is launched on the device which supports the
underlying `dg.grid` data.

If `fill_nan == true` then the array is filled with `NaN`s before calling
`init_state_prognostic!`.
"""
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
    N = polynomialorder(grid)
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
