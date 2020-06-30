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
