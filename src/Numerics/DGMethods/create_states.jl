#### Create states

function create_state(
    balance_law,
    grid,
    st::AbstractStateType;
    fill_nan = false,
)
    topology = grid.topology
    Np = dofs_per_element(grid)

    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    # TODO: Clean up this MPIStateArray interface...
    ns = number_states(balance_law, st)
    st isa GradientLaplacian && (ns = 3ns)
    V = vars_state(balance_law, st, FT)
    state = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        ns,
        length(topology.elems),
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
        weights = weights,
    )
    fill_nan && fill!(state, NaN)
    return state
end

function init_state(state, balance_law, grid, ::Auxiliary)
    topology = grid.topology
    Np = dofs_per_element(grid)
    dim = dimensionality(grid)
    polyorder = polynomialorder(grid)
    vgeo = grid.vgeo
    device = array_device(state)
    nrealelem = length(topology.realelems)
    event = Event(device)
    event = kernel_init_state_auxiliary!(device, min(Np, 1024), Np * nrealelem)(
        balance_law,
        Val(dim),
        Val(polyorder),
        state.data,
        vgeo,
        topology.realelems,
        dependencies = (event,),
    )
    event = MPIStateArrays.begin_ghost_exchange!(state; dependencies = event)
    event = MPIStateArrays.end_ghost_exchange!(state; dependencies = event)
    wait(device, event)

    return state
end
