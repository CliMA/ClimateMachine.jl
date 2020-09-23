#### Create states

function create_state(
    ::Type{MPIStateArray},
    balance_law,
    grid,
    st::AbstractStateType;
    fill_nan = false,
)
    topology = grid.topology
    Np = dofs_per_element(grid)

    FT = eltype(grid.vgeo)
    DA = arraytype(grid)

    weights = view(grid.vgeo, :, grid.Mid, :)
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

function create_state(
    ::Type{Array},
    balance_law,
    grid,
    st::AbstractStateType;
    fill_nan = false,
)
    topology = grid.topology
    Np = dofs_per_element(grid)

    FT = eltype(grid.vgeo)
    DA = arraytype(grid)

    weights = view(grid.vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    ns = number_states(balance_law, st)
    st isa GradientLaplacian && (ns = 3ns)
    V = vars_state(balance_law, st, FT)
    state = Array{FT}(
        undef,
        (Np, number_states(balance_law, st), length(topology.elems)),
    )
    fill_nan && fill!(state, NaN)
    return state
end


function init_state(state, balance_law, grid, direction, ::Auxiliary)
    init_state_auxiliary!(balance_law, state, grid, direction)
    return state
end
