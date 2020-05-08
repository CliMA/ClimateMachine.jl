abstract type SchurComplement end # PDE part

schur_vars_state(::SchurComplement, FT) = @vars(p::FT)
schur_vars_state_gradient(::SchurComplement, FT) = @vars(∇p::SVector{3, FT})

function schur_vars_state_auxiliary end
function schur_vars_gradient_auxiliary end

schur_number_state(m::SchurComplement, FT) = varsize(schur_vars_state(m, FT))
schur_number_state_auxiliary(m::SchurComplement, FT) = varsize(schur_vars_state_auxiliary(m, FT))
schur_number_state_gradient(m::SchurComplement, FT) = varsize(schur_vars_state_gradient(m, FT))
schur_number_gradient_auxiliary(m::SchurComplement, FT) = varsize(schur_vars_gradient_auxiliary(m, FT))

function schur_init_aux! end
function schur_init_state! end
function schur_extract_state! end

function schur_lhs_conservative! end
function schur_lhs_nonconservative! end

function create_schur_state(schur_complement::SchurComplement, grid)
    topology = grid.topology
    # FIXME: Remove after updating CUDA
    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    Np = dofs_per_element(grid)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    V = schur_vars_state(schur_complement, FT)
    schur_state = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        schur_number_state(schur_complement, FT),
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
    return schur_state
end

# TODO: get rid of this
function create_schur_indexmap(schur_complement::SchurComplement)
    # helper function
    _getvars(v, ::Type) = v
    function _getvars(v::Vars, ::Type{T}) where {T <: NamedTuple}
        fields = getproperty.(Ref(v), fieldnames(T))
        collect(Iterators.Flatten(_getvars.(fields, fieldtypes(T))))
    end

    auxvars = schur_vars_state_auxiliary(schur_complement, Int)
    auxgradvars = schur_vars_gradient_auxiliary(schur_complement, Int)
    indices = Vars{auxvars}(1:varsize(auxvars))
    SVector{varsize(auxgradvars)}(_getvars(indices, auxgradvars))
end

function create_schur_auxiliary_state(schur_complement::SchurComplement, balance_law::BalanceLaw,
                                      grid, state_auxiliary)
    topology = grid.topology
    Np = dofs_per_element(grid)
    Nq = polynomialorder(grid) + 1

    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    V = schur_vars_state_auxiliary(schur_complement, FT)
    schur_auxstate = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        schur_number_state_auxiliary(schur_complement, FT),
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

    dim = dimensionality(grid)
    polyorder = polynomialorder(grid)
    vgeo = grid.vgeo
    device = typeof(schur_auxstate.data) <: Array ? CPU() : CUDA()
    nrealelem = length(topology.realelems)
    event = Event(device)
    event = schur_init_auxiliary_state!(device, Np, Np * nrealelem)(
        schur_complement,
        balance_law,
        schur_auxstate.data,
        state_auxiliary.data,
        vgeo,
        topology.realelems,
        Val(dim),
        Val(polyorder),
        dependencies = (event,),
    )
    event = MPIStateArrays.begin_ghost_exchange!(schur_auxstate; dependencies = event)
    event = MPIStateArrays.end_ghost_exchange!(schur_auxstate; dependencies = event)
    
    schur_indexmap = create_schur_indexmap(schur_complement)
    event = schur_auxiliary_gradients!(device, (Nq, Nq, Nq), (Nq * nrealelem, Nq, Nq))(
        schur_complement,
        schur_auxstate.data,
        vgeo,
        grid.D,
        schur_indexmap,
        EveryDirection(),
        Val(dim),
        Val(polyorder),
        dependencies = (event,),
    )
    event = MPIStateArrays.begin_ghost_exchange!(schur_auxstate; dependencies = event)
    event = MPIStateArrays.end_ghost_exchange!(schur_auxstate; dependencies = event)

    wait(device, event)

    return schur_auxstate
end

function create_schur_gradient_state(schur_complement::SchurComplement, grid)
    topology = grid.topology
    Np = dofs_per_element(grid)

    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    # TODO: Clean up this MPIStateArray interface...
    V = schur_vars_state_gradient(schur_complement, FT)
    schur_gradient_state = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        schur_number_state_gradient(schur_complement, FT),
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

    return schur_gradient_state
end
