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

function schur_lhs_conservative! end
function schur_lhs_nonconservative! end
function schur_rhs_conservative! end
function schur_rhs_nonconservative! end
function schur_update_conservative! end
function schur_update_nonconservative! end

function schur_lhs_boundary_state! end
function schur_gradient_boundary_state! end
function schur_update_boundary_state! end
function schur_rhs_boundary_state! end

create_schur_states(::Nothing, _...) = nothing
function create_schur_states(schur_complement::SchurComplement,
                             balance_law::BalanceLaw,
                             state_auxiliary,
                             grid)
    schur_state = create_schur_state(schur_complement, grid)
    schur_rhs = create_schur_state(schur_complement, grid)
    schur_state_auxiliary = create_schur_auxiliary_state(schur_complement,
                                                         balance_law,
                                                         grid,
                                                         state_auxiliary)
    schur_state_gradient = create_schur_gradient_state(schur_complement, grid)
    return (state=schur_state,
            gradient=schur_state_gradient,
            auxiliary=schur_state_auxiliary,
            rhs=schur_rhs)
end

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
    dim = dimensionality(grid)
    Nqk = dim == 2 ? 1 : Nq

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
    event = schur_auxiliary_gradients!(device, (Nq, Nq, Nqk), (Nq * nrealelem, Nq, Nqk))(
        schur_complement,
        balance_law,
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

function schur_lhs!(
    schur_lhs,
    schur_state,
    α, dg
)
    schur_complement = dg.schur_complement
    balance_law = dg.balance_law
    device = typeof(schur_state.data) <: Array ? CPU() : CUDA()

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk
    nrealelem = length(topology.realelems)

    schur_state_gradient = dg.states_schur_complement.gradient
    schur_state_auxiliary = dg.states_schur_complement.auxiliary

    FT = eltype(schur_state)

    workgroups_surface = Nfp
    ndrange_interior_surface = Nfp * length(grid.interiorelems)
    ndrange_exterior_surface = Nfp * length(grid.exteriorelems)

    communicate =
        !(isstacked(topology) && typeof(dg.direction) <: VerticalDirection)

    exchange_schur_state = NoneEvent()
    exchange_schur_state_gradient = NoneEvent()

    comp_stream = Event(device)

    if communicate
        exchange_schur_state = MPIStateArrays.begin_ghost_exchange!(
            schur_state;
            dependencies = comp_stream,
        )
    end

    comp_stream = schur_volume_gradients!(device, (Nq, Nq, Nqk))(
        schur_complement,
        balance_law,
        schur_state_gradient.data,
        schur_state.data,
        schur_state_auxiliary.data,
        grid.vgeo,
        grid.D,
        dg.direction,
        Val(dim),
        Val(N),
        ndrange = (Nq * nrealelem, Nq, Nqk),
        dependencies = (comp_stream,),
    )

    comp_stream = schur_interface_gradients!(device, workgroups_surface)(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        schur_state_gradient.data,
        schur_state.data,
        grid.vgeo,
        grid.sgeo,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.interiorelems;
        ndrange = ndrange_interior_surface,
        dependencies = (comp_stream,),
    )
    
    if communicate
        exchange_schur_state = MPIStateArrays.end_ghost_exchange!(
            schur_state;
            dependencies = exchange_schur_state,
        )
    end
    
    comp_stream = schur_interface_gradients!(device, workgroups_surface)(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        schur_state_gradient.data,
        schur_state.data,
        grid.vgeo,
        grid.sgeo,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.exteriorelems;
        ndrange = ndrange_exterior_surface,
        dependencies = (comp_stream, exchange_schur_state),
    )

    if communicate
       exchange_schur_state_gradient =
           MPIStateArrays.begin_ghost_exchange!(
               schur_state_gradient,
               dependencies = comp_stream,
           )
    end

    comp_stream = schur_volume_lhs!(device, (Nq, Nq, Nqk))(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        schur_lhs.data,
        schur_state.data,
        schur_state_auxiliary.data,
        schur_state_gradient.data,
        grid.vgeo,
        grid.D,
        α;
        ndrange = (nrealelem * Nq, Nq, Nqk),
        dependencies = (comp_stream,),
    )
    
    comp_stream = schur_interface_lhs!(device, workgroups_surface)(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        schur_lhs.data,
        schur_state.data,
        schur_state_auxiliary.data,
        schur_state_gradient.data,
        grid.vgeo,
        grid.sgeo,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.interiorelems,
        α;
        ndrange = ndrange_interior_surface,
        dependencies = (comp_stream,),
    )
    
    if communicate
       exchange_schur_state_gradient =
           MPIStateArrays.end_ghost_exchange!(
               schur_state_gradient,
               dependencies = exchange_schur_state_gradient
           )
    end
    
    comp_stream = schur_interface_lhs!(device, workgroups_surface)(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        schur_lhs.data,
        schur_state.data,
        schur_state_auxiliary.data,
        schur_state_gradient.data,
        grid.vgeo,
        grid.sgeo,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.exteriorelems,
        α;
        ndrange = ndrange_exterior_surface,
        dependencies = (comp_stream, exchange_schur_state_gradient)
    )

    ## The synchronization here through a device event prevents CuArray based and
    ## other default stream kernels from launching before the work scheduled in
    ## this function is finished.
    wait(device, comp_stream)
end

function init_schur_state(state_conservative, α, dg)
    grid = dg.grid
    device = arraytype(grid) <: Array ? CPU() : CUDA()

    topology = grid.topology
    Np = dofs_per_element(grid)

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk
    nrealelem = length(topology.realelems)
    
    workgroups_surface = Nfp
    ndrange_interior_surface = Nfp * length(grid.interiorelems)
    ndrange_exterior_surface = Nfp * length(grid.exteriorelems)
    
    communicate =
        !(isstacked(topology) && typeof(dg.direction) <: VerticalDirection)
    
    schur_complement = dg.schur_complement
    balance_law = dg.balance_law
    state_auxiliary = dg.state_auxiliary
    schur_state = dg.states_schur_complement.state
    schur_rhs = dg.states_schur_complement.rhs
    schur_state_auxiliary = dg.states_schur_complement.auxiliary

    exchange_state_conservative = NoneEvent()

    comp_stream = Event(device)
    
    if communicate
      exchange_state_conservative = MPIStateArrays.begin_ghost_exchange!(
          schur_state;
          dependencies = comp_stream,
      )
    end

    comp_stream = kernel_init_schur_state!(device, min(Np, 1024))(
        schur_complement,
        balance_law,
        schur_state.data,
        schur_state_auxiliary.data,
        state_conservative.data,
        state_auxiliary.data,
        grid.vgeo,
        Val(dim),
        Val(N);
        ndrange = Np * nrealelem,
        dependencies = (comp_stream,),
    )

    # init rhs
    comp_stream = schur_volume_rhs!(device, (Nq, Nq, Nqk))(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        schur_rhs.data,
        state_conservative.data,
        schur_state_auxiliary.data,
        grid.vgeo,
        grid.D,
        α,
        ndrange = (nrealelem * Nq, Nq, Nqk),
        dependencies = (comp_stream,),
    )
    #event = Event(device)
    #event = MPIStateArrays.begin_ghost_exchange!(
    #    schur_state;
    #    dependencies = event,
    #)
    #event = MPIStateArrays.end_ghost_exchange!(
    #    schur_state;
    #    dependencies = event,
    #)

    
    comp_stream = schur_interface_rhs!(device, workgroups_surface)(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        schur_rhs.data,
        state_conservative.data,
        schur_state_auxiliary.data,
        grid.vgeo,
        grid.sgeo,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.interiorelems,
        α;
        ndrange = ndrange_interior_surface,
        dependencies = (comp_stream,),
    )
    
    if communicate
      exchange_state_conservative = MPIStateArrays.end_ghost_exchange!(
          schur_state;
          dependencies = exchange_state_conservative,
      )
    end
    
    comp_stream = schur_interface_rhs!(device, workgroups_surface)(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        schur_rhs.data,
        state_conservative.data,
        schur_state_auxiliary.data,
        grid.vgeo,
        grid.sgeo,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.exteriorelems,
        α;
        ndrange = ndrange_exterior_surface,
        dependencies = (comp_stream, exchange_state_conservative),
    )
    
    if communicate
      comp_stream = MPIStateArrays.end_ghost_exchange!(
          schur_rhs;
          dependencies = comp_stream
      )
      comp_stream = MPIStateArrays.end_ghost_exchange!(
          schur_rhs;
          dependencies = comp_stream,
      )
    end

    wait(device, comp_stream)
end

function schur_extract_state(state_lhs, state_rhs, α, dg)
    grid = dg.grid
    device = arraytype(grid) <: Array ? CPU() : CUDA()

    topology = grid.topology
    Np = dofs_per_element(grid)

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk
    nrealelem = length(topology.realelems)
    workgroups_surface = Nfp
    ndrange_interior_surface = Nfp * length(grid.interiorelems)
    ndrange_exterior_surface = Nfp * length(grid.exteriorelems)
    
    communicate =
        !(isstacked(topology) && typeof(dg.direction) <: VerticalDirection)
    
    schur_complement = dg.schur_complement
    balance_law = dg.balance_law
    state_auxiliary = dg.state_auxiliary
    schur_state = dg.states_schur_complement.state
    schur_rhs = dg.states_schur_complement.rhs
    schur_state_auxiliary = dg.states_schur_complement.auxiliary
    schur_state_gradient = dg.states_schur_complement.gradient

    exchange_schur_state = NoneEvent()
    exchange_schur_state_gradient = NoneEvent()

    comp_stream = Event(device)
    
    if communicate
      exchange_schur_state = MPIStateArrays.begin_ghost_exchange!(
          schur_state;
          dependencies = comp_stream,
      )
      exchange_schur_state_gradient = MPIStateArrays.begin_ghost_exchange!(
          schur_state_gradient;
          dependencies = comp_stream,
      )
    end

    comp_stream = schur_volume_update!(device, (Nq, Nq, Nqk))(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        state_lhs.data,
        state_rhs.data,
        schur_state.data,
        schur_state_gradient.data,
        schur_state_auxiliary.data,
        grid.vgeo,
        grid.D,
        α,
        ndrange = (nrealelem * Nq, Nq, Nqk),
        dependencies = (comp_stream,),
    )
    
    comp_stream = schur_interface_update!(device, workgroups_surface)(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        state_lhs.data,
        state_rhs.data,
        schur_state.data,
        schur_state_gradient.data,
        schur_state_auxiliary.data,
        grid.vgeo,
        grid.sgeo,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.interiorelems,
        α;
        ndrange = ndrange_interior_surface,
        dependencies = (comp_stream,),
    )
    
    if communicate
      exchange_schur_state = MPIStateArrays.end_ghost_exchange!(
          schur_state;
          dependencies = exchange_schur_state,
      )
      exchange_schur_state_gradient = MPIStateArrays.end_ghost_exchange!(
          schur_state_gradient;
          dependencies = exchange_schur_state_gradient,
      )
    end
    
    comp_stream = schur_interface_update!(device, workgroups_surface)(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        state_lhs.data,
        state_rhs.data,
        schur_state.data,
        schur_state_gradient.data,
        schur_state_auxiliary.data,
        grid.vgeo,
        grid.sgeo,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.exteriorelems,
        α;
        ndrange = ndrange_exterior_surface,
        dependencies = (comp_stream, exchange_schur_state, exchange_schur_state_gradient),
    )

    wait(device, comp_stream)
end
