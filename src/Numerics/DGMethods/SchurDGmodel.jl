struct SchurDGModel{SC, BL, G, SS, SA, SD, D}
    schur_complement::SC
    balance_law::BL
    grid::G
    schur_state::SS
    schur_rhs::SS
    schur_state_auxiliary::SA
    schur_state_gradient::SD
    direction::D
end
function SchurDGModel(
    schur_complement,
    balance_law,
    grid,
    state_auxiliary;
    schur_state = create_schur_state(schur_complement, grid),
    schur_rhs = create_schur_state(schur_complement, grid),
    schur_state_auxiliary = create_schur_auxiliary_state(schur_complement, balance_law, grid, state_auxiliary),
    schur_state_gradient = create_schur_gradient_state(schur_complement, grid),
    direction = EveryDirection(),
)
    SchurDGModel(
        schur_complement,
        balance_law,
        grid,
        schur_state,
        schur_rhs,
        schur_state_auxiliary,
        schur_state_gradient,
        direction,
    )
end

function (dg::SchurDGModel)(
    schur_lhs,
    schur_state,
    α,
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

    schur_state_gradient = dg.schur_state_gradient
    schur_state_auxiliary = dg.schur_state_auxiliary
    FT = eltype(schur_state)

    #num_state_gradient = number_state_gradient(balance_law, FT)
    #Np = dofs_per_element(grid)

    #workgroups_volume = (Nq, Nq, Nqk)
    #ndrange_volume = (nrealelem * Nq, Nq, Nqk)
    workgroups_surface = Nfp
    ndrange_interior_surface = Nfp * length(grid.interiorelems)
    #ndrange_exterior_surface = Nfp * length(grid.exteriorelems)

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
    wait(comp_stream)

    ## The synchronization here through a device event prevents CuArray based and
    ## other default stream kernels from launching before the work scheduled in
    ## this function is finished.
    #wait(device, comp_stream)
end

function init_schur_state(schur_dg, state_conservative, state_auxiliary, α)
    grid = schur_dg.grid
    device = arraytype(grid) <: Array ? CPU() : CUDA()

    topology = grid.topology
    Np = dofs_per_element(grid)

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk
    nrealelem = length(topology.realelems)
    
    schur_complement = schur_dg.schur_complement
    balance_law = schur_dg.balance_law
    schur_state = schur_dg.schur_state
    schur_rhs = schur_dg.schur_rhs
    schur_state_auxiliary = schur_dg.schur_state_auxiliary

    event = Event(device)
    event = kernel_init_schur_state!(device, min(Np, 1024))(
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
        dependencies = (event,),
    )
    wait(device, event)

    event = Event(device)
    event = MPIStateArrays.begin_ghost_exchange!(
        schur_dg.schur_state;
        dependencies = event,
    )
    event = MPIStateArrays.end_ghost_exchange!(
        schur_dg.schur_state;
        dependencies = event,
    )

    # init rhs
    event = schur_volume_rhs!(device, (Nq, Nq, Nqk))(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        schur_dg.direction,
        schur_rhs.data,
        state_conservative.data,
        schur_state_auxiliary.data,
        grid.vgeo,
        grid.D,
        α,
        ndrange = (nrealelem * Nq, Nq, Nqk),
        dependencies = (event,),
    )
    
    workgroups_surface = Nfp
    ndrange_interior_surface = Nfp * length(grid.interiorelems)
    event = schur_interface_rhs!(device, workgroups_surface)(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        schur_dg.direction,
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
        dependencies = (event,),
    )
    wait(device, event)
end

function schur_extract_state(schur_dg, state_lhs, state_rhs, state_auxiliary, α)
    grid = schur_dg.grid
    device = arraytype(grid) <: Array ? CPU() : CUDA()

    topology = grid.topology
    Np = dofs_per_element(grid)

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk
    workgroups_surface = Nfp
    ndrange_interior_surface = Nfp * length(grid.interiorelems)
    nrealelem = length(topology.realelems)
    
    schur_complement = schur_dg.schur_complement
    balance_law = schur_dg.balance_law
    schur_state = schur_dg.schur_state
    schur_rhs = schur_dg.schur_rhs
    schur_state_auxiliary = schur_dg.schur_state_auxiliary
    schur_state_gradient = schur_dg.schur_state_gradient

    event = Event(device)
    event = schur_volume_update!(device, (Nq, Nq, Nqk))(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        schur_dg.direction,
        state_lhs.data,
        state_rhs.data,
        schur_state.data,
        schur_state_gradient.data,
        schur_state_auxiliary.data,
        grid.vgeo,
        grid.D,
        α,
        ndrange = (nrealelem * Nq, Nq, Nqk),
        dependencies = (event,),
    )
    
    event = schur_interface_update!(device, workgroups_surface)(
        schur_complement,
        balance_law,
        Val(dim),
        Val(N),
        schur_dg.direction,
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
        dependencies = (event,),
    )
    wait(device, event)
end
