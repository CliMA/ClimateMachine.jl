module ESDGMethods

struct ESDGModel{BL, P, G, SA}
    balance_law::BL
    problem::P
    grid::G
    state_auxiliary::SA
end

function (esdg::DGModel)(
    tendency,
    state_conservative,
    ::Nothing,
    param,
    t;
    increment = false,
)
    # TODO deprecate increment argument
    dg(tendency, state_conservative, param, t, true, increment)
end


function (dg::ESDGModel)(
    tendency,
    state_conservative,
    params::Nothing,
    t,
    α,
    β,
)

    balance_law = dg.balance_law
    device = array_device(state_conservative)

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk
    nrealelem = length(topology.realelems)

    state_auxiliary = dg.state_auxiliary

    FT = eltype(state_conservative)
    num_state_conservative = number_state_conservative(balance_law, FT)

    Np = dofs_per_element(grid)

    workgroups_volume = (Nq, Nq, Nqk)
    ndrange_volume = (nrealelem * Nq, Nq, Nqk)
    workgroups_surface = Nfp
    ndrange_interior_surface = Nfp * length(grid.interiorelems)
    ndrange_exterior_surface = Nfp * length(grid.exteriorelems)

    # XXX: When we do stacked meshes and IMEX this will change
    communicate = true

    exchange_state_conservative = NoneEvent()

    comp_stream = Event(device)

    ########################
    # tendency Computation #
    ########################
    if communicate
        exchange_state_conservative = MPIStateArrays.begin_ghost_exchange!(
            state_conservative;
            dependencies = comp_stream,
        )
    end

    comp_stream = volume_tendency!(device, (Nq, Nq))(
        balance_law,
        Val(dim),
        Val(N),
        dg.volume_two_point_flux,
        tendency.data,
        state_conservative.data,
        state_auxiliary.data,
        grid.vgeo,
        t,
        grid.ω,
        grid.D,
        topology.realelems,
        α,
        β,
        ndrange = (nrealelem * Nq, Nq),
        dependencies = (comp_stream,),
    )

    comp_stream = interface_tendency!(device, workgroups_surface)(
        balance_law,
        Val(dim),
        Val(N),
        dg.interface_two_point_flux,
        tendency.data,
        state_conservative.data,
        state_auxiliary.data,
        grid.vgeo,
        grid.sgeo,
        t,
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
            state_conservative;
            dependencies = exchange_state_conservative,
        )

        # update_aux may start asynchronous work on the compute device and
        # we synchronize those here through a device event.
        wait(device, exchange_state_conservative)
        update_auxiliary_state!(
            dg,
            balance_law,
            state_conservative,
            t,
            dg.grid.topology.ghostelems,
        )
        exchange_state_conservative = Event(device)
    end

    comp_stream = interface_tendency!(device, workgroups_surface)(
        balance_law,
        Val(dim),
        Val(N),
        tendency.data,
        state_conservative.data,
        state_auxiliary.data,
        grid.vgeo,
        grid.sgeo,
        t,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.exteriorelems,
        α;
        ndrange = ndrange_exterior_surface,
        dependencies = (
            comp_stream,
            exchange_state_conservative,
        ),
    )

    # The synchronization here through a device event prevents CuArray based and
    # other default stream kernels from launching before the work scheduled in
    # this function is finished.
    wait(device, comp_stream)
end

end