using .NumericalFluxes:
    CentralNumericalFluxHigherOrder, CentralNumericalFluxDivergence

struct DGModel{BL, G, NFND, NFD, GNF, AS, DS, HDS, D, DD, MD}
    balance_law::BL
    grid::G
    numerical_flux_first_order::NFND
    numerical_flux_second_order::NFD
    numerical_flux_gradient::GNF
    state_auxiliary::AS
    state_gradient_flux::DS
    states_higher_order::HDS
    direction::D
    diffusion_direction::DD
    modeldata::MD
end

function DGModel(
    balance_law,
    grid,
    numerical_flux_first_order,
    numerical_flux_second_order,
    numerical_flux_gradient;
    state_auxiliary = create_auxiliary_state(balance_law, grid),
    state_gradient_flux = create_gradient_state(balance_law, grid),
    states_higher_order = create_higher_order_states(balance_law, grid),
    direction = EveryDirection(),
    diffusion_direction = direction,
    modeldata = nothing,
)
    DGModel(
        balance_law,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        state_auxiliary,
        state_gradient_flux,
        states_higher_order,
        direction,
        diffusion_direction,
        modeldata,
    )
end

# Include the remainder model for composing DG models and balance laws
include("remainder.jl")

function basic_grid_info(dg::DGModel)
    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)

    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk
    Np = dofs_per_element(grid)

    nelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nelem, nvertelem)
    nrealelem = length(topology.realelems)
    nhorzrealelem = div(nrealelem, nvertelem)

    return (
        Nq = Nq,
        Nqk = Nqk,
        Nfp = Nfp,
        Np = Np,
        nvertelem = nvertelem,
        nhorzelem = nhorzelem,
        nhorzrealelem = nhorzrealelem,
        nrealelem = nrealelem,
    )
end

"""
    (dg::DGModel)(tendency, state_conservative, nothing, t, α, β)

Computes the tendency terms compatible with `IncrementODEProblem`

    tendency .= α .* dQdt(state_conservative, p, t) .+ β .* tendency

The 4-argument form will just compute

    tendency .= dQdt(state_conservative, p, t)

"""
function (dg::DGModel)(
    tendency,
    state_conservative,
    param,
    t;
    increment = false,
)
    # TODO deprecate increment argument
    dg(tendency, state_conservative, param, t, true, increment)
end

function (dg::DGModel)(tendency, state_conservative, _, t, α, β)


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

    state_gradient_flux = dg.state_gradient_flux
    Qhypervisc_grad, Qhypervisc_div = dg.states_higher_order
    state_auxiliary = dg.state_auxiliary

    FT = eltype(state_conservative)
    num_state_conservative = number_state_conservative(balance_law, FT)
    num_state_gradient_flux = number_state_gradient_flux(balance_law, FT)
    nhyperviscstate = num_hyperdiffusive(balance_law, FT)
    num_state_tendency = size(tendency, 2)

    @assert num_state_conservative ≤ num_state_tendency

    Np = dofs_per_element(grid)

    workgroups_volume = (Nq, Nq, Nqk)
    ndrange_volume = (nrealelem * Nq, Nq, Nqk)
    workgroups_surface = Nfp
    ndrange_interior_surface = Nfp * length(grid.interiorelems)
    ndrange_exterior_surface = Nfp * length(grid.exteriorelems)

    if num_state_conservative < num_state_tendency && β != 1
        # if we don't operate on the full state, then we need to scale here instead of volume_tendency!
        tendency .*= β
        β = β != 0 # if β==0 then we can avoid the memory load in volume_tendency!
    end

    communicate =
        !(isstacked(topology) && typeof(dg.direction) <: VerticalDirection)

    update_auxiliary_state!(
        dg,
        balance_law,
        state_conservative,
        t,
        dg.grid.topology.realelems,
    )

    if nhyperviscstate > 0
        hypervisc_indexmap = varsindices(
            vars_state_gradient(balance_law, FT),
            fieldnames(vars_gradient_laplacian(balance_law, FT)),
        )
    else
        hypervisc_indexmap = nothing
    end

    exchange_state_conservative = NoneEvent()
    exchange_state_gradient_flux = NoneEvent()
    exchange_Qhypervisc_grad = NoneEvent()
    exchange_Qhypervisc_div = NoneEvent()

    comp_stream = Event(device)

    ########################
    # Gradient Computation #
    ########################
    if communicate
        exchange_state_conservative = MPIStateArrays.begin_ghost_exchange!(
            state_conservative;
            dependencies = comp_stream,
        )
    end

    if num_state_gradient_flux > 0 || nhyperviscstate > 0

        comp_stream = volume_gradients!(device, (Nq, Nq))(
            balance_law,
            Val(dim),
            Val(N),
            dg.diffusion_direction,
            state_conservative.data,
            state_gradient_flux.data,
            Qhypervisc_grad.data,
            state_auxiliary.data,
            grid.vgeo,
            t,
            grid.D,
            Val(hypervisc_indexmap),
            topology.realelems,
            ndrange = (Nq * nrealelem, Nq),
            dependencies = (comp_stream,),
        )

        comp_stream = interface_gradients!(device, workgroups_surface)(
            balance_law,
            Val(dim),
            Val(N),
            dg.diffusion_direction,
            dg.numerical_flux_gradient,
            state_conservative.data,
            state_gradient_flux.data,
            Qhypervisc_grad.data,
            state_auxiliary.data,
            grid.vgeo,
            grid.sgeo,
            t,
            grid.vmap⁻,
            grid.vmap⁺,
            grid.elemtobndy,
            Val(hypervisc_indexmap),
            grid.interiorelems;
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

        comp_stream = interface_gradients!(device, workgroups_surface)(
            balance_law,
            Val(dim),
            Val(N),
            dg.diffusion_direction,
            dg.numerical_flux_gradient,
            state_conservative.data,
            state_gradient_flux.data,
            Qhypervisc_grad.data,
            state_auxiliary.data,
            grid.vgeo,
            grid.sgeo,
            t,
            grid.vmap⁻,
            grid.vmap⁺,
            grid.elemtobndy,
            Val(hypervisc_indexmap),
            grid.exteriorelems;
            ndrange = ndrange_exterior_surface,
            dependencies = (comp_stream, exchange_state_conservative),
        )

        if communicate
            if num_state_gradient_flux > 0
                exchange_state_gradient_flux =
                    MPIStateArrays.begin_ghost_exchange!(
                        state_gradient_flux,
                        dependencies = comp_stream,
                    )
            end
            if nhyperviscstate > 0
                exchange_Qhypervisc_grad = MPIStateArrays.begin_ghost_exchange!(
                    Qhypervisc_grad,
                    dependencies = comp_stream,
                )
            end
        end

        if num_state_gradient_flux > 0
            # update_aux_diffusive may start asynchronous work on the compute device
            # and we synchronize those here through a device event.
            wait(device, comp_stream)
            update_auxiliary_state_gradient!(
                dg,
                balance_law,
                state_conservative,
                t,
                dg.grid.topology.realelems,
            )
            comp_stream = Event(device)
        end
    end

    if nhyperviscstate > 0
        #########################
        # Laplacian Computation #
        #########################

        comp_stream =
            volume_divergence_of_gradients!(device, workgroups_volume)(
                balance_law,
                Val(dim),
                Val(N),
                dg.diffusion_direction,
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                grid.vgeo,
                grid.D,
                topology.realelems;
                ndrange = ndrange_volume,
                dependencies = (comp_stream,),
            )

        comp_stream =
            interface_divergence_of_gradients!(device, workgroups_surface)(
                balance_law,
                Val(dim),
                Val(N),
                dg.diffusion_direction,
                CentralNumericalFluxDivergence(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
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
            exchange_Qhypervisc_grad = MPIStateArrays.end_ghost_exchange!(
                Qhypervisc_grad,
                dependencies = exchange_Qhypervisc_grad,
            )
        end

        comp_stream =
            interface_divergence_of_gradients!(device, workgroups_surface)(
                balance_law,
                Val(dim),
                Val(N),
                dg.diffusion_direction,
                CentralNumericalFluxDivergence(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                grid.vgeo,
                grid.sgeo,
                grid.vmap⁻,
                grid.vmap⁺,
                grid.elemtobndy,
                grid.exteriorelems;
                ndrange = ndrange_exterior_surface,
                dependencies = (comp_stream, exchange_Qhypervisc_grad),
            )

        if communicate
            exchange_Qhypervisc_div = MPIStateArrays.begin_ghost_exchange!(
                Qhypervisc_div,
                dependencies = comp_stream,
            )
        end

        ####################################
        # Hyperdiffusive terms computation #
        ####################################

        comp_stream =
            volume_gradients_of_laplacians!(device, workgroups_volume)(
                balance_law,
                Val(dim),
                Val(N),
                dg.diffusion_direction,
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                state_conservative.data,
                state_auxiliary.data,
                grid.vgeo,
                grid.ω,
                grid.D,
                topology.realelems,
                t;
                ndrange = ndrange_volume,
                dependencies = (comp_stream,),
            )

        comp_stream =
            interface_gradients_of_laplacians!(device, workgroups_surface)(
                balance_law,
                Val(dim),
                Val(N),
                dg.diffusion_direction,
                CentralNumericalFluxHigherOrder(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                state_conservative.data,
                state_auxiliary.data,
                grid.vgeo,
                grid.sgeo,
                grid.vmap⁻,
                grid.vmap⁺,
                grid.elemtobndy,
                grid.interiorelems,
                t;
                ndrange = ndrange_interior_surface,
                dependencies = (comp_stream,),
            )

        if communicate
            exchange_Qhypervisc_div = MPIStateArrays.end_ghost_exchange!(
                Qhypervisc_div,
                dependencies = exchange_Qhypervisc_div,
            )
        end

        comp_stream =
            interface_gradients_of_laplacians!(device, workgroups_surface)(
                balance_law,
                Val(dim),
                Val(N),
                dg.diffusion_direction,
                CentralNumericalFluxHigherOrder(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                state_conservative.data,
                state_auxiliary.data,
                grid.vgeo,
                grid.sgeo,
                grid.vmap⁻,
                grid.vmap⁺,
                grid.elemtobndy,
                grid.exteriorelems,
                t;
                ndrange = ndrange_exterior_surface,
                dependencies = (comp_stream, exchange_Qhypervisc_div),
            )

        if communicate
            exchange_Qhypervisc_grad = MPIStateArrays.begin_ghost_exchange!(
                Qhypervisc_grad,
                dependencies = comp_stream,
            )
        end
    end


    ###################
    # RHS Computation #
    ###################
    comp_stream = volume_tendency!(device, (Nq, Nq))(
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        tendency.data,
        state_conservative.data,
        state_gradient_flux.data,
        Qhypervisc_grad.data,
        state_auxiliary.data,
        grid.vgeo,
        t,
        grid.ω,
        grid.D,
        topology.realelems,
        α,
        β;
        ndrange = (nrealelem * Nq, Nq),
        dependencies = (comp_stream,),
    )

    comp_stream = interface_tendency!(device, workgroups_surface)(
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        tendency.data,
        state_conservative.data,
        state_gradient_flux.data,
        Qhypervisc_grad.data,
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
        if num_state_gradient_flux > 0 || nhyperviscstate > 0
            if num_state_gradient_flux > 0
                exchange_state_gradient_flux =
                    MPIStateArrays.end_ghost_exchange!(
                        state_gradient_flux;
                        dependencies = exchange_state_gradient_flux,
                    )

                # update_aux_diffusive may start asynchronous work on the
                # compute device and we synchronize those here through a device
                # event.
                wait(device, exchange_state_gradient_flux)
                update_auxiliary_state_gradient!(
                    dg,
                    balance_law,
                    state_conservative,
                    t,
                    dg.grid.topology.ghostelems,
                )
                exchange_state_gradient_flux = Event(device)
            end
            if nhyperviscstate > 0
                exchange_Qhypervisc_grad = MPIStateArrays.end_ghost_exchange!(
                    Qhypervisc_grad;
                    dependencies = exchange_Qhypervisc_grad,
                )
            end
        else
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
    end

    comp_stream = interface_tendency!(device, workgroups_surface)(
        balance_law,
        Val(dim),
        Val(N),
        dg.direction,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        tendency.data,
        state_conservative.data,
        state_gradient_flux.data,
        Qhypervisc_grad.data,
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
            exchange_state_gradient_flux,
            exchange_Qhypervisc_grad,
        ),
    )

    # The synchronization here through a device event prevents CuArray based and
    # other default stream kernels from launching before the work scheduled in
    # this function is finished.
    wait(device, comp_stream)
end

function init_ode_state(dg::DGModel, args...; init_on_cpu = false)
    device = arraytype(dg.grid) <: Array ? CPU() : CUDADevice()

    balance_law = dg.balance_law
    grid = dg.grid

    state_conservative = create_conservative_state(balance_law, grid)

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

function restart_ode_state(dg::DGModel, state_data; init_on_cpu = false)
    bl = dg.balance_law
    grid = dg.grid

    state = create_conservative_state(bl, grid)
    state .= state_data

    device = arraytype(dg.grid) <: Array ? CPU() : CUDADevice()
    event = Event(device)
    event = MPIStateArrays.begin_ghost_exchange!(state; dependencies = event)
    event = MPIStateArrays.end_ghost_exchange!(state; dependencies = event)
    wait(device, event)

    return state
end

function restart_auxiliary_state(bl, grid, aux_data)
    state_auxiliary = create_auxiliary_state(bl, grid)
    state_auxiliary .= aux_data
    return state_auxiliary
end

# fallback
function update_auxiliary_state!(dg, balance_law, state_conservative, t, elems)
    return false
end

function update_auxiliary_state_gradient!(
    dg::DGModel,
    balance_law,
    state_conservative,
    t,
    elems,
)
    return false
end

function indefinite_stack_integral!(
    dg::DGModel,
    m::BalanceLaw,
    state_conservative::MPIStateArray,
    state_auxiliary::MPIStateArray,
    t::Real,
    elems::UnitRange = dg.grid.topology.elems,
)

    device = array_device(state_conservative)

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq

    FT = eltype(state_conservative)

    # do integrals
    nelem = length(elems)
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)

    event = Event(device)
    event = kernel_indefinite_stack_integral!(device, (Nq, Nqk))(
        m,
        Val(dim),
        Val(N),
        Val(nvertelem),
        state_conservative.data,
        state_auxiliary.data,
        grid.vgeo,
        grid.Imat,
        horzelems;
        ndrange = (length(horzelems) * Nq, Nqk),
        dependencies = (event,),
    )
    wait(device, event)
end

function reverse_indefinite_stack_integral!(
    dg::DGModel,
    m::BalanceLaw,
    state_conservative::MPIStateArray,
    state_auxiliary::MPIStateArray,
    t::Real,
    elems::UnitRange = dg.grid.topology.elems,
)

    device = array_device(state_auxiliary)

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq

    FT = eltype(state_auxiliary)

    # do integrals
    nelem = length(elems)
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)

    event = Event(device)
    event = kernel_reverse_indefinite_stack_integral!(device, (Nq, Nqk))(
        m,
        Val(dim),
        Val(N),
        Val(nvertelem),
        state_conservative.data,
        state_auxiliary.data,
        horzelems;
        ndrange = (length(horzelems) * Nq, Nqk),
        dependencies = (event,),
    )
    wait(device, event)
end

# TODO: Move to BalanceLaws
function nodal_update_auxiliary_state!(
    f!,
    dg::DGModel,
    m::BalanceLaw,
    state_conservative::MPIStateArray,
    t::Real,
    elems::UnitRange = dg.grid.topology.realelems;
    diffusive = false,
)
    device = array_device(state_conservative)

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    nelem = length(elems)

    Np = dofs_per_element(grid)

    nodal_update_auxiliary_state! =
        kernel_nodal_update_auxiliary_state!(device, min(Np, 1024))
    ### update state_auxiliary variables
    event = Event(device)
    if diffusive
        event = nodal_update_auxiliary_state!(
            m,
            Val(dim),
            Val(N),
            f!,
            state_conservative.data,
            dg.state_auxiliary.data,
            dg.state_gradient_flux.data,
            t,
            elems,
            grid.activedofs;
            ndrange = Np * nelem,
            dependencies = (event,),
        )
    else
        event = nodal_update_auxiliary_state!(
            m,
            Val(dim),
            Val(N),
            f!,
            state_conservative.data,
            dg.state_auxiliary.data,
            t,
            elems,
            grid.activedofs;
            ndrange = Np * nelem,
            dependencies = (event,),
        )
    end
    wait(device, event)
end

"""
    courant(local_courant::Function, dg::DGModel, m::BalanceLaw,
            state_conservative::MPIStateArray, direction=EveryDirection())
Returns the maximum of the evaluation of the function `local_courant`
pointwise throughout the domain.  The function `local_courant` is given an
approximation of the local node distance `Δx`.  The `direction` controls which
reference directions are considered when computing the minimum node distance
`Δx`.
An example `local_courant` function is
    function local_courant(m::AtmosModel, state_conservative::Vars, state_auxiliary::Vars,
                           diffusive::Vars, Δx)
      return Δt * cmax / Δx
    end
where `Δt` is the time step size and `cmax` is the maximum flow speed in the
model.
"""
function courant(
    local_courant::Function,
    dg::DGModel,
    m::BalanceLaw,
    state_conservative::MPIStateArray,
    Δt,
    simtime,
    direction = EveryDirection(),
)
    grid = dg.grid
    topology = grid.topology
    nrealelem = length(topology.realelems)

    if nrealelem > 0
        N = polynomialorder(grid)
        dim = dimensionality(grid)
        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq
        device = array_device(grid.vgeo)
        pointwise_courant = similar(grid.vgeo, Nq^dim, nrealelem)
        event = Event(device)
        event = Grids.kernel_min_neighbor_distance!(
            device,
            min(Nq * Nq * Nqk, 1024),
        )(
            Val(N),
            Val(dim),
            direction,
            pointwise_courant,
            grid.vgeo,
            topology.realelems;
            ndrange = (nrealelem * Nq * Nq * Nqk),
            dependencies = (event,),
        )
        event = kernel_local_courant!(device, min(Nq * Nq * Nqk, 1024))(
            m,
            Val(dim),
            Val(N),
            pointwise_courant,
            local_courant,
            state_conservative.data,
            dg.state_auxiliary.data,
            dg.state_gradient_flux.data,
            topology.realelems,
            Δt,
            simtime,
            direction;
            ndrange = nrealelem * Nq * Nq * Nqk,
            dependencies = (event,),
        )
        wait(device, event)
        rank_courant_max = maximum(pointwise_courant)
    else
        rank_courant_max = typemin(eltype(state_conservative))
    end

    MPI.Allreduce(rank_courant_max, max, topology.mpicomm)
end

function MPIStateArrays.MPIStateArray(dg::DGModel)
    balance_law = dg.balance_law
    grid = dg.grid

    state_conservative = create_conservative_state(balance_law, grid)

    return state_conservative
end
