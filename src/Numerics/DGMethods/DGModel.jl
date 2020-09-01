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
    fill_nan = false,
    state_auxiliary = create_state(
        balance_law,
        grid,
        Auxiliary(),
        fill_nan = fill_nan,
    ),
    state_gradient_flux = create_state(balance_law, grid, GradientFlux()),
    states_higher_order = (
        create_state(balance_law, grid, GradientLaplacian()),
        create_state(balance_law, grid, Hyperdiffusive()),
    ),
    direction = EveryDirection(),
    diffusion_direction = direction,
    modeldata = nothing,
)
    state_auxiliary =
        init_state(state_auxiliary, balance_law, grid, direction, Auxiliary())
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
    (dg::DGModel)(tendency, state_prognostic, nothing, t, α, β)

Computes the tendency terms compatible with `IncrementODEProblem`

    tendency .= α .* dQdt(state_prognostic, p, t) .+ β .* tendency

The 4-argument form will just compute

    tendency .= dQdt(state_prognostic, p, t)

"""
function (dg::DGModel)(tendency, state_prognostic, param, t; increment = false)
    # TODO deprecate increment argument
    dg(tendency, state_prognostic, param, t, true, increment)
end

function (dg::DGModel)(tendency, state_prognostic, _, t, α, β)

    balance_law = dg.balance_law
    device = array_device(state_prognostic)

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

    FT = eltype(state_prognostic)
    num_state_prognostic = number_states(balance_law, Prognostic())
    num_state_gradient_flux = number_states(balance_law, GradientFlux())
    nhyperviscstate = number_states(balance_law, Hyperdiffusive())
    num_state_tendency = size(tendency, 2)

    @assert num_state_prognostic ≤ num_state_tendency

    Np = dofs_per_element(grid)

    workgroups_volume = (Nq, Nq, Nqk)
    ndrange_volume = (nrealelem * Nq, Nq, Nqk)
    workgroups_surface = Nfp
    ndrange_interior_surface = Nfp * length(grid.interiorelems)
    ndrange_exterior_surface = Nfp * length(grid.exteriorelems)

    if num_state_prognostic < num_state_tendency && β != 1
        # if we don't operate on the full state, then we need to scale here instead of volume_tendency!
        tendency .*= β
        β = β != 0 # if β==0 then we can avoid the memory load in volume_tendency!
    end

    communicate =
        !(isstacked(topology) && typeof(dg.direction) <: VerticalDirection)

    update_auxiliary_state!(
        dg,
        balance_law,
        state_prognostic,
        t,
        dg.grid.topology.realelems,
    )

    if nhyperviscstate > 0
        hypervisc_indexmap = varsindices(
            vars_state(balance_law, Gradient(), FT),
            fieldnames(vars_state(balance_law, GradientLaplacian(), FT)),
        )
    else
        hypervisc_indexmap = nothing
    end

    exchange_state_prognostic = NoneEvent()
    exchange_state_gradient_flux = NoneEvent()
    exchange_Qhypervisc_grad = NoneEvent()
    exchange_Qhypervisc_div = NoneEvent()

    comp_stream = Event(device)

    ########################
    # Gradient Computation #
    ########################
    if communicate
        exchange_state_prognostic = MPIStateArrays.begin_ghost_exchange!(
            state_prognostic;
            dependencies = comp_stream,
        )
    end

    if num_state_gradient_flux > 0 || nhyperviscstate > 0

        comp_stream = volume_gradients!(device, (Nq, Nq))(
            balance_law,
            Val(dim),
            Val(N),
            dg.diffusion_direction,
            state_prognostic.data,
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
            state_prognostic.data,
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
            exchange_state_prognostic = MPIStateArrays.end_ghost_exchange!(
                state_prognostic;
                dependencies = exchange_state_prognostic,
            )

            # update_aux may start asynchronous work on the compute device and
            # we synchronize those here through a device event.
            wait(device, exchange_state_prognostic)
            update_auxiliary_state!(
                dg,
                balance_law,
                state_prognostic,
                t,
                dg.grid.topology.ghostelems,
            )
            exchange_state_prognostic = Event(device)
        end

        comp_stream = interface_gradients!(device, workgroups_surface)(
            balance_law,
            Val(dim),
            Val(N),
            dg.diffusion_direction,
            dg.numerical_flux_gradient,
            state_prognostic.data,
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
            dependencies = (comp_stream, exchange_state_prognostic),
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
                state_prognostic,
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
                state_prognostic.data,
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
                state_prognostic.data,
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
                state_prognostic.data,
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
        state_prognostic.data,
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
        state_prognostic.data,
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
                    state_prognostic,
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
            exchange_state_prognostic = MPIStateArrays.end_ghost_exchange!(
                state_prognostic;
                dependencies = exchange_state_prognostic,
            )

            # update_aux may start asynchronous work on the compute device and
            # we synchronize those here through a device event.
            wait(device, exchange_state_prognostic)
            update_auxiliary_state!(
                dg,
                balance_law,
                state_prognostic,
                t,
                dg.grid.topology.ghostelems,
            )
            exchange_state_prognostic = Event(device)
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
        state_prognostic.data,
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
            exchange_state_prognostic,
            exchange_state_gradient_flux,
            exchange_Qhypervisc_grad,
        ),
    )

    # The synchronization here through a device event prevents CuArray based and
    # other default stream kernels from launching before the work scheduled in
    # this function is finished.
    wait(device, comp_stream)
end

function init_ode_state(
    dg::DGModel,
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

function restart_ode_state(dg::DGModel, state_data; init_on_cpu = false)
    bl = dg.balance_law
    grid = dg.grid

    state = create_state(bl, grid, Prognostic())
    state .= state_data

    device = arraytype(dg.grid) <: Array ? CPU() : CUDADevice()
    event = Event(device)
    event = MPIStateArrays.begin_ghost_exchange!(state; dependencies = event)
    event = MPIStateArrays.end_ghost_exchange!(state; dependencies = event)
    wait(device, event)

    return state
end

function restart_auxiliary_state(bl, grid, aux_data, direction)
    state_auxiliary = create_state(bl, grid, Auxiliary())
    state_auxiliary =
        init_state(state_auxiliary, bl, grid, direction, Auxiliary())
    state_auxiliary .= aux_data
    return state_auxiliary
end

@deprecate nodal_init_state_auxiliary! init_state_auxiliary!

# By default, we call init_state_auxiliary!, given
# nodal_init_state_auxiliary!, defined for the
# particular balance_law:
function init_state_auxiliary!(
    balance_law::BalanceLaw,
    state_auxiliary,
    grid,
    direction,
)
    init_state_auxiliary!(
        balance_law,
        nodal_init_state_auxiliary!,
        state_auxiliary,
        grid,
        direction,
    )
end

# Should we provide a fallback implementation here?
# Maybe better to throw a method error?
function nodal_init_state_auxiliary!(m::BalanceLaw, aux, tmp, geom) end

function init_state_auxiliary!(
    balance_law,
    init_f!,
    state_auxiliary,
    grid,
    direction;
    state_temporary = nothing,
)
    topology = grid.topology
    dim = dimensionality(grid)
    Np = dofs_per_element(grid)
    polyorder = polynomialorder(grid)
    vgeo = grid.vgeo
    device = array_device(state_auxiliary)
    nrealelem = length(topology.realelems)

    event = Event(device)
    event = kernel_nodal_init_state_auxiliary!(
        device,
        min(Np, 1024),
        Np * nrealelem,
    )(
        balance_law,
        Val(dim),
        Val(polyorder),
        init_f!,
        state_auxiliary.data,
        isnothing(state_temporary) ? nothing : state_temporary.data,
        Val(isnothing(state_temporary) ? @vars() : vars(state_temporary)),
        vgeo,
        topology.realelems,
        dependencies = (event,),
    )

    event = MPIStateArrays.begin_ghost_exchange!(
        state_auxiliary;
        dependencies = event,
    )
    event = MPIStateArrays.end_ghost_exchange!(
        state_auxiliary;
        dependencies = event,
    )
    wait(device, event)
end

function update_auxiliary_state_gradient!(
    dg::DGModel,
    balance_law,
    state_prognostic,
    t,
    elems,
)
    return false
end

function indefinite_stack_integral!(
    dg::DGModel,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    state_auxiliary::MPIStateArray,
    t::Real,
    elems::UnitRange = dg.grid.topology.elems,
)

    device = array_device(state_prognostic)

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq

    FT = eltype(state_prognostic)

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
        state_prognostic.data,
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
    state_prognostic::MPIStateArray,
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
        state_prognostic.data,
        state_auxiliary.data,
        horzelems;
        ndrange = (length(horzelems) * Nq, Nqk),
        dependencies = (event,),
    )
    wait(device, event)
end

# By default, we call update_auxiliary_state!, given
# nodal_update_auxiliary_state!, defined for the
# particular balance_law:
function update_auxiliary_state!(
    dg::DGModel,
    balance_law::BalanceLaw,
    state_prognostic,
    t,
    elems,
    diffusive = false,
)
    update_auxiliary_state!(
        nodal_update_auxiliary_state!,
        dg,
        balance_law,
        state_prognostic,
        t,
        elems;
        diffusive = diffusive,
    )
end

# Should we provide a fallback implementation here?
# Maybe better to throw a method error?
function nodal_update_auxiliary_state!(balance_law, state, aux, t) end

function update_auxiliary_state!(
    f!,
    dg::DGModel,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    t::Real,
    elems::UnitRange = dg.grid.topology.realelems;
    diffusive = false,
)
    device = array_device(state_prognostic)

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    nelem = length(elems)

    Np = dofs_per_element(grid)

    knl_nodal_update_auxiliary_state! =
        kernel_nodal_update_auxiliary_state!(device, min(Np, 1024))
    ### update state_auxiliary variables
    event = Event(device)
    if diffusive
        event = knl_nodal_update_auxiliary_state!(
            m,
            Val(dim),
            Val(N),
            f!,
            state_prognostic.data,
            dg.state_auxiliary.data,
            dg.state_gradient_flux.data,
            t,
            elems,
            grid.activedofs;
            ndrange = Np * nelem,
            dependencies = (event,),
        )
    else
        event = knl_nodal_update_auxiliary_state!(
            m,
            Val(dim),
            Val(N),
            f!,
            state_prognostic.data,
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
            state_prognostic::MPIStateArray, direction=EveryDirection())
Returns the maximum of the evaluation of the function `local_courant`
pointwise throughout the domain.  The function `local_courant` is given an
approximation of the local node distance `Δx`.  The `direction` controls which
reference directions are considered when computing the minimum node distance
`Δx`.
An example `local_courant` function is
    function local_courant(m::AtmosModel, state_prognostic::Vars, state_auxiliary::Vars,
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
    state_prognostic::MPIStateArray,
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
            state_prognostic.data,
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
        rank_courant_max = typemin(eltype(state_prognostic))
    end

    MPI.Allreduce(rank_courant_max, max, topology.mpicomm)
end

function MPIStateArrays.MPIStateArray(dg::DGModel)
    balance_law = dg.balance_law
    grid = dg.grid

    state_prognostic = create_state(balance_law, grid, Prognostic())

    return state_prognostic
end

"""
    continuous_field_gradient!(::BalanceLaw, ∇state::MPIStateArray,
                               vars_out, state::MPIStateArray, vars_in, grid;
                               direction = EveryDirection())

Take the gradient of the variables `vars_in` located in the array `state`
and stores it in the variables `vars_out` of `∇state`. This function computes
element wise gradient without accounting for numerical fluxes and hence
its primary purpose is to take the gradient of continuous reference fields.

## Examples
```julia
FT = eltype(state_auxiliary)
grad_Φ = similar(state_auxiliary, vars=@vars(∇Φ::SVector{3, FT}))
continuous_field_gradient!(
    model,
    grad_Φ,
    ("∇Φ",),
    state_auxiliary,
    ("orientation.Φ",),
    grid,
)
```
"""
function continuous_field_gradient!(
    m::BalanceLaw,
    ∇state::MPIStateArray,
    vars_out,
    state::MPIStateArray,
    vars_in,
    grid,
    direction = EveryDirection(),
)
    topology = grid.topology
    nrealelem = length(topology.realelems)

    N = polynomialorder(grid)
    dim = dimensionality(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk
    device = array_device(state)

    I = varsindices(vars(state), vars_in)
    O = varsindices(vars(∇state), vars_out)

    event = Event(device)

    event = kernel_continuous_field_gradient!(device, (Nq, Nq, Nqk))(
        m,
        Val(dim),
        Val(N),
        direction,
        ∇state.data,
        state.data,
        grid.vgeo,
        grid.D,
        grid.ω,
        Val(I),
        Val(O),
        ndrange = (nrealelem * Nq, Nq, Nqk),
        dependencies = (event,),
    )
    wait(device, event)
end
