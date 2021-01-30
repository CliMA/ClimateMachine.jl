using .NumericalFluxes:
    CentralNumericalFluxHigherOrder, CentralNumericalFluxDivergence

"""
    SpaceDiscretization

Supertype for spatial discretizations.

Must have the following properties:

    - `grid`
    - `balance_law`
    - `state_auxiliary`
"""
abstract type SpaceDiscretization end

struct DGFVModel{BL, G, FVR, NFND, NFD, GNF, AS, DS, HDS, D, DD, MD, GF, TF} <:
       SpaceDiscretization
    balance_law::BL
    grid::G
    fv_reconstruction::FVR
    numerical_flux_first_order::NFND
    numerical_flux_second_order::NFD
    numerical_flux_gradient::GNF
    state_auxiliary::AS
    state_gradient_flux::DS
    states_higher_order::HDS
    direction::D
    diffusion_direction::DD
    modeldata::MD
    gradient_filter::GF
    tendency_filter::TF
end

function DGFVModel(
    balance_law,
    grid,
    fv_reconstruction,
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
    gradient_filter = nothing,
    tendency_filter = nothing,
)
    # Make sure we are FVM in the vertical
    @assert polynomialorders(grid)[end] == 0
    @assert isstacked(grid.topology)
    state_auxiliary =
        init_state(state_auxiliary, balance_law, grid, direction, Auxiliary())
    DGFVModel(
        balance_law,
        grid,
        fv_reconstruction,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        state_auxiliary,
        state_gradient_flux,
        states_higher_order,
        direction,
        diffusion_direction,
        modeldata,
        gradient_filter,
        tendency_filter,
    )
end

struct DGModel{BL, G, NFND, NFD, GNF, AS, DS, HDS, D, DD, MD, GF, TF} <:
       SpaceDiscretization
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
    gradient_filter::GF
    tendency_filter::TF
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
    gradient_filter = nothing,
    tendency_filter = nothing,
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
        gradient_filter,
        tendency_filter,
    )
end

# Include the remainder model for composing DG models and balance laws
include("remainder.jl")

basic_grid_info(spacedisc::SpaceDiscretization) =
    basic_grid_info(spacedisc.grid)
function basic_grid_info(grid)
    dim = dimensionality(grid)
    # Tuple of polynomial degrees (N₁, N₂, N₃)
    N = polynomialorders(grid)
    @assert dim == 2 || N[1] == N[2]

    # Number of quadrature point in each direction (Nq₁, Nq₂, Nq₃)
    Nq = N .+ 1

    # Number of quadrature point in the horizontal direction Nq₁ * Nq₂
    Nqh = Nq[1] * Nq[2]

    # Number of quadrature points in the vertical Nq₃
    Nqk = dim == 2 ? 1 : Nq[dim]

    Np = dofs_per_element(grid)
    Nfp_v, Nfp_h = div.(Np, (Nq[1], Nq[end]))

    topology_info = basic_topology_info(grid.topology)

    ninteriorelem = length(grid.interiorelems)
    nexteriorelem = length(grid.exteriorelems)

    nface = 2 * dim

    grid_info = (
        dim = dim,
        N = N,
        Nq = Nq,
        Nqh = Nqh,
        Nqk = Nqk,
        Nfp_v = Nfp_v,
        Nfp_h = Nfp_h,
        Np = Np,
        nface = nface,
        ninteriorelem = ninteriorelem,
        nexteriorelem = nexteriorelem,
    )

    return merge(grid_info, topology_info)
end

function basic_launch_info(spacedisc::SpaceDiscretization)
    device = array_device(spacedisc.state_auxiliary)
    grid_info = basic_grid_info(spacedisc.grid)
    return merge(grid_info, (device = device,))
end

function (spacedisc::SpaceDiscretization)(
    tendency,
    state_prognostic,
    param,
    t;
    increment = false,
)
    # TODO deprecate increment argument
    spacedisc(tendency, state_prognostic, param, t, true, increment)
end

"""
    (dgfvm::DGFVModel)(tendency, state_prognostic, _, t, α, β)

Uses spectral element discontinuous Galerkin in the horizontal and finite volume
in the vertical to compute the tendency.

Computes the tendency terms compatible with `IncrementODEProblem`

    tendency .= α .* dQdt(state_prognostic, p, t) .+ β .* tendency

The 4-argument form will just compute

    tendency .= dQdt(state_prognostic, p, t)
"""
function (dgfvm::DGFVModel)(tendency, state_prognostic, _, t, α, β)
    device = array_device(state_prognostic)

    FT = eltype(state_prognostic)
    num_state_prognostic = number_states(dgfvm.balance_law, Prognostic())
    num_state_gradient_flux = number_states(dgfvm.balance_law, GradientFlux())
    @assert 0 == number_states(dgfvm.balance_law, Hyperdiffusive())
    num_state_tendency = size(tendency, 2)

    if num_state_prognostic < num_state_tendency && β != 1
        # If we don't operate on the full state, then we need to scale here instead of volume_tendency!
        tendency .*= β
        β = β != 0 # if β==0 then we can avoid the memory load in volume_tendency!
    end

    communicate =
        !(
            isstacked(dgfvm.grid.topology) &&
            typeof(dgfvm.direction) <: VerticalDirection
        )

    update_auxiliary_state!(
        dgfvm,
        dgfvm.balance_law,
        state_prognostic,
        t,
        dgfvm.grid.topology.realelems,
    )

    exchange_state_gradient_flux = NoneEvent()
    exchange_state_prognostic = NoneEvent()

    comp_stream = Event(device)

    if communicate
        exchange_state_prognostic = MPIStateArrays.begin_ghost_exchange!(
            state_prognostic;
            dependencies = comp_stream,
        )
    end

    if num_state_gradient_flux > 0
        ########################
        # Gradient Computation #
        ########################

        comp_stream = launch_volume_gradients!(
            dgfvm,
            state_prognostic,
            t;
            dependencies = comp_stream,
        )

        comp_stream = launch_interface_gradients!(
            dgfvm,
            state_prognostic,
            t;
            surface = :interior,
            dependencies = comp_stream,
        )

        if communicate
            exchange_state_prognostic = MPIStateArrays.end_ghost_exchange!(
                state_prognostic;
                dependencies = exchange_state_prognostic,
            )

            # Update_aux may start asynchronous work on the compute device and
            # we synchronize those here through a device event.
            wait(device, exchange_state_prognostic)
            update_auxiliary_state!(
                dgfvm,
                dgfvm.balance_law,
                state_prognostic,
                t,
                dgfvm.grid.topology.ghostelems,
            )
            exchange_state_prognostic = Event(device)
        end

        comp_stream = launch_interface_gradients!(
            dgfvm,
            state_prognostic,
            t;
            surface = :exterior,
            dependencies = (comp_stream, exchange_state_prognostic),
        )

        if dgfvm.gradient_filter !== nothing
            comp_stream = Filters.apply_async!(
                dgfvm.state_gradient_flux,
                1:num_state_gradient_flux,
                dgfvm.grid,
                dgfvm.gradient_filter;
                dependencies = comp_stream,
            )
        end


        if communicate
            if num_state_gradient_flux > 0
                exchange_state_gradient_flux =
                    MPIStateArrays.begin_ghost_exchange!(
                        dgfvm.state_gradient_flux,
                        dependencies = comp_stream,
                    )
            end
        end

        if num_state_gradient_flux > 0
            # Update_aux_diffusive may start asynchronous work on the compute device
            # and we synchronize those here through a device event.
            wait(device, comp_stream)
            update_auxiliary_state_gradient!(
                dgfvm,
                dgfvm.balance_law,
                state_prognostic,
                t,
                dgfvm.grid.topology.realelems,
            )
            comp_stream = Event(device)
        end
    end

    ###################
    # RHS Computation #
    ###################
    comp_stream = launch_volume_tendency!(
        dgfvm,
        tendency,
        state_prognostic,
        t,
        α,
        β;
        dependencies = comp_stream,
    )

    comp_stream = launch_interface_tendency!(
        dgfvm,
        tendency,
        state_prognostic,
        t,
        α,
        β;
        surface = :interior,
        dependencies = comp_stream,
    )

    if communicate
        if num_state_gradient_flux > 0
            exchange_state_gradient_flux = MPIStateArrays.end_ghost_exchange!(
                dgfvm.state_gradient_flux;
                dependencies = exchange_state_gradient_flux,
            )

            # Update_aux_diffusive may start asynchronous work on the
            # compute device and we synchronize those here through a device
            # event.
            wait(device, exchange_state_gradient_flux)
            update_auxiliary_state_gradient!(
                dgfvm,
                dgfvm.balance_law,
                state_prognostic,
                t,
                dgfvm.grid.topology.ghostelems,
            )
            exchange_state_gradient_flux = Event(device)
        else
            exchange_state_prognostic = MPIStateArrays.end_ghost_exchange!(
                state_prognostic;
                dependencies = exchange_state_prognostic,
            )

            # Update_aux may start asynchronous work on the compute device and
            # we synchronize those here through a device event.
            wait(device, exchange_state_prognostic)
            update_auxiliary_state!(
                dgfvm,
                dgfvm.balance_law,
                state_prognostic,
                t,
                dgfvm.grid.topology.ghostelems,
            )
            exchange_state_prognostic = Event(device)
        end
    end

    comp_stream = launch_interface_tendency!(
        dgfvm,
        tendency,
        state_prognostic,
        t,
        α,
        β;
        surface = :exterior,
        dependencies = (
            comp_stream,
            exchange_state_prognostic,
            exchange_state_gradient_flux,
            # XXX: This is disabled until FVM with hyperdiffusion for DG is implemented: exchange_Qhypervisc_grad,
        ),
    )

    if dgfvm.tendency_filter !== nothing
        Filters.apply_async!(
            tendency,
            1:num_state_tendency,
            dgfvm.grid,
            dgfvm.tendency_filter;
            dependencies = comp_stream,
        )
    end

    # The synchronization here through a device event prevents CuArray based and
    # other default stream kernels from launching before the work scheduled in
    # this function is finished.
    wait(device, comp_stream)
end

"""
    (dg::DGModel)(tendency, state_prognostic, _, t, α, β)

Uses spectral element discontinuous Galerkin in all direction to compute the
tendency.

Computes the tendency terms compatible with `IncrementODEProblem`

    tendency .= α .* dQdt(state_prognostic, p, t) .+ β .* tendency

The 4-argument form will just compute

    tendency .= dQdt(state_prognostic, p, t)
"""
function (dg::DGModel)(tendency, state_prognostic, _, t, α, β)

    device = array_device(state_prognostic)
    Qhypervisc_grad, Qhypervisc_div = dg.states_higher_order

    FT = eltype(state_prognostic)
    num_state_prognostic = number_states(dg.balance_law, Prognostic())
    num_state_gradient_flux = number_states(dg.balance_law, GradientFlux())
    nhyperviscstate = number_states(dg.balance_law, Hyperdiffusive())
    num_state_tendency = size(tendency, 2)

    @assert num_state_prognostic ≤ num_state_tendency

    if num_state_prognostic < num_state_tendency && β != 1
        # If we don't operate on the full state, then we need to scale here instead of volume_tendency!
        tendency .*= β
        β = β != 0 # if β==0 then we can avoid the memory load in volume_tendency!
    end

    communicate =
        !(
            isstacked(dg.grid.topology) &&
            typeof(dg.direction) <: VerticalDirection
        )

    update_auxiliary_state!(
        dg,
        dg.balance_law,
        state_prognostic,
        t,
        dg.grid.topology.realelems,
    )

    exchange_state_prognostic = NoneEvent()
    exchange_state_gradient_flux = NoneEvent()
    exchange_Qhypervisc_grad = NoneEvent()
    exchange_Qhypervisc_div = NoneEvent()

    comp_stream = Event(device)

    if communicate
        exchange_state_prognostic = MPIStateArrays.begin_ghost_exchange!(
            state_prognostic;
            dependencies = comp_stream,
        )
    end

    if num_state_gradient_flux > 0 || nhyperviscstate > 0
        ########################
        # Gradient Computation #
        ########################

        comp_stream = launch_volume_gradients!(
            dg,
            state_prognostic,
            t;
            dependencies = comp_stream,
        )

        comp_stream = launch_interface_gradients!(
            dg,
            state_prognostic,
            t;
            surface = :interior,
            dependencies = comp_stream,
        )

        if communicate
            exchange_state_prognostic = MPIStateArrays.end_ghost_exchange!(
                state_prognostic;
                dependencies = exchange_state_prognostic,
            )

            # Update_aux may start asynchronous work on the compute device and
            # we synchronize those here through a device event.
            wait(device, exchange_state_prognostic)
            update_auxiliary_state!(
                dg,
                dg.balance_law,
                state_prognostic,
                t,
                dg.grid.topology.ghostelems,
            )
            exchange_state_prognostic = Event(device)
        end

        comp_stream = launch_interface_gradients!(
            dg,
            state_prognostic,
            t;
            surface = :exterior,
            dependencies = (comp_stream, exchange_state_prognostic),
        )

        if dg.gradient_filter !== nothing
            wait(device, comp_stream)
            Filters.apply!(
                dg.state_gradient_flux,
                1:num_state_gradient_flux,
                dg.grid,
                dg.gradient_filter,
            )
        end

        if communicate
            if num_state_gradient_flux > 0
                exchange_state_gradient_flux =
                    MPIStateArrays.begin_ghost_exchange!(
                        dg.state_gradient_flux,
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
            # Update_aux_diffusive may start asynchronous work on the compute device
            # and we synchronize those here through a device event.
            wait(device, comp_stream)
            update_auxiliary_state_gradient!(
                dg,
                dg.balance_law,
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

        comp_stream = launch_volume_divergence_of_gradients!(
            dg,
            state_prognostic,
            t;
            dependencies = comp_stream,
        )

        comp_stream = launch_interface_divergence_of_gradients!(
            dg,
            state_prognostic,
            t;
            surface = :interior,
            dependencies = comp_stream,
        )

        if communicate
            exchange_Qhypervisc_grad = MPIStateArrays.end_ghost_exchange!(
                Qhypervisc_grad,
                dependencies = exchange_Qhypervisc_grad,
            )
        end

        comp_stream = launch_interface_divergence_of_gradients!(
            dg,
            state_prognostic,
            t;
            surface = :exterior,
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

        comp_stream = launch_volume_gradients_of_laplacians!(
            dg,
            state_prognostic,
            t,
            dependencies = (comp_stream,),
        )

        comp_stream = launch_interface_gradients_of_laplacians!(
            dg,
            state_prognostic,
            t;
            surface = :interior,
            dependencies = (comp_stream,),
        )

        if communicate
            exchange_Qhypervisc_div = MPIStateArrays.end_ghost_exchange!(
                Qhypervisc_div,
                dependencies = exchange_Qhypervisc_div,
            )
        end

        comp_stream = launch_interface_gradients_of_laplacians!(
            dg,
            state_prognostic,
            t;
            surface = :exterior,
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
    comp_stream = launch_volume_tendency!(
        dg,
        tendency,
        state_prognostic,
        t,
        α,
        β;
        dependencies = (comp_stream,),
    )

    comp_stream = launch_interface_tendency!(
        dg,
        tendency,
        state_prognostic,
        t,
        α,
        β;
        surface = :interior,
        dependencies = (comp_stream,),
    )

    if communicate
        if num_state_gradient_flux > 0 || nhyperviscstate > 0
            if num_state_gradient_flux > 0
                exchange_state_gradient_flux =
                    MPIStateArrays.end_ghost_exchange!(
                        dg.state_gradient_flux;
                        dependencies = exchange_state_gradient_flux,
                    )

                # Update_aux_diffusive may start asynchronous work on the
                # compute device and we synchronize those here through a device
                # event.
                wait(device, exchange_state_gradient_flux)
                update_auxiliary_state_gradient!(
                    dg,
                    dg.balance_law,
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

            # Update_aux may start asynchronous work on the compute device and
            # we synchronize those here through a device event.
            wait(device, exchange_state_prognostic)
            update_auxiliary_state!(
                dg,
                dg.balance_law,
                state_prognostic,
                t,
                dg.grid.topology.ghostelems,
            )
            exchange_state_prognostic = Event(device)
        end
    end

    comp_stream = launch_interface_tendency!(
        dg,
        tendency,
        state_prognostic,
        t,
        α,
        β;
        surface = :exterior,
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
    if dg.tendency_filter !== nothing
        Filters.apply!(
            tendency,
            1:num_state_tendency,
            dg.grid,
            dg.tendency_filter,
        )
    end
end

function init_ode_state(
    spacedisc::SpaceDiscretization,
    args...;
    init_on_cpu = false,
    fill_nan = false,
)
    grid = spacedisc.grid
    balance_law = spacedisc.balance_law
    state_auxiliary = spacedisc.state_auxiliary

    device = arraytype(grid) <: Array ? CPU() : CUDADevice()

    state_prognostic =
        create_state(balance_law, grid, Prognostic(), fill_nan = fill_nan)

    topology = grid.topology
    Np = dofs_per_element(grid)

    dim = dimensionality(grid)
    N = polynomialorders(grid)
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
        state_auxiliary .= h_state_auxiliary
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

# TODO: this should really be a separate function
"""
    init_state_auxiliary!(
        bl::BalanceLaw,
        f!,
        statearray_auxiliary,
        grid,
        direction;
        state_temporary = nothing
    )

Apply `f!(bl, state_auxiliary, tmp, geom)` at each node, storing the result in
`statearray_auxiliary`, where `tmp` are the values at the corresponding node in
`state_temporary` and `geom` contains the geometry information.
"""
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
    N = polynomialorders(grid)
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
        Val(N),
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

update_auxiliary_state_gradient!(::SpaceDiscretization, _...) = false

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
    N = polynomialorders(grid)
    Nq = N .+ 1
    Nqj = dim == 2 ? 1 : Nq[2]

    FT = eltype(state_prognostic)

    # Compute integrals
    nelem = length(elems)
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)

    event = Event(device)
    event = kernel_indefinite_stack_integral!(device, (Nq[1], Nqj))(
        m,
        Val(dim),
        Val(N),
        Val(nvertelem),
        state_prognostic.data,
        state_auxiliary.data,
        grid.vgeo,
        # Only need the vertical Imat since this kernel is vertically oriented
        grid.Imat[dim],
        horzelems;
        ndrange = (length(horzelems) * Nq[1], Nqj),
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
    N = polynomialorders(grid)
    Nq = N .+ 1
    Nqj = dim == 2 ? 1 : Nq[2]

    FT = eltype(state_auxiliary)

    # Compute integrals
    nelem = length(elems)
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)

    event = Event(device)
    event = kernel_reverse_indefinite_stack_integral!(device, (Nq[1], Nqj))(
        m,
        Val(dim),
        Val(N),
        Val(nvertelem),
        state_prognostic.data,
        state_auxiliary.data,
        horzelems;
        ndrange = (length(horzelems) * Nq[1], Nqj),
        dependencies = (event,),
    )
    wait(device, event)
end

# By default, we call update_auxiliary_state!, given
# nodal_update_auxiliary_state!, defined for the
# particular balance_law:

# TODO: this should really be a separate function
function update_auxiliary_state!(
    f!,
    spacedisc::SpaceDiscretization,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    t::Real,
    elems::UnitRange = spacedisc.grid.topology.realelems;
    diffusive = false,
)
    device = array_device(state_prognostic)

    grid = spacedisc.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorders(grid)
    nelem = length(elems)

    Np = dofs_per_element(grid)

    knl_nodal_update_auxiliary_state! =
        kernel_nodal_update_auxiliary_state!(device, min(Np, 1024))
    ### Update state_auxiliary variables
    event = Event(device)
    if diffusive
        event = knl_nodal_update_auxiliary_state!(
            m,
            Val(dim),
            Val(N),
            f!,
            state_prognostic.data,
            spacedisc.state_auxiliary.data,
            spacedisc.state_gradient_flux.data,
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
            spacedisc.state_auxiliary.data,
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
    dg::SpaceDiscretization,
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
        N = polynomialorders(grid)
        dim = dimensionality(grid)
        Nq = N .+ 1
        Nqk = dim == 2 ? 1 : Nq[dim]
        device = array_device(grid.vgeo)
        pointwise_courant = similar(grid.vgeo, prod(Nq), nrealelem)
        event = Event(device)
        event = Grids.kernel_min_neighbor_distance!(
            device,
            min(Nq[1] * Nq[2] * Nqk, 1024),
        )(
            Val(N),
            Val(dim),
            direction,
            pointwise_courant,
            grid.vgeo,
            topology.realelems;
            ndrange = (nrealelem * Nq[1] * Nq[2] * Nqk),
            dependencies = (event,),
        )
        event = kernel_local_courant!(device, min(Nq[1] * Nq[2] * Nqk, 1024))(
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
            ndrange = (nrealelem * Nq[1] * Nq[2] * Nqk),
            dependencies = (event,),
        )
        wait(device, event)

        rank_courant_max = maximum(pointwise_courant)
    else
        rank_courant_max = typemin(eltype(state_prognostic))
    end

    MPI.Allreduce(rank_courant_max, max, topology.mpicomm)
end

function MPIStateArrays.MPIStateArray(dg::SpaceDiscretization)
    balance_law = dg.balance_law
    grid = dg.grid

    state_prognostic = create_state(balance_law, grid, Prognostic())

    return state_prognostic
end

"""
    auxiliary_field_gradient!(::BalanceLaw, ∇state::MPIStateArray,
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
auxiliary_field_gradient!(
    model,
    grad_Φ,
    ("∇Φ",),
    state_auxiliary,
    ("orientation.Φ",),
    grid,
)
```
"""
function auxiliary_field_gradient!(
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
    N = polynomialorders(grid)
    dim = dimensionality(grid)
    Nq = N .+ 1
    Nqk = dim == 2 ? 1 : Nq[dim]
    device = array_device(state)

    I = varsindices(vars(state), vars_in)
    O = varsindices(vars(∇state), vars_out)

    event = Event(device)

    if direction isa EveryDirection || direction isa HorizontalDirection
        # We assume N₁ = N₂, so the same polyorder, quadrature weights,
        # and differentiation operators are used
        horizontal_polyorder = N[1]
        horizontal_D = grid.D[1]
        horizontal_ω = grid.ω[1]
        event = dgsem_auxiliary_field_gradient!(device, (Nq[1], Nq[2]))(
            m,
            Val(dim),
            Val(N),
            HorizontalDirection(),
            ∇state.data,
            state.data,
            grid.vgeo,
            horizontal_D,
            horizontal_ω,
            Val(I),
            Val(O),
            false,
            ndrange = (nrealelem * Nq[1], Nq[2]),
            dependencies = (event,),
        )
    end

    if direction isa EveryDirection || direction isa VerticalDirection
        vertical_polyorder = N[dim]
        if vertical_polyorder > 0
            vertical_D = grid.D[dim]
            vertical_ω = grid.ω[dim]
            event = dgsem_auxiliary_field_gradient!(device, (Nq[1], Nq[2]))(
                m,
                Val(dim),
                Val(N),
                VerticalDirection(),
                ∇state.data,
                state.data,
                grid.vgeo,
                vertical_D,
                vertical_ω,
                Val(I),
                Val(O),
                # If we are computing in every direction, we need to
                # increment after we compute the horizontal values
                (direction isa EveryDirection);
                ndrange = (nrealelem * Nq[1], Nq[2]),
                dependencies = (event,),
            )
        else
            info = basic_grid_info(grid)
            event = vert_fvm_auxiliary_field_gradient!(device, info.Nfp_h)(
                m,
                Val(info),
                ∇state.data,
                state.data,
                grid.vgeo,
                grid.sgeo,
                grid.vmap⁻,
                grid.vmap⁺,
                grid.elemtobndy,
                Val(I),
                Val(O),
                # If we are computing in every direction, we need to
                # increment after we compute the horizontal values
                (direction isa EveryDirection);
                ndrange = (nrealelem * info.Nfp_h),
                dependencies = (event,),
            )
        end
    end
    wait(device, event)
end

function hyperdiff_indexmap(balance_law, ::Type{FT}) where {FT}
    ns_hyperdiff = number_states(balance_law, Hyperdiffusive())
    if ns_hyperdiff > 0
        return varsindices(
            vars_state(balance_law, Gradient(), FT),
            fieldnames(vars_state(balance_law, GradientLaplacian(), FT)),
        )
    else
        return nothing
    end
end

"""
    launch_volume_gradients!(spacedisc, state_prognostic, t; dependencies)

Launches horizontal and vertical kernels for computing the volume gradients.
"""
function launch_volume_gradients!(spacedisc, state_prognostic, t; dependencies)
    FT = eltype(state_prognostic)
    # XXX: This is until FVM with hyperdiffusion for DG is implemented
    if spacedisc isa DGFVModel
        @assert 0 == number_states(spacedisc.balance_law, Hyperdiffusive())
        Qhypervisc_grad_data = nothing
    elseif spacedisc isa DGModel
        Qhypervisc_grad_data = spacedisc.states_higher_order[1].data
    end

    # Workgroup is determined by the number of quadrature points
    # in the horizontal direction. For each horizontal quadrature
    # point, we operate on a stack of quadrature in the vertical
    # direction. (Iteration space is in the horizontal)
    info = basic_launch_info(spacedisc)

    # We assume (in 3-D) that both x and y directions
    # are discretized using the same polynomial order, Nq[1] == Nq[2].
    # In 2-D, the workgroup spans the entire set of quadrature points:
    # Nq[1] * Nq[2]
    workgroup = (info.Nq[1], info.Nq[2])
    ndrange = (info.Nq[1] * info.nrealelem, info.Nq[2])
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `spacedisc.diffusion_direction`
    if spacedisc.diffusion_direction isa EveryDirection ||
       spacedisc.diffusion_direction isa HorizontalDirection

        # We assume N₁ = N₂, so the same polyorder, quadrature weights,
        # and differentiation operators are used
        horizontal_polyorder = info.N[1]
        horizontal_D = spacedisc.grid.D[1]
        comp_stream = volume_gradients!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            HorizontalDirection(),
            state_prognostic.data,
            spacedisc.state_gradient_flux.data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            t,
            horizontal_D,
            Val(hyperdiff_indexmap(spacedisc.balance_law, FT)),
            spacedisc.grid.topology.realelems,
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # Now we call the kernel corresponding to the vertical direction
    if spacedisc isa DGModel && (
        spacedisc.diffusion_direction isa EveryDirection ||
        spacedisc.diffusion_direction isa VerticalDirection
    )

        # Vertical polynomial degree and differentiation matrix
        vertical_polyorder = info.N[info.dim]
        vertical_D = spacedisc.grid.D[info.dim]
        comp_stream = volume_gradients!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            VerticalDirection(),
            state_prognostic.data,
            spacedisc.state_gradient_flux.data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            t,
            vertical_D,
            Val(hyperdiff_indexmap(spacedisc.balance_law, FT)),
            spacedisc.grid.topology.realelems,
            # If we are computing the volume gradient in every direction, we
            # need to increment into the appropriate fields _after_ the
            # horizontal computation.
            !(spacedisc.diffusion_direction isa VerticalDirection),
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end
    return comp_stream
end

"""
    launch_interface_gradients!(spacedisc, state_prognostic, t; surface::Symbol, dependencies)

Launches horizontal and vertical kernels for computing the interface gradients.
The argument `surface` is either `:interior` or `:exterior`, which denotes whether
we are computing interface gradients on boundaries which are interior (exterior resp.)
to the _parallel_ boundary.
"""
function launch_interface_gradients!(
    spacedisc,
    state_prognostic,
    t;
    surface::Symbol,
    dependencies,
)
    @assert surface === :interior || surface === :exterior
    # XXX: This is until FVM with DG hyperdiffusion is implemented
    if spacedisc isa DGFVModel
        @assert 0 == number_states(spacedisc.balance_law, Hyperdiffusive())
        Qhypervisc_grad_data = nothing
    elseif spacedisc isa DGModel
        Qhypervisc_grad_data = spacedisc.states_higher_order[1].data
    end

    FT = eltype(state_prognostic)

    info = basic_launch_info(spacedisc)
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `spacedisc.diffusion_direction`
    if spacedisc.diffusion_direction isa EveryDirection ||
       spacedisc.diffusion_direction isa HorizontalDirection

        workgroup = info.Nfp_v
        if surface === :interior
            elems = spacedisc.grid.interiorelems
            ndrange = workgroup * info.ninteriorelem
        else
            elems = spacedisc.grid.exteriorelems
            ndrange = workgroup * info.nexteriorelem
        end

        # Hoirzontal polynomial order (assumes same for both horizontal directions)
        horizontal_polyorder = info.N[1]

        comp_stream = dgsem_interface_gradients!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            HorizontalDirection(),
            spacedisc.numerical_flux_gradient,
            state_prognostic.data,
            spacedisc.state_gradient_flux.data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            spacedisc.grid.sgeo,
            t,
            spacedisc.grid.vmap⁻,
            spacedisc.grid.vmap⁺,
            spacedisc.grid.elemtobndy,
            Val(hyperdiff_indexmap(spacedisc.balance_law, FT)),
            elems;
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # Vertical interface kernel call
    if spacedisc.diffusion_direction isa EveryDirection ||
       spacedisc.diffusion_direction isa VerticalDirection

        workgroup = info.Nfp_h
        if surface === :interior
            elems = spacedisc.grid.interiorelems
            ndrange = workgroup * info.ninteriorelem
        else
            elems = spacedisc.grid.exteriorelems
            ndrange = workgroup * info.nexteriorelem
        end

        # Vertical polynomial degree
        vertical_polyorder = info.N[info.dim]

        if spacedisc isa DGModel
            comp_stream = dgsem_interface_gradients!(info.device, workgroup)(
                spacedisc.balance_law,
                Val(info),
                VerticalDirection(),
                spacedisc.numerical_flux_gradient,
                state_prognostic.data,
                spacedisc.state_gradient_flux.data,
                Qhypervisc_grad_data,
                spacedisc.state_auxiliary.data,
                spacedisc.grid.vgeo,
                spacedisc.grid.sgeo,
                t,
                spacedisc.grid.vmap⁻,
                spacedisc.grid.vmap⁺,
                spacedisc.grid.elemtobndy,
                Val(hyperdiff_indexmap(spacedisc.balance_law, FT)),
                elems;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
        elseif spacedisc isa DGFVModel
            # Make sure FVM in the vertical
            @assert info.N[info.dim] == 0

            # The FVM will only work on stacked grids!
            @assert isstacked(spacedisc.grid.topology)
            nvertelem = spacedisc.grid.topology.stacksize
            periodicstack = spacedisc.grid.topology.periodicstack

            # 1 thread per degree freedom per element
            comp_stream = vert_fvm_interface_gradients!(info.device, workgroup)(
                spacedisc.balance_law,
                Val(info),
                Val(nvertelem),
                Val(periodicstack),
                VerticalDirection(),
                state_prognostic.data,
                spacedisc.state_gradient_flux.data,
                spacedisc.state_auxiliary.data,
                spacedisc.grid.vgeo,
                spacedisc.grid.sgeo,
                t,
                spacedisc.grid.elemtobndy,
                elems,
                # If we are computing in every direction, we need to
                # increment after we compute the horizontal values
                spacedisc.direction isa EveryDirection,
                ndrange = ndrange,
                dependencies = comp_stream,
            )
        else
            error("unknown spatial discretization: $(typeof(spacedisc))")
        end
    end
    return comp_stream
end

"""
    launch_volume_divergence_of_gradients!(dg, state_prognostic, t; dependencies)

Launches horizontal and vertical volume kernels for computing the divergence of gradients.
"""
function launch_volume_divergence_of_gradients!(
    dg,
    state_prognostic,
    t;
    dependencies,
)
    Qhypervisc_grad, Qhypervisc_div = dg.states_higher_order

    info = basic_launch_info(dg)
    workgroup = (info.Nq[1], info.Nq[2])
    ndrange = (info.nrealelem * info.Nq[1], info.Nq[2])
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa HorizontalDirection

        # Horizontal polynomial order and differentiation matrix
        horizontal_polyorder = info.N[1]
        horizontal_D = dg.grid.D[1]

        comp_stream = volume_divergence_of_gradients!(info.device, workgroup)(
            dg.balance_law,
            Val(info),
            HorizontalDirection(),
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            dg.grid.vgeo,
            horizontal_D,
            dg.grid.topology.realelems;
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # And now the vertical kernel call
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa VerticalDirection

        # Vertical polynomial order and differentiation matrix
        vertical_polyorder = info.N[info.dim]
        vertical_D = dg.grid.D[info.dim]

        comp_stream = volume_divergence_of_gradients!(info.device, workgroup)(
            dg.balance_law,
            Val(info),
            VerticalDirection(),
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            dg.grid.vgeo,
            vertical_D,
            dg.grid.topology.realelems,
            # If we are computing the volume gradient in every direction, we
            # need to increment into the appropriate fields _after_ the
            # horizontal computation.
            !(dg.diffusion_direction isa VerticalDirection);
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end
    return comp_stream
end

"""
    launch_interface_divergence_of_gradients!(dg, state_prognostic, t; surface::Symbol, dependencies)

Launches horizontal and vertical interface kernels for computing the divergence of gradients.
The argument `surface` is either `:interior` or `:exterior`, which denotes whether
we are computing values on boundaries which are interior (exterior resp.)
to the _parallel_ boundary.
"""
function launch_interface_divergence_of_gradients!(
    dg,
    state_prognostic,
    t;
    surface::Symbol,
    dependencies,
)
    @assert surface === :interior || surface === :exterior
    Qhypervisc_grad, Qhypervisc_div = dg.states_higher_order

    info = basic_launch_info(dg)
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa HorizontalDirection

        workgroup = info.Nfp_v
        if surface === :interior
            elems = dg.grid.interiorelems
            ndrange = info.Nfp_v * info.ninteriorelem
        else
            elems = dg.grid.exteriorelems
            ndrange = info.Nfp_v * info.nexteriorelem
        end

        # Hoirzontal polynomial order (assumes same for both horizontal directions)
        horizontal_polyorder = info.N[1]

        comp_stream =
            interface_divergence_of_gradients!(info.device, workgroup)(
                dg.balance_law,
                Val(info),
                HorizontalDirection(),
                CentralNumericalFluxDivergence(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                dg.grid.vgeo,
                dg.grid.sgeo,
                dg.grid.vmap⁻,
                dg.grid.vmap⁺,
                dg.grid.elemtobndy,
                elems;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
    end

    # Vertical kernel call
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa VerticalDirection

        workgroup = info.Nfp_h
        if surface === :interior
            elems = dg.grid.interiorelems
            ndrange = info.Nfp_h * info.ninteriorelem
        else
            elems = dg.grid.exteriorelems
            ndrange = info.Nfp_h * info.nexteriorelem
        end

        # Vertical polynomial degree
        vertical_polyorder = info.N[info.dim]

        comp_stream =
            interface_divergence_of_gradients!(info.device, workgroup)(
                dg.balance_law,
                Val(info),
                VerticalDirection(),
                CentralNumericalFluxDivergence(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                dg.grid.vgeo,
                dg.grid.sgeo,
                dg.grid.vmap⁻,
                dg.grid.vmap⁺,
                dg.grid.elemtobndy,
                elems;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
    end

    return comp_stream
end

"""
    launch_volume_gradients_of_laplacians!(dg, state_prognostic, t; dependencies)

Launches horizontal and vertical volume kernels for computing the DG gradient of
a second-order DG gradient (Laplacian).
"""
function launch_volume_gradients_of_laplacians!(
    dg,
    state_prognostic,
    t;
    dependencies,
)
    Qhypervisc_grad, Qhypervisc_div = dg.states_higher_order

    info = basic_launch_info(dg)
    workgroup = (info.Nq[1], info.Nq[2])
    ndrange = (info.nrealelem * info.Nq[1], info.Nq[2])
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa HorizontalDirection

        # Horizontal polynomial degree
        horizontal_polyorder = info.N[1]
        # Horizontal quadrature weights and differentiation matrix
        horizontal_ω = dg.grid.ω[1]
        horizontal_D = dg.grid.D[1]

        comp_stream = volume_gradients_of_laplacians!(info.device, workgroup)(
            dg.balance_law,
            Val(info),
            HorizontalDirection(),
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            state_prognostic.data,
            dg.state_auxiliary.data,
            dg.grid.vgeo,
            horizontal_ω,
            horizontal_D,
            dg.grid.topology.realelems,
            t;
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # Vertical kernel call
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa VerticalDirection

        # Vertical polynomial degree
        vertical_polyorder = info.N[info.dim]
        # Vertical quadrature weights and differentiation matrix
        vertical_ω = dg.grid.ω[info.dim]
        vertical_D = dg.grid.D[info.dim]

        comp_stream = volume_gradients_of_laplacians!(info.device, workgroup)(
            dg.balance_law,
            Val(info),
            VerticalDirection(),
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            state_prognostic.data,
            dg.state_auxiliary.data,
            dg.grid.vgeo,
            vertical_ω,
            vertical_D,
            dg.grid.topology.realelems,
            t,
            # If we are computing the volume gradient in every direction, we
            # need to increment into the appropriate fields _after_ the
            # horizontal computation.
            !(dg.diffusion_direction isa VerticalDirection);
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    return comp_stream
end

"""
    launch_interface_gradients_of_laplacians!(dg, state_prognostic, t; surface::Symbol, dependencies)

Launches horizontal and vertical interface kernels for computing the gradients of Laplacians
(second-order gradients). The argument `surface` is either `:interior` or `:exterior`,
which denotes whether we are computing values on boundaries which are interior (exterior resp.)
to the _parallel_ boundary.
"""
function launch_interface_gradients_of_laplacians!(
    dg,
    state_prognostic,
    t;
    surface::Symbol,
    dependencies,
)
    @assert surface === :interior || surface === :exterior
    Qhypervisc_grad, Qhypervisc_div = dg.states_higher_order
    comp_stream = dependencies
    info = basic_launch_info(dg)

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa HorizontalDirection

        workgroup = info.Nfp_v
        if surface === :interior
            elems = dg.grid.interiorelems
            ndrange = info.Nfp_v * info.ninteriorelem
        else
            elems = dg.grid.exteriorelems
            ndrange = info.Nfp_v * info.nexteriorelem
        end

        # Hoirzontal polynomial order (assumes same for both horizontal directions)
        horizontal_polyorder = info.N[1]

        comp_stream =
            interface_gradients_of_laplacians!(info.device, workgroup)(
                dg.balance_law,
                Val(info),
                HorizontalDirection(),
                CentralNumericalFluxHigherOrder(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                state_prognostic.data,
                dg.state_auxiliary.data,
                dg.grid.vgeo,
                dg.grid.sgeo,
                dg.grid.vmap⁻,
                dg.grid.vmap⁺,
                dg.grid.elemtobndy,
                elems,
                t;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
    end

    # Vertical kernel call
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa VerticalDirection

        workgroup = info.Nfp_h
        if surface === :interior
            elems = dg.grid.interiorelems
            ndrange = info.Nfp_h * info.ninteriorelem
        else
            elems = dg.grid.exteriorelems
            ndrange = info.Nfp_h * info.nexteriorelem
        end

        # Vertical polynomial degree
        vertical_polyorder = info.N[info.dim]

        comp_stream =
            interface_gradients_of_laplacians!(info.device, workgroup)(
                dg.balance_law,
                Val(info),
                VerticalDirection(),
                CentralNumericalFluxHigherOrder(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                state_prognostic.data,
                dg.state_auxiliary.data,
                dg.grid.vgeo,
                dg.grid.sgeo,
                dg.grid.vmap⁻,
                dg.grid.vmap⁺,
                dg.grid.elemtobndy,
                elems,
                t;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
    end

    return comp_stream
end

"""
    launch_volume_tendency!(spacedisc, state_prognostic, t; dependencies)

Launches horizontal and vertical volume kernels for computing tendencies (sources, sinks, etc).
"""
function launch_volume_tendency!(
    spacedisc,
    tendency,
    state_prognostic,
    t,
    α,
    β;
    dependencies,
)
    # XXX: This is until FVM with hyperdiffusion is implemented
    if spacedisc isa DGFVModel
        @assert 0 == number_states(spacedisc.balance_law, Hyperdiffusive())
        Qhypervisc_grad_data = nothing
    elseif spacedisc isa DGModel
        Qhypervisc_grad_data = spacedisc.states_higher_order[1].data
    end
    grad_flux_data = spacedisc.state_gradient_flux.data

    # Workgroup is determined by the number of quadrature points
    # in the horizontal direction. For each horizontal quadrature
    # point, we operate on a stack of quadrature in the vertical
    # direction. (Iteration space is in the horizontal)
    info = basic_launch_info(spacedisc)

    # We assume (in 3-D) that both x and y directions
    # are discretized using the same polynomial order, Nq[1] == Nq[2].
    # In 2-D, the workgroup spans the entire set of quadrature points:
    # Nq[1] * Nq[2]
    workgroup = (info.Nq[1], info.Nq[2])
    ndrange = (info.Nq[1] * info.nrealelem, info.Nq[2])
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction
    # `spacedisc.diffusion_direction`
    if spacedisc.direction isa EveryDirection ||
       spacedisc.direction isa HorizontalDirection

        # Horizontal polynomial degree
        horizontal_polyorder = info.N[1]
        # Horizontal quadrature weights and differentiation matrix
        horizontal_ω = spacedisc.grid.ω[1]
        horizontal_D = spacedisc.grid.D[1]

        comp_stream = volume_tendency!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            spacedisc.direction,
            HorizontalDirection(),
            tendency.data,
            state_prognostic.data,
            grad_flux_data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            t,
            horizontal_ω,
            horizontal_D,
            spacedisc.grid.topology.realelems,
            α,
            β,
            # If the model direction is horizontal or FV in the vertical,
            # we want to be sure to add sources
            spacedisc.direction isa HorizontalDirection ||
            spacedisc isa DGFVModel,
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # Vertical kernel
    if spacedisc isa DGModel && (
        spacedisc.direction isa EveryDirection ||
        spacedisc.direction isa VerticalDirection
    )

        # Vertical polynomial degree
        vertical_polyorder = info.N[info.dim]
        # Vertical quadrature weights and differentiation matrix
        vertical_ω = spacedisc.grid.ω[info.dim]
        vertical_D = spacedisc.grid.D[info.dim]

        comp_stream = volume_tendency!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            spacedisc.direction,
            VerticalDirection(),
            tendency.data,
            state_prognostic.data,
            grad_flux_data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            t,
            vertical_ω,
            vertical_D,
            spacedisc.grid.topology.realelems,
            α,
            # If we are computing the volume gradient in every direction, we
            # need to increment into the appropriate fields _after_ the
            # horizontal computation.
            spacedisc.direction isa EveryDirection ? true : β,
            # Boolean to add source. In the case of EveryDirection, we always add the sources
            # in the vertical kernel. Here, we make the assumption that we're either computing
            # in every direction, or _just_ the vertical direction.
            true;
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    return comp_stream
end

"""
    launch_interface_tendency!(spacedisc, state_prognostic, t; surface::Symbol, dependencies)

Launches horizontal and vertical interface kernels for computing tendencies (sources, sinks, etc).
The argument `surface` is either `:interior` or `:exterior`, which denotes whether we are computing
values on boundaries which are interior (exterior resp.) to the _parallel_ boundary.
"""
function launch_interface_tendency!(
    spacedisc,
    tendency,
    state_prognostic,
    t,
    α,
    β;
    surface::Symbol,
    dependencies,
)
    @assert surface === :interior || surface === :exterior
    # XXX: This is until FVM with diffusion is implemented
    if spacedisc isa DGFVModel
        @assert 0 == number_states(spacedisc.balance_law, Hyperdiffusive())
        Qhypervisc_grad_data = nothing
    elseif spacedisc isa DGModel
        Qhypervisc_grad_data = spacedisc.states_higher_order[1].data
    end
    grad_flux_data = spacedisc.state_gradient_flux.data
    numerical_flux_second_order = spacedisc.numerical_flux_second_order

    info = basic_launch_info(spacedisc)
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction
    # `spacedisc.diffusion_direction`
    if spacedisc.direction isa EveryDirection ||
       spacedisc.direction isa HorizontalDirection

        workgroup = info.Nfp_v
        if surface === :interior
            elems = spacedisc.grid.interiorelems
            ndrange = workgroup * info.ninteriorelem
        else
            elems = spacedisc.grid.exteriorelems
            ndrange = workgroup * info.nexteriorelem
        end

        # Hoirzontal polynomial order (assumes same for both horizontal
        # directions)
        horizontal_polyorder = info.N[1]

        comp_stream = dgsem_interface_tendency!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            HorizontalDirection(),
            spacedisc.numerical_flux_first_order,
            numerical_flux_second_order,
            tendency.data,
            state_prognostic.data,
            grad_flux_data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            spacedisc.grid.sgeo,
            t,
            spacedisc.grid.vmap⁻,
            spacedisc.grid.vmap⁺,
            spacedisc.grid.elemtobndy,
            elems,
            α;
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # Vertical kernel call
    if spacedisc.direction isa EveryDirection ||
       spacedisc.direction isa VerticalDirection
        elems =
            surface === :interior ? elems = spacedisc.grid.interiorelems :
            spacedisc.grid.exteriorelems

        if spacedisc isa DGModel
            workgroup = info.Nfp_h
            ndrange = workgroup * length(elems)

            # Vertical polynomial degree
            vertical_polyorder = info.N[info.dim]

            comp_stream = dgsem_interface_tendency!(info.device, workgroup)(
                spacedisc.balance_law,
                Val(info),
                VerticalDirection(),
                spacedisc.numerical_flux_first_order,
                numerical_flux_second_order,
                tendency.data,
                state_prognostic.data,
                grad_flux_data,
                Qhypervisc_grad_data,
                spacedisc.state_auxiliary.data,
                spacedisc.grid.vgeo,
                spacedisc.grid.sgeo,
                t,
                spacedisc.grid.vmap⁻,
                spacedisc.grid.vmap⁺,
                spacedisc.grid.elemtobndy,
                elems,
                α;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
        elseif spacedisc isa DGFVModel
            # Make sure FVM in the vertical
            @assert info.N[info.dim] == 0

            # The FVM will only work on stacked grids!
            @assert isstacked(spacedisc.grid.topology)

            # Figute out the stacking of the mesh
            nvertelem = spacedisc.grid.topology.stacksize
            nhorzelem = div(length(elems), nvertelem)
            periodicstack = spacedisc.grid.topology.periodicstack

            # 2-D workgroup
            workgroup = info.Nfp_h
            ndrange = workgroup * nhorzelem

            # XXX: This will need to be updated to diffusion
            comp_stream = vert_fvm_interface_tendency!(info.device, workgroup)(
                spacedisc.balance_law,
                Val(info),
                Val(nvertelem),
                Val(periodicstack),
                VerticalDirection(),
                spacedisc.fv_reconstruction,
                spacedisc.numerical_flux_first_order,
                numerical_flux_second_order,
                tendency.data,
                state_prognostic.data,
                grad_flux_data,
                spacedisc.state_auxiliary.data,
                spacedisc.grid.vgeo,
                spacedisc.grid.sgeo,
                t,
                spacedisc.grid.elemtobndy,
                elems,
                α,
                β,
                # If we are computing in every direction, we need to
                # increment after we compute the horizontal values
                spacedisc.direction isa EveryDirection,
                # If we are computing in vertical direction, we need to
                # add sources here
                spacedisc.direction isa VerticalDirection,
                ndrange = ndrange,
                dependencies = comp_stream,
            )
        else
            error("unknown spatial discretization: $(typeof(spacedisc))")
        end
    end

    return comp_stream
end

function fvm_balance!(
    balance_func,
    m::BalanceLaw,
    state_auxiliary::MPIStateArray,
    grid,
)
    device = array_device(state_auxiliary)


    dim = dimensionality(grid)
    N = polynomialorders(grid)
    Nq = N .+ 1
    Nqj = dim == 2 ? 1 : Nq[2]

    topology = grid.topology
    elems = topology.elems
    nelem = length(elems)
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)

    event = Event(device)
    event = kernel_fvm_balance!(device, (Nq[1], Nqj))(
        balance_func,
        m,
        Val(nvertelem),
        state_auxiliary.data,
        grid.vgeo,
        horzelems;
        ndrange = (length(horzelems) * Nq[1], Nqj),
        dependencies = (event,),
    )
    wait(device, event)
end
