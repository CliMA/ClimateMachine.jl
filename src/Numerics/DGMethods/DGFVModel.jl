include("DGFVModel_kernels.jl")
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
    check_for_crashes::Bool
end

function DGFVModel(
    balance_law,
    grid,
    fv_reconstruction,
    numerical_flux_first_order,
    numerical_flux_second_order,
    numerical_flux_gradient;
    fill_nan = false,
    check_for_crashes = false,
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
        check_for_crashes,
    )
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
                check_for_crashes = dgfvm.check_for_crashes,
            )

            # Update_aux may start asynchronous work on the compute device and
            # we synchronize those here through a device event.
            checked_wait(
                device,
                exchange_state_prognostic,
                nothing,
                dgfvm.check_for_crashes,
            )
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
            checked_wait(device, comp_stream, nothing, dgfvm.check_for_crashes)
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
                check_for_crashes = dgfvm.check_for_crashes,
            )

            # Update_aux_diffusive may start asynchronous work on the
            # compute device and we synchronize those here through a device
            # event.
            checked_wait(
                device,
                exchange_state_gradient_flux,
                nothing,
                dgfvm.check_for_crashes,
            )
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
                check_for_crashes = dgfvm.check_for_crashes,
            )

            # Update_aux may start asynchronous work on the compute device and
            # we synchronize those here through a device event.
            checked_wait(
                device,
                exchange_state_prognostic,
                nothing,
                dgfvm.check_for_crashes,
            )
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
        comp_stream = Filters.apply_async!(
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
    checked_wait(device, comp_stream, nothing, dgfvm.check_for_crashes)
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
