using .NumericalFluxes: CentralHyperDiffusiveFlux, CentralDivPenalty

struct DGModel{BL, G, NFND, NFD, GNF, AS, DS, HDS, D, DD, MD}
    balancelaw::BL
    grid::G
    numfluxnondiff::NFND
    numfluxdiff::NFD
    gradnumflux::GNF
    auxstate::AS
    diffstate::DS
    hyperdiffstate::HDS
    direction::D
    diffusion_direction::DD
    modeldata::MD
end
function DGModel(
    balancelaw,
    grid,
    numfluxnondiff,
    numfluxdiff,
    gradnumflux;
    auxstate = create_auxstate(balancelaw, grid),
    diffstate = create_diffstate(balancelaw, grid),
    hyperdiffstate = create_hyperdiffstate(balancelaw, grid),
    direction = EveryDirection(),
    diffusion_direction = direction,
    modeldata = nothing,
)
    DGModel(
        balancelaw,
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux,
        auxstate,
        diffstate,
        hyperdiffstate,
        direction,
        diffusion_direction,
        modeldata,
    )
end

function (dg::DGModel)(dQdt, Q, ::Nothing, t; increment = false)

    bl = dg.balancelaw
    device = typeof(Q.data) <: Array ? CPU() : CUDA()

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk
    nrealelem = length(topology.realelems)

    Qvisc = dg.diffstate
    Qhypervisc_grad, Qhypervisc_div = dg.hyperdiffstate
    auxstate = dg.auxstate

    FT = eltype(Q)
    nviscstate = num_diffusive(bl, FT)
    nhyperviscstate = num_hyperdiffusive(bl, FT)

    Np = dofs_per_element(grid)

    workgroups_volume = (Nq, Nq, Nqk)
    ndrange_volume = (nrealelem * Nq, Nq, Nqk)
    workgroups_surface = Nfp
    ndrange_surface = Nfp * nrealelem

    communicate =
        !(isstacked(topology) && typeof(dg.direction) <: VerticalDirection)

    aux_comm = update_aux!(dg, bl, Q, t)
    @assert typeof(aux_comm) == Bool

    if nhyperviscstate > 0
        hypervisc_indexmap = create_hypervisc_indexmap(bl)
    else
        hypervisc_indexmap = nothing
    end

    ########################
    # Gradient Computation #
    ########################
    if communicate
        MPIStateArrays.start_ghost_exchange!(Q)
        if aux_comm
            MPIStateArrays.start_ghost_exchange!(auxstate)
        end
    end

    if nviscstate > 0 || nhyperviscstate > 0

        event = Event(device)
        event = volumeviscterms!(device, workgroups_volume)(
            bl,
            Val(dim),
            Val(N),
            dg.diffusion_direction,
            Q.data,
            Qvisc.data,
            Qhypervisc_grad.data,
            auxstate.data,
            grid.vgeo,
            t,
            grid.D,
            hypervisc_indexmap,
            topology.realelems,
            ndrange = ndrange_volume,
            dependencies = (event,),
        )
        wait(device, event)

        if communicate
            MPIStateArrays.finish_ghost_recv!(Q)
            if aux_comm
                MPIStateArrays.finish_ghost_recv!(auxstate)
            end
        end

        event = Event(device)
        event = faceviscterms!(device, workgroups_surface)(
            bl,
            Val(dim),
            Val(N),
            dg.diffusion_direction,
            dg.gradnumflux,
            Q.data,
            Qvisc.data,
            Qhypervisc_grad.data,
            auxstate.data,
            grid.vgeo,
            grid.sgeo,
            t,
            grid.vmap⁻,
            grid.vmap⁺,
            grid.elemtobndy,
            hypervisc_indexmap,
            topology.realelems;
            ndrange = ndrange_surface,
            dependencies = (event,),
        )
        wait(device, event)

        if communicate
            nviscstate > 0 && MPIStateArrays.start_ghost_exchange!(Qvisc)
            nhyperviscstate > 0 &&
            MPIStateArrays.start_ghost_exchange!(Qhypervisc_grad)
        end

        if nviscstate > 0
            aux_comm = update_aux_diffusive!(dg, bl, Q, t)
            @assert typeof(aux_comm) == Bool
        end

        if aux_comm
            MPIStateArrays.start_ghost_exchange!(auxstate)
        end
    end

    if nhyperviscstate > 0
        #########################
        # Laplacian Computation #
        #########################

        event = Event(device)
        event = volumedivgrad!(device, workgroups_volume)(
            bl,
            Val(dim),
            Val(N),
            dg.diffusion_direction,
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            grid.vgeo,
            grid.D,
            topology.realelems;
            ndrange = ndrange_volume,
            dependencies = (event,),
        )
        wait(device, event)

        communicate && MPIStateArrays.finish_ghost_recv!(Qhypervisc_grad)

        event = Event(device)
        event = facedivgrad!(device, workgroups_surface)(
            bl,
            Val(dim),
            Val(N),
            dg.diffusion_direction,
            CentralDivPenalty(),
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            grid.vgeo,
            grid.sgeo,
            grid.vmap⁻,
            grid.vmap⁺,
            grid.elemtobndy,
            topology.realelems;
            ndrange = ndrange_surface,
            dependencies = (event,),
        )
        wait(device, event)

        communicate && MPIStateArrays.start_ghost_exchange!(Qhypervisc_div)

        ####################################
        # Hyperdiffusive terms computation #
        ####################################

        event = Event(device)
        event = volumehyperviscterms!(device, workgroups_volume)(
            bl,
            Val(dim),
            Val(N),
            dg.diffusion_direction,
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            Q.data,
            auxstate.data,
            grid.vgeo,
            grid.ω,
            grid.D,
            topology.realelems,
            t;
            ndrange = ndrange_volume,
            dependencies = (event,),
        )
        wait(device, event)

        communicate && MPIStateArrays.finish_ghost_recv!(Qhypervisc_div)

        event = Event(device)
        event = facehyperviscterms!(device, workgroups_surface)(
            bl,
            Val(dim),
            Val(N),
            dg.diffusion_direction,
            CentralHyperDiffusiveFlux(),
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            Q.data,
            auxstate.data,
            grid.vgeo,
            grid.sgeo,
            grid.vmap⁻,
            grid.vmap⁺,
            grid.elemtobndy,
            topology.realelems,
            t;
            ndrange = ndrange_surface,
            dependencies = (event,),
        )
        wait(device, event)

        communicate && MPIStateArrays.start_ghost_exchange!(Qhypervisc_grad)
    end


    ###################
    # RHS Computation #
    ###################
    event = Event(device)
    event = volumerhs!(device, workgroups_volume)(
        bl,
        Val(dim),
        Val(N),
        dg.direction,
        dQdt.data,
        Q.data,
        Qvisc.data,
        Qhypervisc_grad.data,
        auxstate.data,
        grid.vgeo,
        t,
        grid.ω,
        grid.D,
        topology.realelems,
        increment;
        ndrange = ndrange_volume,
        dependencies = (event,),
    )
    wait(device, event)

    if communicate
        if nviscstate > 0 || nhyperviscstate > 0
            if nviscstate > 0
                MPIStateArrays.finish_ghost_recv!(Qvisc)
                if aux_comm
                    MPIStateArrays.finish_ghost_recv!(auxstate)
                end
            end
            nhyperviscstate > 0 &&
            MPIStateArrays.finish_ghost_recv!(Qhypervisc_grad)
        else
            MPIStateArrays.finish_ghost_recv!(Q)
            if aux_comm
                MPIStateArrays.finish_ghost_recv!(auxstate)
            end
        end
    end

    event = Event(device)
    event = facerhs!(device, workgroups_surface)(
        bl,
        Val(dim),
        Val(N),
        dg.direction,
        dg.numfluxnondiff,
        dg.numfluxdiff,
        dQdt.data,
        Q.data,
        Qvisc.data,
        Qhypervisc_grad.data,
        auxstate.data,
        grid.vgeo,
        grid.sgeo,
        t,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        topology.realelems;
        ndrange = ndrange_surface,
        dependencies = (event,),
    )
    wait(device, event)

    # Just to be safe, we wait on the sends we started.
    if communicate
        MPIStateArrays.finish_ghost_send!(Qhypervisc_div)
        MPIStateArrays.finish_ghost_send!(Qvisc)
        MPIStateArrays.finish_ghost_send!(Qhypervisc_grad)
        MPIStateArrays.finish_ghost_send!(Q)
    end
end

function init_ode_state(
    dg::DGModel,
    args...;
    init_on_cpu = false,
    commtag = 888,
)
    device = arraytype(dg.grid) <: Array ? CPU() : CUDA()

    bl = dg.balancelaw
    grid = dg.grid

    state = create_state(bl, grid, commtag)

    topology = grid.topology
    Np = dofs_per_element(grid)

    auxstate = dg.auxstate
    dim = dimensionality(grid)
    N = polynomialorder(grid)
    nrealelem = length(topology.realelems)

    if !init_on_cpu
        event = Event(device)
        event = initstate!(device, Np)(
            bl,
            Val(dim),
            Val(N),
            state.data,
            auxstate.data,
            grid.vgeo,
            topology.realelems,
            args...;
            ndrange = Np * nrealelem,
            dependencies = (event,),
        )
        wait(device, event)
    else
        h_state = similar(state, Array)
        h_auxstate = similar(auxstate, Array)
        h_auxstate .= auxstate
        event = initstate!(CPU(), Np)(
            bl,
            Val(dim),
            Val(N),
            h_state.data,
            h_auxstate.data,
            Array(grid.vgeo),
            topology.realelems,
            args...;
            ndrange = Np * nrealelem,
        )
        wait(event) # XXX: This could be `wait(device, event)` once KA supports that.
        state .= h_state
    end

    MPIStateArrays.start_ghost_exchange!(state)
    MPIStateArrays.finish_ghost_exchange!(state)

    return state
end

# fallback
function update_aux!(dg::DGModel, bl::BalanceLaw, Q::MPIStateArray, t::Real)
    return false
end

function update_aux_diffusive!(
    dg::DGModel,
    bl::BalanceLaw,
    Q::MPIStateArray,
    t::Real,
)
    return false
end

function indefinite_stack_integral!(
    dg::DGModel,
    m::BalanceLaw,
    Q::MPIStateArray,
    auxstate::MPIStateArray,
    t::Real,
)

    device = typeof(Q.data) <: Array ? CPU() : CUDA()

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq

    FT = eltype(Q)

    # do integrals
    nelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nelem, nvertelem)

    event = Event(device)
    event = knl_indefinite_stack_integral!(device, (Nq, Nqk))(
        m,
        Val(dim),
        Val(N),
        Val(nvertelem),
        Q.data,
        auxstate.data,
        grid.vgeo,
        grid.Imat,
        1:nhorzelem;
        ndrange = (nhorzelem * Nq, Nqk),
        dependencies = (event,),
    )
    wait(device, event)
end

function reverse_indefinite_stack_integral!(
    dg::DGModel,
    m::BalanceLaw,
    Q::MPIStateArray,
    auxstate::MPIStateArray,
    t::Real,
)

    device = typeof(auxstate.data) <: Array ? CPU() : CUDA()

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq

    FT = eltype(auxstate)

    # do integrals
    nelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nelem, nvertelem)

    event = Event(device)
    event = knl_reverse_indefinite_stack_integral!(device, (Nq, Nqk))(
        m,
        Val(dim),
        Val(N),
        Val(nvertelem),
        Q.data,
        auxstate.data,
        1:nhorzelem;
        ndrange = (nhorzelem * Nq, Nqk),
        dependencies = (event,),
    )
    wait(device, event)
end

function nodal_update_aux!(
    f!,
    dg::DGModel,
    m::BalanceLaw,
    Q::MPIStateArray,
    t::Real;
    diffusive = false,
)
    device = typeof(Q.data) <: Array ? CPU() : CUDA()

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    nrealelem = length(topology.realelems)

    Np = dofs_per_element(grid)

    nodal_update_aux! = knl_nodal_update_aux!(device, Np)
    ### update aux variables
    event = Event(device)
    if diffusive
        event = nodal_update_aux!(
            m,
            Val(dim),
            Val(N),
            f!,
            Q.data,
            dg.auxstate.data,
            dg.diffstate.data,
            t,
            topology.realelems;
            ndrange = Np * nrealelem,
            dependencies = (event,),
        )
    else
        event = nodal_update_aux!(
            m,
            Val(dim),
            Val(N),
            f!,
            Q.data,
            dg.auxstate.data,
            t,
            topology.realelems;
            ndrange = Np * nrealelem,
            dependencies = (event,),
        )
    end
    wait(device, event)
end

"""
    courant(local_courant::Function, dg::DGModel, m::BalanceLaw,
            Q::MPIStateArray, direction=EveryDirection())
Returns the maximum of the evaluation of the function `local_courant`
pointwise throughout the domain.  The function `local_courant` is given an
approximation of the local node distance `Δx`.  The `direction` controls which
reference directions are considered when computing the minimum node distance
`Δx`.
An example `local_courant` function is
    function local_courant(m::AtmosModel, state::Vars, aux::Vars,
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
    Q::MPIStateArray,
    Δt,
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
        device = grid.vgeo isa Array ? CPU() : CUDA()
        pointwise_courant = similar(grid.vgeo, Nq^dim, nrealelem)
        event = Event(device)
        event = Grids.knl_min_neighbor_distance!(device, (Nq, Nq, Nqk))(
            Val(N),
            Val(dim),
            direction,
            pointwise_courant,
            grid.vgeo,
            topology.realelems;
            ndrange = (nrealelem * Nq, Nq, Nqk),
            dependencies = (event,),
        )
        event = knl_local_courant!(device, Nq * Nq * Nqk)(
            m,
            Val(dim),
            Val(N),
            pointwise_courant,
            local_courant,
            Q.data,
            dg.auxstate.data,
            dg.diffstate.data,
            topology.realelems,
            direction,
            Δt;
            ndrange = nrealelem * Nq * Nq * Nqk,
            dependencies = (event,),
        )
        wait(device, event)
        rank_courant_max = maximum(pointwise_courant)
    else
        rank_courant_max = typemin(eltype(Q))
    end

    MPI.Allreduce(rank_courant_max, max, topology.mpicomm)
end

function copy_stack_field_down!(
    dg::DGModel,
    m::BalanceLaw,
    auxstate::MPIStateArray,
    fldin,
    fldout,
)

    device = typeof(auxstate.data) <: Array ? CPU() : CUDA()

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq

    # do integrals
    nelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nelem, nvertelem)

    event = Event(device)
    event = knl_copy_stack_field_down!(device, (Nq, Nqk))(
        Val(dim),
        Val(N),
        Val(nvertelem),
        auxstate.data,
        1:nhorzelem,
        Val(fldin),
        Val(fldout);
        ndrange = (nhorzelem * Nq, Nqk),
        dependencies = (event,),
    )
    wait(device, event)
end

function MPIStateArrays.MPIStateArray(dg::DGModel, commtag = 888)
    bl = dg.balancelaw
    grid = dg.grid

    state = create_state(bl, grid, commtag)

    return state
end

function create_hypervisc_indexmap(bl::BalanceLaw)
    # helper function
    _getvars(v, ::Type) = v
    function _getvars(v::Vars, ::Type{T}) where {T <: NamedTuple}
        fields = getproperty.(Ref(v), fieldnames(T))
        collect(Iterators.Flatten(_getvars.(fields, fieldtypes(T))))
    end

    gradvars = vars_gradient(bl, Int)
    gradlapvars = vars_gradient_laplacian(bl, Int)
    indices = Vars{gradvars}(1:varsize(gradvars))
    SVector{varsize(gradlapvars)}(_getvars(indices, gradlapvars))
end
