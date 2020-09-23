using ..Mesh.Grids: commmapping
using ..Mesh.Filters:
    AbstractFilterTarget,
    FilterIndices,
    vars_state_filtered,
    number_state_filtered
using ..BalanceLaws: compute_face_normal_velocity!

"""
    mpp_initialize(dg, state_prognostic, mpp_target)

Creates the data needed for the maximum principle preserving method. The
`DGModel` that the scheme is being applied to is `dg`, the initial condition for
the scheme is in the `MPIStateArray` `state_prognostic`, and the `mpp_target`
specifies which of the `Prognostic` states associates with the balance law
`dg.balance_law` should be handled by the MPP scheme.

The `mpp_target` can be specified as

  - `:` for all states
  - range of integers
  - a `Tuple` of symbols or numbers
  - an `AbstractFilterTarget`

The `mpp_target` possibilities for the MPP method mimic those of
[`Filters.apply!`](@ref)

If the `mpp_target` is a tuple of symbols, these symbols are propagated as the vars
names in the created `MPIStateArray`s
"""
function mpp_initialize(
    dg,
    state_prognostic,
    indices::Union{Colon, AbstractRange, Tuple{Vararg{Integer}}},
)
    if indices isa Colon
        indices = 1:size(state_prognostic, 2)
    end
    mpp_initialize(dg, state_prognostic, FilterIndices(indices...))
end

function mpp_initialize(dg, state_prognostic, vs::Tuple{Vararg{Symbol}})
    vars_target = NamedTuple{vs, NTuple{length(vs), eltype(state_prognostic)}}
    mpp_initialize(
        dg,
        state_prognostic,
        FilterIndices(varsindices(vars(state_prognostic), vs)...),
        vars_target,
    )
end
function mpp_initialize(dg, state_prognostic, vs::Tuple)
    vars_target =
        NamedTuple{Symbol.(vs), NTuple{length(vs), eltype(state_prognostic)}}
    mpp_initialize(
        dg,
        state_prognostic,
        FilterIndices(varsindices(vars(state_prognostic), vs)...),
        vars_target,
    )
end

vars_state_mpp_aux(FT) = @vars(velᵀn::SVector{6, FT}, vol::FT)
function mpp_initialize(
    dg,
    state_prognostic,
    mpp_target::AbstractFilterTarget,
    vars_target = vars_state_filtered(mpp_target, eltype(state_prognostic)),
)
    # Extract necessary information from the inputs
    balance_law = dg.balance_law
    grid = dg.grid
    topology = grid.topology
    FT = eltype(state_prognostic)
    DA = arraytype(grid)
    N = polynomialorder(grid)
    dim = dimensionality(grid)
    Nq = N + 1
    Nqj = dim == 2 ? 1 : Nq

    # Compute the targets for MPP
    num_state_mpp = varsize(vars_target)

    # Build the FVM communication patterns
    (fvm_vmaprecv, fvm_nabrtovmaprecv) = commmapping(
        0,
        topology.ghostelems,
        topology.ghostfaces,
        topology.nabrtorecv,
    )
    (fvm_vmapsend, fvm_nabrtovmapsend) = commmapping(
        0,
        topology.sendelems,
        topology.sendfaces,
        topology.nabrtosend,
    )

    # Create storage for the auxiliary data for the MPP scheme
    vars_mpp_aux = vars_state_mpp_aux(FT)
    mpp_aux = MPIStateArray{FT, vars_mpp_aux}(
        topology.mpicomm,
        DA,
        1,
        varsize(vars_mpp_aux),
        length(topology.elems);
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = fvm_vmaprecv,
        vmapsend = fvm_vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = fvm_nabrtovmaprecv,
        nabrtovmapsend = fvm_nabrtovmapsend,
    )

    # Create storage for the average state
    mpp_avg = MPIStateArray{FT, vars_target}(
        topology.mpicomm,
        DA,
        1,
        num_state_mpp,
        length(topology.elems);
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = fvm_vmaprecv,
        vmapsend = fvm_vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = fvm_nabrtovmaprecv,
        nabrtovmapsend = fvm_nabrtovmapsend,
        weights = mpp_aux.vol,
    )

    # Create storage for the FVM flux
    fvm_flux = MPIStateArray{FT, vars_target}(
        topology.mpicomm,
        DA,
        2 * dim, # number of faces
        num_state_mpp,
        length(topology.elems);
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = fvm_vmaprecv,
        vmapsend = fvm_vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = fvm_nabrtovmaprecv,
        nabrtovmapsend = fvm_nabrtovmapsend,
        weights = mpp_aux.vol,
    )

    # Create storage for the DG flux
    ∫dg_flux = MPIStateArray{FT, vars_target}(
        topology.mpicomm,
        DA,
        2 * dim, # number of faces
        num_state_mpp,
        length(topology.elems);
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = fvm_vmaprecv,
        vmapsend = fvm_vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = fvm_nabrtovmaprecv,
        nabrtovmapsend = fvm_nabrtovmapsend,
        weights = mpp_aux.vol,
    )
    fill!(∫dg_flux, 0)

    # Create storage for the DG flux
    Λ_min = MPIStateArray{FT, vars_target}(
        topology.mpicomm,
        DA,
        2 * dim, # number of faces
        num_state_mpp,
        length(topology.elems);
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = fvm_vmaprecv,
        vmapsend = fvm_vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = fvm_nabrtovmaprecv,
        nabrtovmapsend = fvm_nabrtovmapsend,
        weights = mpp_aux.vol,
    )

    # Initialize the averages and volumes of the elements
    device = array_device(state_prognostic)
    nreduce = 2^ceil(Int, log2(Nq * Nqj))
    nrealelem = length(topology.realelems)
    event = Event(device)
    event = knl_mpp_initialize!(device, (Nq, Nqj))(
        Val(nreduce),
        Val(dim),
        Val(N),
        mpp_avg.data,
        mpp_aux.data,
        state_prognostic.data,
        mpp_target,
        grid.vgeo,
        dependencies = (event,);
        ndrange = (nrealelem * Nq, Nqj),
    )

    # Exchange averages and volumes to neighbors
    avg_exchange =
        MPIStateArrays.begin_ghost_exchange!(mpp_avg, dependencies = event)
    avg_exchange =
        MPIStateArrays.end_ghost_exchange!(mpp_avg; dependencies = avg_exchange)

    aux_exchange =
        MPIStateArrays.begin_ghost_exchange!(mpp_aux, dependencies = event)
    aux_exchange =
        MPIStateArrays.end_ghost_exchange!(mpp_aux; dependencies = aux_exchange)

    wait(device, MultiEvent((avg_exchange, aux_exchange)))

    # Since the weights were not right when the MPIStateArray was created we
    # reset them with the right volume averages
    copyto!(mpp_avg.weights, mpp_aux.vol)
    copyto!(fvm_flux.weights, mpp_aux.vol)

    return (
        state = mpp_avg,
        fvm_flux = fvm_flux,
        ∫dg_flux = ∫dg_flux,
        Λ_min = Λ_min,
        aux = mpp_aux,
        target = mpp_target,
        elemtoelem = DA(topology.elemtoelem),
        elemtoface = DA(topology.elemtoface),
    )
end

function mpp_step_initialize!(
    dg::DGModel,
    state_prognostic::MPIStateArray,
    mppdata,
    t,
)

    # TODO:
    # - MPI communication

    balance_law = dg.balance_law
    device = array_device(state_prognostic)

    #XXX: Need to handle MPI case (comm state_prognostic?)
    update_auxiliary_state!(
        dg,
        balance_law,
        state_prognostic,
        t,
        dg.grid.topology.realelems,
    )

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk
    nrealelem = length(topology.realelems)
    nreduce = 2^ceil(Int, log2(Nfp))

    workgroups_surface = Nfp
    ndrange_interior_surface = Nfp * length(grid.interiorelems)
    ndrange_exterior_surface = Nfp * length(grid.exteriorelems)

    event = Event(device)
    avg_exchange = MPIStateArrays.begin_ghost_exchange!(
        mppdata.state,
        dependencies = event,
    )
    int_comp = knl_fvmflux!(device, workgroups_surface)(
        balance_law,
        Val(dim),
        Val(N),
        Val(nreduce),
        mppdata.target,
        mppdata.fvm_flux.data,
        mppdata.state.data,
        state_prognostic.data,
        dg.state_auxiliary.data,
        grid.sgeo,
        t,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.interiorelems;
        ndrange = ndrange_interior_surface,
        dependencies = (event,),
    )
    avg_exchange = MPIStateArrays.end_ghost_exchange!(
        mppdata.state;
        dependencies = avg_exchange,
    )
    ext_comp = knl_fvmflux!(device, workgroups_surface)(
        balance_law,
        Val(dim),
        Val(N),
        Val(nreduce),
        mppdata.target,
        mppdata.fvm_flux.data,
        mppdata.state.data,
        state_prognostic.data,
        dg.state_auxiliary.data,
        grid.sgeo,
        t,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.exteriorelems;
        ndrange = ndrange_exterior_surface,
        dependencies = (avg_exchange,),
    )
    wait(device, MultiEvent((int_comp, ext_comp)))
end

function mpp_update!(dg::DGModel, state_prognostic::MPIStateArray, mppdata, dt)
    balance_law = dg.balance_law
    device = array_device(state_prognostic)

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqj = dim == 2 ? 1 : Nq
    nrealelem = length(topology.realelems)

    groupsize = 256
    comp_stream = Event(device)

    # Compute the Λ for each element face
    comp_stream = knl_mpp_compute_Λ!(device, groupsize)(
        Val(dim),
        Val(N),
        mppdata.target,
        mppdata.state.data,
        mppdata.Λ_min.data,
        mppdata.fvm_flux.data,
        mppdata.∫dg_flux.data,
        mppdata.aux.data,
        dt,
        topology.realelems,
        ndrange = nrealelem,
        dependencies = (comp_stream,),
    )

    # TODO: Compare against the DG flux
    comp_stream = knl_mpp_update!(device, groupsize)(
        Val(dim),
        Val(N),
        mppdata.target,
        mppdata.state.data,
        mppdata.Λ_min.data,
        mppdata.fvm_flux.data,
        mppdata.∫dg_flux.data,
        mppdata.aux.data,
        dt,
        mppdata.elemtoelem,
        mppdata.elemtoface,
        topology.realelems,
        ndrange = nrealelem,
        dependencies = (comp_stream,),
    )

    # Initialize the averages and volumes of the elements
    nreduce = 2^ceil(Int, log2(Nq * Nqj))
    comp_stream = knl_mpp_swap_average!(device, (Nq, Nqj))(
        Val(nreduce),
        Val(dim),
        Val(N),
        state_prognostic.data,
        mppdata.state.data,
        mppdata.aux.data,
        mppdata.target,
        grid.vgeo,
        dependencies = (comp_stream,);
        ndrange = (nrealelem * Nq, Nqj),
    )

    # The synchronization here through a device event prevents CuArray based and
    # other default stream kernels from launching before the work scheduled in
    # this function is finished.
    wait(device, comp_stream)
end

@kernel function fct_fvmupdate!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    numerical_fvm_flux_first_order,
    average_state_prognostic,
    average_state_prognostic_flux,
    surface_volume_ratio,
    dt,
    elems,
) where {dim, polyorder}
    @uniform begin
        N = polyorder
        FT = eltype(average_state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())

        if dim == 1
            nface = 2
        elseif dim == 2
            nface = 4
        elseif dim == 3
            nface = 6
        end

        faces = 1:nface

        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq

        local_average_state_prognostic =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
    end

    # Read the global thread ID
    gid = @index(Global, Linear)

    # Get the element we are working on
    @inbounds e = elems[gid]

    # Load our cells data
    @inbounds @unroll for s in 1:num_state_prognostic
        local_average_state_prognostic[s] = average_state_prognostic[1, s, e]
    end

    # Loop over fluxes and add in
    @inbounds for f in faces
        λ = dt * surface_volume_ratio[f, 1, e]
        @inbounds @unroll for s in 1:num_state_prognostic
            local_average_state_prognostic[s] +=
                λ * average_state_prognostic_flux[f, s, e]
        end
    end

    @inbounds @unroll for s in 1:num_state_prognostic
        average_state_prognostic[1, s, e] = local_average_state_prognostic[s]
    end
end

@kernel function knl_mpp_initialize!(
    ::Val{nreduce},
    ::Val{dim},
    ::Val{N},
    mpp_avg,
    mpp_aux,
    state_prognostic,
    mpp_target::FilterIndices{I},
    vgeo,
) where {nreduce, dim, N, I}
    @uniform begin
        FT = eltype(state_prognostic)

        Nq = N + 1
        Nqj = dim == 2 ? 1 : Nq

        num_state_mpp = number_state_filtered(mpp_target, FT)
    end

    # For the pencil mass matrix
    l_MJ = @private FT (Nq,)

    # Storage for element volume
    l_vol = @private FT (1,)

    # Shared memory for the reduction
    s_reduce = @localmem FT (Nq * Nqj,)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        # loop up the pencil and load MJ into local memory
        @unroll for k in 1:Nq
            ijk = i + Nq * ((j - 1) + Nqj * (k - 1))

            l_MJ[k] = vgeo[ijk, _M, e]
        end

        ##############################
        # Compute the element volume #
        ##############################

        # Loop up pencil to compute this pencils contribution
        V = -zero(FT)
        @unroll for k in 1:Nq
            V += l_MJ[k]
        end
        ij = i + Nq * (j - 1)
        s_reduce[ij] = V
        @synchronize

        # Reduce thread block to get total volume
        @unroll for n in 11:-1:1
            if nreduce ≥ 2^n
                ij = i + Nq * (j - 1)
                ijshift = ij + 2^(n - 1)
                if ij ≤ 2^(n - 1) && ijshift ≤ Nq * Nqj
                    s_reduce[ij] += s_reduce[ijshift]
                end
                @synchronize
            end
        end

        # Save the volume for this element to use below
        l_vol[1] = s_reduce[1]

        # Store the volume in the mpp_aux
        if i == 1 && j == 1
            Vars{vars_state_mpp_aux(FT)}(view(mpp_aux, 1, :, e)).vol = l_vol[1]
        end

        ###############################
        # Compute the element average #
        ###############################

        # loop over the filtered states and compute the total mass for the
        # filtered states
        @unroll for s in 1:num_state_mpp
            # Get the prognostic state to apply mpp to
            s_prognostic = I[s]

            # compute the mass in the pencil
            MJ_state_prognostic = -zero(FT)
            @unroll for k in 1:Nq
                ijk = i + Nq * ((j - 1) + Nqj * (k - 1))
                MJ = l_MJ[k]
                Qs = state_prognostic[ijk, s_prognostic, e]

                MJ_state_prognostic += MJ * Qs
            end

            # do the reduction for the element
            ij = i + Nq * (j - 1)
            s_reduce[ij] = MJ_state_prognostic
            @synchronize
            @unroll for n in 11:-1:1
                if nreduce ≥ 2^n
                    ij = i + Nq * (j - 1)
                    ijshift = ij + 2^(n - 1)
                    if ij ≤ 2^(n - 1) && ijshift ≤ Nq * Nqj
                        s_reduce[ij] += s_reduce[ijshift]
                    end
                    @synchronize
                end
            end

            # Save the average value for the element
            if i == 1 && j == 1
                mpp_avg[1, s, e] = s_reduce[1] / l_vol[1]
            end
        end
    end
end

@kernel function knl_fvmflux!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    ::Val{nreduce},
    mpp_target::FilterIndices{I},
    fvm_flux,
    mpp_avg,
    state_prognostic,
    state_auxiliary,
    sgeo,
    t,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
) where {dim, polyorder, I, nreduce}
    @uniform begin
        N = polyorder
        FT = eltype(mpp_avg)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        num_state_mpp = number_state_filtered(mpp_target, FT)

        if dim == 1
            Np = (N + 1)
            Nfp = 1
            nface = 2
        elseif dim == 2
            Np = (N + 1) * (N + 1)
            Nfp = (N + 1)
            nface = 4
        elseif dim == 3
            Np = (N + 1) * (N + 1) * (N + 1)
            Nfp = (N + 1) * (N + 1)
            nface = 6
        end

        faces = 1:nface

        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq

        local_state_prognostic⁻ = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic⁺ = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary⁻ = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_auxiliary⁺ = MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_velocity⁻ = MArray{Tuple{1}, FT}(undef)
        local_velocity⁺ = MArray{Tuple{1}, FT}(undef)
    end

    # Read the global thread ID
    grp_id = @index(Group, Linear)

    # Get the element we are working on
    e = @private Int (1,)
    @inbounds e[1] = elems[grp_id]
    n = @index(Local, Linear)
    p_sM = @private FT (1,)

    s_flux = @localmem FT (Nfp, num_state_mpp)

    # Loop over neighbors and compute flux
    @inbounds for f in faces
        e⁻ = e[1]
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1

        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1

        normal_vector = SVector(
            sgeo[_n1, n, f, e⁻],
            sgeo[_n2, n, f, e⁻],
            sgeo[_n3, n, f, e⁻],
        )

        p_sM[1] = sgeo[_sM, n, f, e⁻]

        # Load minus side data
        @unroll for s in 1:num_state_prognostic
            local_state_prognostic⁻[s] = state_prognostic[vid⁻, s, e⁻]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁻[s] = state_auxiliary[vid⁻, s, e⁻]
        end

        # Load neighboring data
        @unroll for s in 1:num_state_prognostic
            local_state_prognostic⁺[s] = state_prognostic[vid⁺, s, e⁺]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁺[s] = state_auxiliary[vid⁺, s, e⁺]
        end

        # FIXME: Do we need to handle diffusion?
        bctype = elemtobndy[f, e⁻]
        if bctype == 0
            # Get the normal velocity on both sides of the face
            compute_face_normal_velocity!(
                balance_law,
                local_velocity⁻,
                normal_vector,
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁻,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁻,
                ),
                t,
            )

            compute_face_normal_velocity!(
                balance_law,
                local_velocity⁺,
                normal_vector,
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁺,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁺,
                ),
                t,
            )

            # FIXME: use proper upwind flux for discontinuous velocity field?
            vel = (local_velocity⁻[1] + local_velocity⁺[1]) / 2

            # Compute the contribution of this face point to the total flux
            @unroll for s in 1:num_state_mpp
                up_state = vel > 0 ? mpp_avg[1, s, e⁻] : mpp_avg[1, s, e⁺]

                s_flux[n, s] = p_sM[1] * vel * up_state
            end

        else
            # FIXME: we assume that the flux is zero at the boundary. Need to
            # figure out how to handle more general BCs
            @unroll for s in 1:num_state_mpp
                s_flux[n, s] = 0
            end
        end

        # Sum up fluxes along the face (e.g., surface integral)
        @unroll for m in 11:-1:1
            if nreduce ≥ 2^m
                nshift = n + 2^(m - 1)
                if n ≤ 2^(m - 1) && nshift ≤ Nfp
                    @unroll for s in 1:num_state_mpp
                        s_flux[n, s] += s_flux[nshift, s]
                    end
                end
                @synchronize
            end
        end

        # Store the total fluxes for this face
        if n == 1
            @unroll for s in 1:num_state_mpp
                fvm_flux[f, s, e[1]] = s_flux[n, s]
            end
        end
    end
end

@kernel function knl_mpp_compute_Λ!(
    ::Val{dim},
    ::Val{polyorder},
    mpp_target::FilterIndices{I},
    mpp_avg,
    mpp_Λ_min,
    fvm_flux,
    ∫dg_flux, # Scaled by dt
    mpp_aux,
    dt,
    elems,
) where {dim, polyorder, I}
    @uniform begin
        N = polyorder
        FT = eltype(mpp_avg)
        num_state_mpp = number_state_filtered(mpp_target, FT)

        if dim == 1
            Np = (N + 1)
            nface = 2
        elseif dim == 2
            Np = (N + 1) * (N + 1)
            nface = 4
        elseif dim == 3
            Np = (N + 1) * (N + 1) * (N + 1)
            nface = 6
        end

        faces = 1:nface

        l_Λ = MArray{Tuple{nface}, FT}(undef)
        l_Δflx = MArray{Tuple{nface}, FT}(undef)
    end

    # Get the element we are working on
    gid = @index(Global, Linear)
    @inbounds e = elems[gid]

    # Load geometry
    elem_vol = Vars{vars_state_mpp_aux(FT)}(view(mpp_aux, 1, :, e)).vol

    @inbounds for s in 1:num_state_mpp
        # Load element data
        fld = fv_fld = dg_fld = mpp_avg[1, s, e]

        ΔF_outflow = -zero(FT)

        # Loop through faces and add flux
        @unroll for f in faces
            fv = dt * fvm_flux[f, s, e]
            dg = ∫dg_flux[f, s, e]

            fv_fld -= fv / elem_vol
            dg_fld -= dg / elem_vol

            l_Δflx[f] = (dg - fv) / elem_vol
            l_Λ[f] = 1

            ΔF_outflow += l_Δflx[f] > 0 ? l_Δflx[f] : 0
        end

        fld_min = 0 # TODO: should be the lower bound
        if dg_fld < fld_min
            @unroll for f in faces
                if l_Δflx[f] > 0
                    # XXX: max needed for case when fv_fld ≈ fld_min and
                    # fv_fld - fld_min < 0
                    l_Λ[f] = max(0, fv_fld - fld_min) / ΔF_outflow
                    # @assert 0 ≤ l_Λ[f] ≤ 1
                end
            end
        end
        @unroll for f in faces
            mpp_Λ_min[f, s, e] = l_Λ[f]
        end
    end
end

@kernel function knl_mpp_update!(
    ::Val{dim},
    ::Val{polyorder},
    mpp_target::FilterIndices{I},
    mpp_avg,
    mpp_Λ_min,
    fvm_flux,
    ∫dg_flux, # Scaled by dt
    mpp_aux,
    dt,
    elemtoelem,
    elemtoface,
    elems,
) where {dim, polyorder, I}
    @uniform begin
        N = polyorder
        FT = eltype(mpp_avg)
        num_state_mpp = number_state_filtered(mpp_target, FT)

        if dim == 1
            Np = (N + 1)
            nface = 2
        elseif dim == 2
            Np = (N + 1) * (N + 1)
            nface = 4
        elseif dim == 3
            Np = (N + 1) * (N + 1) * (N + 1)
            nface = 6
        end

        faces = 1:nface
    end

    # Get the element we are working on
    gid = @index(Global, Linear)
    @inbounds e⁻ = elems[gid]

    # Load geometry
    elem_vol = Vars{vars_state_mpp_aux(FT)}(view(mpp_aux, 1, :, e⁻)).vol

    @inbounds for s in 1:num_state_mpp
        # Load element data
        fld = mpp_avg[1, s, e⁻]

        outflow = -zero(FT)

        # Loop through faces and add flux
        @unroll for f⁻ in faces
            # Get the flux
            fv = dt * fvm_flux[f⁻, s, e⁻]
            dg = ∫dg_flux[f⁻, s, e⁻]

            # Get the plus side face and element number
            e⁺ = elemtoelem[f⁻, e⁻]
            f⁺ = elemtoface[f⁻, e⁻]

            # Compute θ mpp limiter
            θ = min(mpp_Λ_min[f⁻, s, e⁻], mpp_Λ_min[f⁺, s, e⁺])

            # Compute the limited FVM-style update
            fld -= (θ * dg + (1 - θ) * fv) / elem_vol
        end

        # Write out new average
        mpp_avg[1, s, e⁻] = fld
    end
end

@kernel function knl_mpp_swap_average!(
    ::Val{nreduce},
    ::Val{dim},
    ::Val{N},
    state_prognostic,
    mpp_avg,
    mpp_aux,
    mpp_target::FilterIndices{I},
    vgeo,
) where {nreduce, dim, N, I}
    @uniform begin
        FT = eltype(state_prognostic)

        Nq = N + 1
        Nqj = dim == 2 ? 1 : Nq

        num_state_mpp = number_state_filtered(mpp_target, FT)
    end

    # For the pencil mass matrix
    l_MJ = @private FT (Nq,)
    l_state_prognostic = @private FT (Nq,)

    # Storage for element volume
    l_vol = @private FT (1,)
    l_mpp_avg = @private FT (1,)

    # Shared memory for the reduction
    s_reduce = @localmem FT (Nq * Nqj,)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        # loop up the pencil and load MJ into local memory
        @unroll for k in 1:Nq
            ijk = i + Nq * ((j - 1) + Nqj * (k - 1))

            l_MJ[k] = vgeo[ijk, _M, e]
        end

        # Store the volume in the mpp_aux
        l_vol[1] = Vars{vars_state_mpp_aux(FT)}(view(mpp_aux, 1, :, e)).vol

        ###############################
        # Compute the element average #
        ###############################

        # loop over the filtered states and compute the total mass for the
        # filtered states
        @unroll for s in 1:num_state_mpp
            l_mpp_avg[1] = mpp_avg[1, s, e]

            # Get the prognostic state to apply mpp to
            s_prognostic = I[s]

            # compute the mass in the pencil
            MJ_state_prognostic = -zero(FT)
            @unroll for k in 1:Nq
                ijk = i + Nq * ((j - 1) + Nqj * (k - 1))
                MJ = l_MJ[k]
                l_state_prognostic[k] = state_prognostic[ijk, s_prognostic, e]

                MJ_state_prognostic += MJ * l_state_prognostic[k]
            end

            # do the reduction for the element
            ij = i + Nq * (j - 1)
            s_reduce[ij] = MJ_state_prognostic
            @synchronize
            @unroll for n in 11:-1:1
                if nreduce ≥ 2^n
                    ij = i + Nq * (j - 1)
                    ijshift = ij + 2^(n - 1)
                    if ij ≤ 2^(n - 1) && ijshift ≤ Nq * Nqj
                        s_reduce[ij] += s_reduce[ijshift]
                    end
                    @synchronize
                end
            end

            # Replace average mass
            s_prognostic = I[s]
            Δ_avg_mass = l_mpp_avg[1] - s_reduce[1] / l_vol[1]
            @unroll for k in 1:Nq
                ijk = i + Nq * ((j - 1) + Nqj * (k - 1))
                state_prognostic[ijk, s_prognostic, e] =
                    l_state_prognostic[k] + Δ_avg_mass
            end
        end
    end
end
