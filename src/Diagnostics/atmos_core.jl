using ..Atmos
using ..Atmos: thermo_state
using ..Mesh.Topologies
using ..Mesh.Grids
using ..MoistThermodynamics
using LinearAlgebra

"""
    atmos_core_init(bl, currtime)

Initialize the 'AtmosCore' diagnostics group.
"""
function atmos_core_init(dgngrp::DiagnosticsGroup, currtime)
    dg = Settings.dg
    bl = dg.balancelaw
    if isa(bl.moisture, DryModel)
        @warn """
            Diagnostics ($dgngrp.name): cannot be used with `DryModel`
            """
    end

    atmos_collect_onetime(Settings.mpicomm, Settings.dg, Settings.Q)

    return nothing
end

# Simple horizontal averages
function vars_atmos_core_simple(m::AtmosModel, FT)
    @vars begin
        u_core::FT
        v_core::FT
        w_core::FT
        avg_rho_core::FT        # ρ
        rho_core::FT            # ρρ
        qt_core::FT             # q_tot
        thl_core::FT            # θ_liq
        ei_core::FT             # e_int
    end
end
num_atmos_core_simple_vars(m, FT) = varsize(vars_atmos_core_simple(m, FT))
atmos_core_simple_vars(m, array) =
    Vars{vars_atmos_core_simple(m, eltype(array))}(array)

function atmos_core_simple_sums!(atmos::AtmosModel, state, thermo, MH, sums)
    sums.u_core += MH * state.ρu[1]
    sums.v_core += MH * state.ρu[2]
    sums.w_core += MH * state.ρu[3]
    sums.avg_rho_core += MH * state.ρ
    sums.rho_core += MH * state.ρ * state.ρ
    sums.qt_core += MH * state.moisture.ρq_tot
    sums.thl_core += MH * thermo.moisture.θ_liq_ice * state.ρ
    sums.ei_core += MH * thermo.e_int * state.ρ

    return nothing
end

# Variances and covariances
function vars_atmos_core_ho(m::AtmosModel, FT)
    @vars begin
        var_u_core::FT          # u′u′
        var_v_core::FT          # v′v′
        var_w_core::FT          # w′w′
        var_qt_core::FT         # q_tot′q_tot′
        var_thl_core::FT        # θ_liq_ice′θ_liq_ice′
        var_ei_core::FT         # e_int′e_int′

        cov_w_rho_core::FT      # w′ρ′
        cov_w_qt_core::FT       # w′q_tot′
        cov_w_thl_core::FT      # w′θ_liq_ice′
        cov_w_ei_core::FT       # w′e_int′
        cov_qt_thl_core::FT     # q_tot′θ_liq_ice′
        cov_qt_ei_core::FT      # q_tot′e_int′
    end
end
num_atmos_core_ho_vars(m, FT) = varsize(vars_atmos_core_ho(m, FT))
atmos_core_ho_vars(m, array) = Vars{vars_atmos_core_ho(m, eltype(array))}(array)

function atmos_core_ho_sums!(atmos::AtmosModel, state, thermo, MH, ha, sums)
    u = state.ρu[1] / state.ρ
    u′ = u - ha.u_core
    v = state.ρu[2] / state.ρ
    v′ = v - ha.v_core
    w = state.ρu[3] / state.ρ
    w′ = w - ha.w_core
    q_tot = state.moisture.ρq_tot / state.ρ
    q_tot′ = q_tot - ha.qt_core
    θ_liq_ice′ = thermo.moisture.θ_liq_ice - ha.thl_core
    e_int′ = thermo.e_int - ha.ei_core

    sums.var_u_core += MH * u′^2 * state.ρ
    sums.var_v_core += MH * v′^2 * state.ρ
    sums.var_w_core += MH * w′^2 * state.ρ
    sums.var_qt_core += MH * q_tot′^2 * state.ρ
    sums.var_thl_core += MH * θ_liq_ice′^2 * state.ρ
    sums.var_ei_core += MH * e_int′^2 * state.ρ

    sums.cov_w_rho_core += MH * w′ * (state.ρ - ha.avg_rho_core) * state.ρ
    sums.cov_w_qt_core += MH * w′ * q_tot′ * state.ρ
    sums.cov_w_thl_core += MH * w′ * θ_liq_ice′ * state.ρ
    sums.cov_qt_thl_core += MH * q_tot′ * θ_liq_ice′ * state.ρ
    sums.cov_qt_ei_core += MH * q_tot′ * e_int′ * state.ρ
    sums.cov_w_ei_core += MH * w′ * e_int′ * state.ρ

    return nothing
end

"""
    atmos_core_collect(bl, currtime)

Perform a global grid traversal to compute various diagnostics.
"""
function atmos_core_collect(dgngrp::DiagnosticsGroup, currtime)
    dg = Settings.dg
    bl = dg.balancelaw
    if isa(bl.moisture, DryModel)
        @warn """
            Diagnostics $(dgngrp.name): cannot be used with `DryModel`
            """
    end

    mpicomm = Settings.mpicomm
    Q = Settings.Q
    mpirank = MPI.Comm_rank(mpicomm)
    current_time = string(currtime)

    # extract grid information
    grid = dg.grid
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    npoints = Nq * Nq * Nqk
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    # get the state, auxiliary and geo variables onto the host if needed
    if Array ∈ typeof(Q).parameters
        localQ = Q.realdata
        localaux = dg.auxstate.realdata
        localvgeo = grid.vgeo
        localdiff = dg.diffstate.realdata
    else
        localQ = Array(Q.realdata)
        localaux = Array(dg.auxstate.realdata)
        localvgeo = Array(grid.vgeo)
        localdiff = Array(dg.diffstate.realdata)
    end
    FT = eltype(localQ)

    zvals = AtmosCollected.zvals

    # Visit each node of the state variables array and:
    # - generate and store the thermo variables,
    # - if core condition holds (q_liq > 0 && w > 0)
    #   - count that point in the core fraction for that z
    #   - count the point's weighting towards averaging for that z, and
    #   - accumulate the simple horizontal sums
    #
    core_repdvsr = zeros(FT, Nqk * nvertelem)
    thermo_array =
        [zeros(FT, num_thermo(bl, FT)) for _ in 1:npoints, _ in 1:nrealelem]
    simple_sums = [
        zeros(FT, num_atmos_core_simple_vars(bl, FT))
        for _ in 1:(Nqk * nvertelem)
    ]
    ql_w_gt_0 = [zeros(FT, (Nq * Nq * nhorzelem)) for _ in 1:(Nqk * nvertelem)]
    @visitQ nhorzelem nvertelem Nqk Nq begin
        evk = Nqk * (ev - 1) + k

        state = extract_state(dg, localQ, ijk, e)
        aux = extract_aux(dg, localaux, ijk, e)
        MH = localvgeo[ijk, grid.MHid, e]

        thermo = thermo_vars(bl, thermo_array[ijk, e])
        compute_thermo!(bl, state, aux, thermo)

        if thermo.moisture.q_liq > 0 && state.ρu[3] > 0
            idx = (Nq * Nq * (eh - 1)) + (Nq * (j - 1)) + i
            ql_w_gt_0[evk][idx] = one(FT)
            core_repdvsr[evk] += MH

            simple = atmos_core_simple_vars(bl, simple_sums[evk])
            atmos_core_simple_sums!(bl, state, thermo, MH, simple)
        end
    end

    # reduce horizontal sums and core fraction across ranks and compute averages
    simple_avgs = [
        zeros(FT, num_atmos_core_simple_vars(bl, FT))
        for _ in 1:(Nqk * nvertelem)
    ]
    core_frac = zeros(FT, Nqk * nvertelem)
    MPI.Allreduce!(core_repdvsr, +, mpicomm)
    for evk in 1:(Nqk * nvertelem)
        tot_ql_w_gt_0 = MPI.Reduce(sum(ql_w_gt_0[evk]), +, 0, mpicomm)
        tot_horz = MPI.Reduce(length(ql_w_gt_0[evk]), +, 0, mpicomm)

        MPI.Allreduce!(simple_sums[evk], simple_avgs[evk], +, mpicomm)
        simple_avgs[evk] .= simple_avgs[evk] ./ core_repdvsr[evk]

        if mpirank == 0
            core_frac[evk] = tot_ql_w_gt_0 / tot_horz
        end
    end

    # complete density averaging
    simple_varnames = flattenednames(vars_atmos_core_simple(bl, FT))
    for vari in 1:length(simple_varnames)
        for evk in 1:(Nqk * nvertelem)
            simple_ha = atmos_core_simple_vars(bl, simple_avgs[evk])
            avg_rho = simple_ha.avg_rho_core
            if simple_varnames[vari] != "avg_rho_core"
                simple_avgs[evk][vari] /= avg_rho
            end
        end
    end

    # compute the variances and covariances
    ho_sums =
        [zeros(FT, num_atmos_core_ho_vars(bl, FT)) for _ in 1:(Nqk * nvertelem)]
    @visitQ nhorzelem nvertelem Nqk Nq begin
        evk = Nqk * (ev - 1) + k

        state = extract_state(dg, localQ, ijk, e)
        thermo = thermo_vars(bl, thermo_array[ijk, e])
        MH = localvgeo[ijk, grid.MHid, e]

        if thermo.moisture.q_liq > 0 && state.ρu[3] > 0
            simple_ha = atmos_core_simple_vars(bl, simple_avgs[evk])
            ho = atmos_core_ho_vars(bl, ho_sums[evk])
            atmos_core_ho_sums!(bl, state, thermo, MH, simple_ha, ho)
        end
    end

    # reduce across ranks and compute averages
    ho_avgs =
        [zeros(FT, num_atmos_core_ho_vars(bl, FT)) for _ in 1:(Nqk * nvertelem)]
    for evk in 1:(Nqk * nvertelem)
        MPI.Reduce!(ho_sums[evk], ho_avgs[evk], +, 0, mpicomm)
        if mpirank == 0
            ho_avgs[evk] .= ho_avgs[evk] ./ core_repdvsr[evk]
        end
    end

    # complete density averaging and prepare output
    if mpirank == 0
        varvals = OrderedDict()
        varvals["core_frac"] = (("z",), core_frac)

        for vari in 1:length(simple_varnames)
            davg = zeros(FT, Nqk * nvertelem)
            for evk in 1:(Nqk * nvertelem)
                davg[evk] = simple_avgs[evk][vari]
            end
            varvals[simple_varnames[vari]] = (("z",), davg)
        end

        ho_varnames = flattenednames(vars_atmos_core_ho(bl, FT))
        for vari in 1:length(ho_varnames)
            davg = zeros(FT, Nqk * nvertelem)
            for evk in 1:(Nqk * nvertelem)
                simple_ha = atmos_core_simple_vars(bl, simple_avgs[evk])
                avg_rho = simple_ha.avg_rho_core
                davg[evk] = ho_avgs[evk][vari] / avg_rho
            end
            varvals[ho_varnames[vari]] = (("z",), davg)
        end

        # write output
        dprefix = @sprintf(
            "%s_%s_%s_num%04d",
            dgngrp.out_prefix,
            dgngrp.name,
            Settings.starttime,
            dgngrp.num
        )
        dfilename = joinpath(Settings.output_dir, dprefix)
        write_data(
            dgngrp.writer,
            dfilename,
            OrderedDict("z" => zvals),
            varvals,
            currtime,
        )
    end

    MPI.Barrier(mpicomm)
    return nothing
end # function collect

function atmos_core_fini(dgngrp::DiagnosticsGroup, currtime) end
