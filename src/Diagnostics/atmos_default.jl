using ..Atmos
using ..Atmos: MoistureModel, thermo_state, turbulence_tensors
using ..Mesh.Topologies
using ..Mesh.Grids
using ..MoistThermodynamics
using LinearAlgebra

"""
    atmos_default_init(bl, currtime)

Initialize the 'AtmosDefault' diagnostics group.
"""
function atmos_default_init(dgngrp::DiagnosticsGroup, currtime)
    atmos_collect_onetime(Settings.mpicomm, Settings.dg, Settings.Q)

    return nothing
end

# Simple horizontal averages
function vars_atmos_default_simple(m::AtmosModel, FT)
    @vars begin
        u::FT
        v::FT
        w::FT
        avg_rho::FT             # ρ
        rho::FT                 # ρρ
        temp::FT
        thd::FT                 # θ_dry
        thv::FT                 # θ_vir
        et::FT                  # e_tot
        ei::FT                  # e_int
        ht::FT
        hm::FT
        w_ht_sgs::FT

        moisture::vars_atmos_default_simple(m.moisture, FT)
    end
end
vars_atmos_default_simple(::MoistureModel, FT) = @vars()
function vars_atmos_default_simple(m::EquilMoist, FT)
    @vars begin
        qt::FT                  # q_tot
        ql::FT                  # q_liq
        qv::FT                  # q_vap
        thl::FT                 # θ_liq
        w_qt_sgs::FT
    end
end
num_atmos_default_simple_vars(m, FT) = varsize(vars_atmos_default_simple(m, FT))
atmos_default_simple_vars(m, array) =
    Vars{vars_atmos_default_simple(m, eltype(array))}(array)

function atmos_default_simple_sums!(
    atmos::AtmosModel,
    state_conservative,
    state_gradient_flux,
    state_auxiliary,
    thermo,
    currtime,
    MH,
    sums,
)
    sums.u += MH * state_conservative.ρu[1]
    sums.v += MH * state_conservative.ρu[2]
    sums.w += MH * state_conservative.ρu[3]
    sums.avg_rho += MH * state_conservative.ρ
    sums.rho += MH * state_conservative.ρ * state_conservative.ρ
    sums.temp += MH * thermo.T * state_conservative.ρ
    sums.thd += MH * thermo.θ_dry * state_conservative.ρ
    sums.thv += MH * thermo.θ_vir * state_conservative.ρ
    sums.et += MH * state_conservative.ρe
    sums.ei += MH * thermo.e_int * state_conservative.ρ
    sums.ht += MH * thermo.h_tot * state_conservative.ρ
    sums.hm += MH * thermo.h_moi * state_conservative.ρ

    ν, D_t, _ = turbulence_tensors(
        atmos,
        state_conservative,
        state_gradient_flux,
        state_auxiliary,
        currtime,
    )
    d_h_tot = -D_t .* state_gradient_flux.∇h_tot
    sums.w_ht_sgs += MH * d_h_tot[end] * state_conservative.ρ

    atmos_default_simple_sums!(
        atmos.moisture,
        state_conservative,
        state_gradient_flux,
        thermo,
        MH,
        D_t,
        sums,
    )

    return nothing
end
function atmos_default_simple_sums!(
    ::MoistureModel,
    state_conservative,
    state_gradient_flux,
    thermo,
    MH,
    D_t,
    sums,
)
    return nothing
end
function atmos_default_simple_sums!(
    moist::EquilMoist,
    state_conservative,
    state_gradient_flux,
    thermo,
    MH,
    D_t,
    sums,
)
    sums.moisture.qt += MH * state_conservative.moisture.ρq_tot
    sums.moisture.ql += MH * thermo.moisture.q_liq * state_conservative.ρ
    sums.moisture.qv += MH * thermo.moisture.q_vap * state_conservative.ρ
    sums.moisture.thl += MH * thermo.moisture.θ_liq_ice * state_conservative.ρ
    d_q_tot = (-D_t) .* state_gradient_flux.moisture.∇q_tot
    sums.moisture.w_qt_sgs += MH * d_q_tot[end] * state_conservative.ρ

    return nothing
end

# Variances and covariances
function vars_atmos_default_ho(m::AtmosModel, FT)
    @vars begin
        var_u::FT               # u′u′
        var_v::FT               # v′v′
        var_w::FT               # w′w′
        w3::FT                  # w′w′w′
        tke::FT
        var_ei::FT              # e_int′e_int′

        cov_w_u::FT             # w′u′
        cov_w_v::FT             # w′v′
        cov_w_rho::FT           # w′ρ′
        cov_w_thd::FT           # w′θ_dry′
        cov_w_thv::FT           # w′θ_v′
        cov_w_ei::FT            # w′e_int′

        moisture::vars_atmos_default_ho(m.moisture, FT)
    end
end
vars_atmos_default_ho(::MoistureModel, FT) = @vars()
function vars_atmos_default_ho(m::EquilMoist, FT)
    @vars begin
        var_qt::FT              # q_tot′q_tot′
        var_thl::FT             # θ_liq_ice′θ_liq_ice′

        cov_w_qt::FT            # w′q_tot′
        cov_w_ql::FT            # w′q_liq′
        cov_w_qv::FT            # w′q_vap′
        cov_w_thl::FT           # w′θ_liq_ice′
        cov_qt_thl::FT          # q_tot′θ_liq_ice′
        cov_qt_ei::FT           # q_tot′e_int′
    end
end
num_atmos_default_ho_vars(m, FT) = varsize(vars_atmos_default_ho(m, FT))
atmos_default_ho_vars(m, array) =
    Vars{vars_atmos_default_ho(m, eltype(array))}(array)

function atmos_default_ho_sums!(
    atmos::AtmosModel,
    state_conservative,
    thermo,
    MH,
    ha,
    sums,
)
    u = state_conservative.ρu[1] / state_conservative.ρ
    u′ = u - ha.u
    v = state_conservative.ρu[2] / state_conservative.ρ
    v′ = v - ha.v
    w = state_conservative.ρu[3] / state_conservative.ρ
    w′ = w - ha.w
    e_int′ = thermo.e_int - ha.ei
    θ_dry′ = thermo.θ_dry - ha.thd
    θ_vir′ = thermo.θ_vir - ha.thv

    sums.var_u += MH * u′^2 * state_conservative.ρ
    sums.var_v += MH * v′^2 * state_conservative.ρ
    sums.var_w += MH * w′^2 * state_conservative.ρ
    sums.w3 += MH * w′^3 * state_conservative.ρ
    sums.tke +=
        0.5 * (
            MH * u′^2 * state_conservative.ρ +
            MH * v′^2 * state_conservative.ρ +
            MH * w′^2 * state_conservative.ρ
        )
    sums.var_ei += MH * e_int′^2 * state_conservative.ρ

    sums.cov_w_u += MH * w′ * u′ * state_conservative.ρ
    sums.cov_w_v += MH * w′ * v′ * state_conservative.ρ
    sums.cov_w_rho +=
        MH * w′ * (state_conservative.ρ - ha.avg_rho) * state_conservative.ρ
    sums.cov_w_thd += MH * w′ * θ_dry′ * state_conservative.ρ
    sums.cov_w_thv += MH * w′ * θ_vir′ * state_conservative.ρ
    sums.cov_w_ei += MH * w′ * e_int′ * state_conservative.ρ

    atmos_default_ho_sums!(
        atmos.moisture,
        state_conservative,
        thermo,
        MH,
        ha,
        w′,
        e_int′,
        sums,
    )

    return nothing
end
function atmos_default_ho_sums!(
    ::MoistureModel,
    state_conservative,
    thermo,
    MH,
    ha,
    w′,
    e_int′,
    sums,
)
    return nothing
end
function atmos_default_ho_sums!(
    moist::EquilMoist,
    state_conservative,
    thermo,
    MH,
    ha,
    w′,
    e_int′,
    sums,
)
    q_tot = state_conservative.moisture.ρq_tot / state_conservative.ρ
    q_tot′ = q_tot - ha.moisture.qt
    q_liq′ = thermo.moisture.q_liq - ha.moisture.ql
    q_vap′ = thermo.moisture.q_vap - ha.moisture.qv
    θ_liq_ice′ = thermo.moisture.θ_liq_ice - ha.moisture.thl

    sums.moisture.var_qt += MH * q_tot′^2 * state_conservative.ρ
    sums.moisture.var_thl += MH * θ_liq_ice′^2 * state_conservative.ρ

    sums.moisture.cov_w_qt += MH * w′ * q_tot′ * state_conservative.ρ
    sums.moisture.cov_w_ql += MH * w′ * q_liq′ * state_conservative.ρ
    sums.moisture.cov_w_qv += MH * w′ * q_vap′ * state_conservative.ρ
    sums.moisture.cov_w_thl += MH * w′ * θ_liq_ice′ * state_conservative.ρ
    sums.moisture.cov_qt_thl += MH * q_tot′ * θ_liq_ice′ * state_conservative.ρ
    sums.moisture.cov_qt_ei += MH * q_tot′ * e_int′ * state_conservative.ρ

    return nothing
end

"""
    atmos_default_collect(bl, currtime)

Collect the various 'AtmosDefault' diagnostic variables for the
current timestep and write them into a file.
"""
function atmos_default_collect(dgngrp::DiagnosticsGroup, currtime)
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    mpirank = MPI.Comm_rank(mpicomm)
    bl = dg.balance_law
    grid = dg.grid
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    npoints = Nq * Nq * Nqk
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    # get needed arrays onto the CPU
    if Array ∈ typeof(Q).parameters
        host_state_conservative = Q.realdata
        host_state_auxiliary = dg.state_auxiliary.realdata
        host_vgeo = grid.vgeo
        host_state_gradient_flux = dg.state_gradient_flux.realdata
    else
        host_state_conservative = Array(Q.realdata)
        host_state_auxiliary = Array(dg.state_auxiliary.realdata)
        host_vgeo = Array(grid.vgeo)
        host_state_gradient_flux = Array(dg.state_gradient_flux.realdata)
    end
    FT = eltype(host_state_conservative)

    zvals = AtmosCollected.zvals
    repdvsr = AtmosCollected.repdvsr

    # Visit each node of the state variables array and:
    # - generate and store the thermo variables,
    # - accumulate the simple horizontal sums, and
    # - determine the cloud fraction, top and base
    #
    thermo_array =
        [zeros(FT, num_thermo(bl, FT)) for _ in 1:npoints, _ in 1:nrealelem]
    simple_sums = [
        zeros(FT, num_atmos_default_simple_vars(bl, FT))
        for _ in 1:(Nqk * nvertelem)
    ]
    ql_gt_0_z = [zeros(FT, (Nq * Nq * nhorzelem)) for _ in 1:(Nqk * nvertelem)]
    ql_gt_0_full = zeros(FT, (Nq * Nq * nhorzelem))
    # In honor of PyCLES!
    cld_top = FT(-100000)
    cld_base = FT(100000)
    @visitQ nhorzelem nvertelem Nqk Nq begin
        evk = Nqk * (ev - 1) + k

        state_conservative =
            extract_state_conservative(dg, host_state_conservative, ijk, e)
        state_gradient_flux =
            extract_state_gradient_flux(dg, host_state_gradient_flux, ijk, e)
        state_auxiliary =
            extract_state_auxiliary(dg, host_state_auxiliary, ijk, e)
        MH = host_vgeo[ijk, grid.MHid, e]

        thermo = thermo_vars(bl, thermo_array[ijk, e])
        compute_thermo!(bl, state_conservative, state_auxiliary, thermo)

        simple = atmos_default_simple_vars(bl, simple_sums[evk])
        atmos_default_simple_sums!(
            bl,
            state_conservative,
            state_gradient_flux,
            state_auxiliary,
            thermo,
            currtime,
            MH,
            simple,
        )

        if !iszero(thermo.moisture.q_liq)
            idx = (Nq * Nq * (eh - 1)) + (Nq * (j - 1)) + i
            ql_gt_0_z[evk][idx] = one(FT)
            ql_gt_0_full[idx] = one(FT)

            z = zvals[evk]
            cld_top = max(cld_top, z)
            cld_base = min(cld_base, z)
        end
    end

    # reduce horizontal sums and cloud data across ranks and compute averages
    simple_avgs = [
        zeros(FT, num_atmos_default_simple_vars(bl, FT))
        for _ in 1:(Nqk * nvertelem)
    ]
    cld_frac = zeros(FT, Nqk * nvertelem)
    for evk in 1:(Nqk * nvertelem)
        MPI.Allreduce!(simple_sums[evk], simple_avgs[evk], +, mpicomm)
        simple_avgs[evk] .= simple_avgs[evk] ./ repdvsr[evk]

        tot_ql_gt_0_z = MPI.Reduce(sum(ql_gt_0_z[evk]), +, 0, mpicomm)
        tot_horz_z = MPI.Reduce(length(ql_gt_0_z[evk]), +, 0, mpicomm)
        if mpirank == 0
            cld_frac[evk] = tot_ql_gt_0_z / tot_horz_z
        end
    end
    cld_top = MPI.Reduce(cld_top, MPI.MAX, 0, mpicomm)
    if cld_top == FT(-100000)
        cld_top = NaN
    end
    cld_base = MPI.Reduce(cld_base, MPI.MIN, 0, mpicomm)
    if cld_base == FT(100000)
        cld_base = NaN
    end
    tot_ql_gt_0_full = MPI.Reduce(sum(ql_gt_0_full), +, 0, mpicomm)
    tot_horz_full = MPI.Reduce(length(ql_gt_0_full), +, 0, mpicomm)
    cld_cover = zero(FT)
    if mpirank == 0
        cld_cover = tot_ql_gt_0_full / tot_horz_full
    end

    # complete density averaging
    simple_varnames = map(
        s -> startswith(s, "moisture.") ? s[10:end] : s,
        flattenednames(vars_atmos_default_simple(bl, FT)),
    )
    for vari in 1:length(simple_varnames)
        for evk in 1:(Nqk * nvertelem)
            simple_ha = atmos_default_simple_vars(bl, simple_avgs[evk])
            avg_rho = simple_ha.avg_rho
            if simple_varnames[vari] != "avg_rho"
                simple_avgs[evk][vari] /= avg_rho
            end
        end
    end

    # compute the variances and covariances
    ho_sums = [
        zeros(FT, num_atmos_default_ho_vars(bl, FT))
        for _ in 1:(Nqk * nvertelem)
    ]
    @visitQ nhorzelem nvertelem Nqk Nq begin
        evk = Nqk * (ev - 1) + k

        state_conservative =
            extract_state_conservative(dg, host_state_conservative, ijk, e)
        thermo = thermo_vars(bl, thermo_array[ijk, e])
        MH = host_vgeo[ijk, grid.MHid, e]

        simple_ha = atmos_default_simple_vars(bl, simple_avgs[evk])
        ho = atmos_default_ho_vars(bl, ho_sums[evk])
        atmos_default_ho_sums!(
            bl,
            state_conservative,
            thermo,
            MH,
            simple_ha,
            ho,
        )
    end

    # reduce across ranks and compute averages
    ho_avgs = [
        zeros(FT, num_atmos_default_ho_vars(bl, FT))
        for _ in 1:(Nqk * nvertelem)
    ]
    for evk in 1:(Nqk * nvertelem)
        MPI.Reduce!(ho_sums[evk], ho_avgs[evk], +, 0, mpicomm)
        if mpirank == 0
            ho_avgs[evk] .= ho_avgs[evk] ./ repdvsr[evk]
        end
    end

    # complete density averaging and prepare output
    if mpirank == 0
        varvals = OrderedDict()
        for vari in 1:length(simple_varnames)
            davg = zeros(FT, Nqk * nvertelem)
            for evk in 1:(Nqk * nvertelem)
                davg[evk] = simple_avgs[evk][vari]
            end
            varvals[simple_varnames[vari]] = (("z",), davg)
        end

        ho_varnames = map(
            s -> startswith(s, "moisture.") ? s[10:end] : s,
            flattenednames(vars_atmos_default_ho(bl, FT)),
        )
        for vari in 1:length(ho_varnames)
            davg = zeros(FT, Nqk * nvertelem)
            for evk in 1:(Nqk * nvertelem)
                simple_ha = atmos_default_simple_vars(bl, simple_avgs[evk])
                avg_rho = simple_ha.avg_rho
                davg[evk] = ho_avgs[evk][vari] / avg_rho
            end
            varvals[ho_varnames[vari]] = (("z",), davg)
        end

        varvals["cld_frac"] = (("z",), cld_frac)
        varvals["cld_top"] = (("t",), cld_top)
        varvals["cld_base"] = (("t",), cld_base)
        varvals["cld_cover"] = (("t",), cld_cover)

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

function atmos_default_fini(dgngrp::DiagnosticsGroup, currtime) end
