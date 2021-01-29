using ..Atmos
using ..Atmos: MoistureModel, PrecipitationModel, recover_thermo_state
using ..Mesh.Topologies
using ..Mesh.Grids
using ..Thermodynamics
using ..TurbulenceClosures
using LinearAlgebra

"""
    setup_atmos_default_diagnostics(
        ::AtmosLESConfigType,
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
    )

Create the "AtmosLESDefault" `DiagnosticsGroup` which contains the following
diagnostic variables, all of which are density-averaged horizontal averages,
variances and co-variances:

- u: x-velocity
- v: y-velocity
- w: z-velocity
- avg_rho: air density (_not_ density-averaged)
- rho: air density
- temp: air temperature
- pres: air pressure
- thd: dry potential temperature
- et: total specific energy
- ei: specific internal energy
- ht: specific enthalpy based on total energy
- hi: specific enthalpy based on internal energy
- w_ht_sgs: vertical sgs flux of total specific enthalpy
- var_u: variance of x-velocity
- var_v: variance of y-velocity
- var_w: variance of z-velocity
- w3: third moment of z-velocity
- tke: turbulent kinetic energy
- var_ei: variance of specific internal energy
- cov_w_u: vertical eddy flux of x-velocity
- cov_w_v: vertical eddy flux of y-velocity
- cov_w_rho: vertical eddy flux of density
- cov_w_thd: vertical eddy flux of dry potential temperature
- cov_w_ei: vertical eddy flux of specific internal energy

When an `EquilMoist` or a `NonEquilMoist` moisture model is used, the following
diagnostic variables are also output, also density-averaged horizontal averages,
variances and co-variances:

- qt: mass fraction of total water in air
- ql: mass fraction of liquid water in air
- qv: mass fraction of water vapor in air
- qi: mass fraction of ice in air
- thv: virtual potential temperature
- thl: liquid-ice potential temperature
- w_qt_sgs: vertical sgs flux of total specific humidity
- var_qt: variance of total specific humidity
- var_thl: variance of liquid-ice potential temperature
- cov_w_qt: vertical eddy flux of total specific humidity
- cov_w_ql: vertical eddy flux of liquid water specific humidity
- cov_w_qi: vertical eddy flux of cloud ice specific humidity
- cov_w_qv: vertical eddy flux of water vapor specific humidity
- cov_w_thv: vertical eddy flux of virtual potential temperature
- cov_w_thl: vertical eddy flux of liquid-ice potential temperature
- cov_qt_thl: covariance of total specific humidity and liquid-ice potential temperature
- cov_qt_ei: covariance of total specific humidity and specific internal energy

All these variables are output with the `z` dimension (`x3id`) on the DG grid
(`interpol` may _not_ be specified) as well as a (unlimited) `time` dimension
at the specified `interval`.
"""
function setup_atmos_default_diagnostics(
    ::AtmosLESConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
    # TODO: remove this
    @assert isnothing(interpol)

    return DiagnosticsGroup(
        "AtmosLESDefault",
        Diagnostics.atmos_les_default_init,
        Diagnostics.atmos_les_default_fini,
        Diagnostics.atmos_les_default_collect,
        interval,
        out_prefix,
        writer,
        interpol,
    )
end

# Simple horizontal averages
function vars_atmos_les_default_simple(m::AtmosModel, FT)
    @vars begin
        u::FT
        v::FT
        w::FT
        avg_rho::FT             # ρ
        rho::FT                 # ρρ
        temp::FT
        pres::FT
        thd::FT                 # θ_dry
        et::FT                  # e_tot
        ei::FT                  # e_int
        ht::FT
        hi::FT
        w_ht_sgs::FT

        moisture::vars_atmos_les_default_simple(m.moisture, FT)
        precipitation::vars_atmos_les_default_simple(m.precipitation, FT)
    end
end
vars_atmos_les_default_simple(::MoistureModel, FT) = @vars()
function vars_atmos_les_default_simple(m::Union{EquilMoist, NonEquilMoist}, FT)
    @vars begin
        qt::FT                  # q_tot
        ql::FT                  # q_liq
        qi::FT                  # q_ice
        qv::FT                  # q_vap
        thv::FT                 # θ_vir
        thl::FT                 # θ_liq
        w_qt_sgs::FT
    end
end
vars_atmos_les_default_simple(::PrecipitationModel, FT) = @vars()
function vars_atmos_les_default_simple(::RainModel, FT)
    @vars begin
        qr::FT                  # q_rai
    end
end
function vars_atmos_les_default_simple(::RainSnowModel, FT)
    @vars begin
        qr::FT                  # q_rai
        qs::FT                  # q_sno
    end
end
num_atmos_les_default_simple_vars(m, FT) =
    varsize(vars_atmos_les_default_simple(m, FT))
atmos_les_default_simple_vars(m, array) =
    Vars{vars_atmos_les_default_simple(m, eltype(array))}(array)

function atmos_les_default_simple_sums!(
    atmos::AtmosModel,
    state,
    gradflux,
    aux,
    thermo,
    currtime,
    MH,
    sums,
)
    sums.u += MH * state.ρu[1]
    sums.v += MH * state.ρu[2]
    sums.w += MH * state.ρu[3]
    sums.avg_rho += MH * state.ρ
    sums.rho += MH * state.ρ * state.ρ
    sums.temp += MH * thermo.temp * state.ρ
    sums.pres += MH * thermo.pres * state.ρ
    sums.thd += MH * thermo.θ_dry * state.ρ
    sums.et += MH * state.energy.ρe
    sums.ei += MH * thermo.e_int * state.ρ
    sums.ht += MH * thermo.h_tot * state.ρ
    sums.hi += MH * thermo.h_int * state.ρ

    ν, D_t, _ = turbulence_tensors(atmos, state, gradflux, aux, currtime)
    d_h_tot = -D_t .* gradflux.energy.∇h_tot
    sums.w_ht_sgs += MH * d_h_tot[end] * state.ρ

    atmos_les_default_simple_sums!(
        atmos.moisture,
        state,
        gradflux,
        thermo,
        MH,
        D_t,
        sums,
    )
    atmos_les_default_simple_sums!(
        atmos.precipitation,
        state,
        gradflux,
        thermo,
        MH,
        D_t,
        sums,
    )
    return nothing
end
function atmos_les_default_simple_sums!(
    ::MoistureModel,
    state,
    gradflux,
    thermo,
    MH,
    D_t,
    sums,
)
    return nothing
end
function atmos_les_default_simple_sums!(
    moist::Union{EquilMoist, NonEquilMoist},
    state,
    gradflux,
    thermo,
    MH,
    D_t,
    sums,
)
    sums.moisture.qt += MH * state.moisture.ρq_tot
    sums.moisture.ql += MH * thermo.moisture.q_liq * state.ρ
    sums.moisture.qi += MH * thermo.moisture.q_ice * state.ρ
    sums.moisture.qv += MH * thermo.moisture.q_vap * state.ρ
    sums.moisture.thv += MH * thermo.moisture.θ_vir * state.ρ
    sums.moisture.thl += MH * thermo.moisture.θ_liq_ice * state.ρ
    d_q_tot = (-D_t) .* gradflux.moisture.∇q_tot
    sums.moisture.w_qt_sgs += MH * d_q_tot[end] * state.ρ

    return nothing
end
function atmos_les_default_simple_sums!(
    ::PrecipitationModel,
    state,
    gradflux,
    thermo,
    MH,
    D_t,
    sums,
)
    return nothing
end
function atmos_les_default_simple_sums!(
    precipitation::RainModel,
    state,
    gradflux,
    thermo,
    MH,
    D_t,
    sums,
)
    sums.precipitation.qr += MH * state.precipitation.ρq_rai

    return nothing
end
function atmos_les_default_simple_sums!(
    precipitation::RainSnowModel,
    state,
    gradflux,
    thermo,
    MH,
    D_t,
    sums,
)
    sums.precipitation.qr += MH * state.precipitation.ρq_rai
    sums.precipitation.qs += MH * state.precipitation.ρq_sno

    return nothing
end

function atmos_les_default_clouds(
    ::MoistureModel,
    thermo,
    idx,
    qc_gt_0_z,
    qc_gt_0_full,
    z,
    cld_top,
    cld_base,
)
    return cld_top, cld_base
end
function atmos_les_default_clouds(
    moist::Union{EquilMoist, NonEquilMoist},
    thermo,
    idx,
    qc_gt_0_z,
    qc_gt_0_full,
    z,
    cld_top,
    cld_base,
)
    if thermo.moisture.has_condensate
        FT = eltype(qc_gt_0_z)
        qc_gt_0_z[idx] = one(FT)
        qc_gt_0_full[idx] = one(FT)

        cld_top = max(cld_top, z)
        cld_base = min(cld_base, z)
    end
    return cld_top, cld_base
end

# Variances and covariances
function vars_atmos_les_default_ho(m::AtmosModel, FT)
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
        cov_w_ei::FT            # w′e_int′

        moisture::vars_atmos_les_default_ho(m.moisture, FT)
        precipitation::vars_atmos_les_default_ho(m.precipitation, FT)
    end
end
vars_atmos_les_default_ho(::MoistureModel, FT) = @vars()
function vars_atmos_les_default_ho(m::Union{EquilMoist, NonEquilMoist}, FT)
    @vars begin
        var_qt::FT              # q_tot′q_tot′
        var_thl::FT             # θ_liq_ice′θ_liq_ice′

        cov_w_qt::FT            # w′q_tot′
        cov_w_ql::FT            # w′q_liq′
        cov_w_qi::FT            # w′q_ice′
        cov_w_qv::FT            # w′q_vap′
        cov_w_thv::FT           # w′θ_v′
        cov_w_thl::FT           # w′θ_liq_ice′
        cov_qt_thl::FT          # q_tot′θ_liq_ice′
        cov_qt_ei::FT           # q_tot′e_int′
    end
end
vars_atmos_les_default_ho(::PrecipitationModel, FT) = @vars()
function vars_atmos_les_default_ho(m::RainModel, FT)
    @vars begin
        var_qr::FT              # q_rai′q_rai′
        cov_w_qr::FT            # w′q_rai′
    end
end
function vars_atmos_les_default_ho(m::RainSnowModel, FT)
    @vars begin
        var_qr::FT              # q_rai′q_rai′
        var_qs::FT              # q_sno′q_sno′
        cov_w_qr::FT            # w′q_rai′
        cov_w_qs::FT            # w′q_sno′
    end
end
num_atmos_les_default_ho_vars(m, FT) = varsize(vars_atmos_les_default_ho(m, FT))
atmos_les_default_ho_vars(m, array) =
    Vars{vars_atmos_les_default_ho(m, eltype(array))}(array)

function atmos_les_default_ho_sums!(
    atmos::AtmosModel,
    state,
    thermo,
    MH,
    ha,
    sums,
)
    u = state.ρu[1] / state.ρ
    u′ = u - ha.u
    v = state.ρu[2] / state.ρ
    v′ = v - ha.v
    w = state.ρu[3] / state.ρ
    w′ = w - ha.w
    e_int′ = thermo.e_int - ha.ei
    θ_dry′ = thermo.θ_dry - ha.thd

    sums.var_u += MH * u′^2 * state.ρ
    sums.var_v += MH * v′^2 * state.ρ
    sums.var_w += MH * w′^2 * state.ρ
    sums.w3 += MH * w′^3 * state.ρ
    sums.tke +=
        0.5 * (MH * u′^2 * state.ρ + MH * v′^2 * state.ρ + MH * w′^2 * state.ρ)
    sums.var_ei += MH * e_int′^2 * state.ρ

    sums.cov_w_u += MH * w′ * u′ * state.ρ
    sums.cov_w_v += MH * w′ * v′ * state.ρ
    sums.cov_w_rho += MH * w′ * (state.ρ - ha.avg_rho) * state.ρ
    sums.cov_w_thd += MH * w′ * θ_dry′ * state.ρ
    sums.cov_w_ei += MH * w′ * e_int′ * state.ρ

    atmos_les_default_ho_sums!(
        atmos.moisture,
        state,
        thermo,
        MH,
        ha,
        w′,
        e_int′,
        sums,
    )
    atmos_les_default_ho_sums!(
        atmos.precipitation,
        state,
        thermo,
        MH,
        ha,
        w′,
        e_int′,
        sums,
    )
    return nothing
end
function atmos_les_default_ho_sums!(
    ::MoistureModel,
    state,
    thermo,
    MH,
    ha,
    w′,
    e_int′,
    sums,
)
    return nothing
end
function atmos_les_default_ho_sums!(
    moist::Union{EquilMoist, NonEquilMoist},
    state,
    thermo,
    MH,
    ha,
    w′,
    e_int′,
    sums,
)
    q_tot = state.moisture.ρq_tot / state.ρ
    q_tot′ = q_tot - ha.moisture.qt
    q_liq′ = thermo.moisture.q_liq - ha.moisture.ql
    q_ice′ = thermo.moisture.q_ice - ha.moisture.qi
    q_vap′ = thermo.moisture.q_vap - ha.moisture.qv
    θ_vir′ = thermo.moisture.θ_vir - ha.moisture.thv
    θ_liq_ice′ = thermo.moisture.θ_liq_ice - ha.moisture.thl

    sums.moisture.var_qt += MH * q_tot′^2 * state.ρ
    sums.moisture.var_thl += MH * θ_liq_ice′^2 * state.ρ

    sums.moisture.cov_w_qt += MH * w′ * q_tot′ * state.ρ
    sums.moisture.cov_w_ql += MH * w′ * q_liq′ * state.ρ
    sums.moisture.cov_w_qi += MH * w′ * q_ice′ * state.ρ
    sums.moisture.cov_w_qv += MH * w′ * q_vap′ * state.ρ
    sums.moisture.cov_w_thv += MH * w′ * θ_vir′ * state.ρ
    sums.moisture.cov_w_thl += MH * w′ * θ_liq_ice′ * state.ρ
    sums.moisture.cov_qt_thl += MH * q_tot′ * θ_liq_ice′ * state.ρ
    sums.moisture.cov_qt_ei += MH * q_tot′ * e_int′ * state.ρ

    return nothing
end
function atmos_les_default_ho_sums!(
    ::PrecipitationModel,
    state,
    thermo,
    MH,
    ha,
    w′,
    e_int′,
    sums,
)
    return nothing
end
function atmos_les_default_ho_sums!(
    moist::RainModel,
    state,
    thermo,
    MH,
    ha,
    w′,
    e_int′,
    sums,
)
    q_rai = state.precipitation.ρq_rai / state.ρ
    q_rai′ = q_rai - ha.precipitation.qr

    sums.precipitation.var_qr += MH * q_rai′^2 * state.ρ

    sums.precipitation.cov_w_qr += MH * w′ * q_rai′ * state.ρ

    return nothing
end
function atmos_les_default_ho_sums!(
    moist::RainSnowModel,
    state,
    thermo,
    MH,
    ha,
    w′,
    e_int′,
    sums,
)
    q_rai = state.precipitation.ρq_rai / state.ρ
    q_rai′ = q_rai - ha.precipitation.qr
    q_sno = state.precipitation.ρq_sno / state.ρ
    q_sno′ = q_sno - ha.precipitation.qs

    sums.precipitation.var_qr += MH * q_rai′^2 * state.ρ
    sums.precipitation.var_qs += MH * q_sno′^2 * state.ρ

    sums.precipitation.cov_w_qr += MH * w′ * q_rai′ * state.ρ
    sums.precipitation.cov_w_qs += MH * w′ * q_sno′ * state.ρ

    return nothing
end

function prefix_filter(s)
    if startswith(s, "moisture.")
        s[10:end]
    elseif startswith(s, "precipitation.")
        s[15:end]
    else
        s
    end
end

"""
    atmos_les_default_init(dgngrp, currtime)

Initialize the 'AtmosLESDefault' diagnostics group.
"""
function atmos_les_default_init(dgngrp::DiagnosticsGroup, currtime)
    atmos = Settings.dg.balance_law
    FT = eltype(Settings.Q)
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)

    atmos_collect_onetime(mpicomm, Settings.dg, Settings.Q)

    if mpirank == 0
        dims = OrderedDict("z" => (AtmosCollected.zvals, Dict()))

        # set up the variables we're going to be writing
        vars = OrderedDict()
        varnames = map(
            prefix_filter,
            flattenednames(vars_atmos_les_default_simple(atmos, FT)),
        )
        ho_varnames = map(
            prefix_filter,
            flattenednames(vars_atmos_les_default_ho(atmos, FT)),
        )
        append!(varnames, ho_varnames)
        for varname in varnames
            var = Variables[varname]
            vars[varname] = (("z",), FT, var.attrib)
        end
        vars["cld_frac"] = (("z",), FT, Variables["cld_frac"].attrib)
        vars["cld_top"] = ((), FT, Variables["cld_top"].attrib)
        vars["cld_base"] = ((), FT, Variables["cld_base"].attrib)
        vars["cld_cover"] = ((), FT, Variables["cld_cover"].attrib)
        vars["lwp"] = ((), FT, Variables["lwp"].attrib)
        vars["iwp"] = ((), FT, Variables["iwp"].attrib)
        vars["rwp"] = ((), FT, Variables["rwp"].attrib)
        vars["swp"] = ((), FT, Variables["swp"].attrib)

        # create the output file
        dprefix = @sprintf(
            "%s_%s_%s",
            dgngrp.out_prefix,
            dgngrp.name,
            Settings.starttime,
        )
        dfilename = joinpath(Settings.output_dir, dprefix)
        init_data(dgngrp.writer, dfilename, dims, vars)
    end

    return nothing
end

"""
    atmos_les_default_collect(dgngrp, currtime)

Collect the various 'AtmosLESDefault' diagnostic variables for the
current timestep and write them into a file.
"""
function atmos_les_default_collect(dgngrp::DiagnosticsGroup, currtime)
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    mpirank = MPI.Comm_rank(mpicomm)
    atmos = dg.balance_law
    grid = dg.grid
    grid_info = basic_grid_info(dg)
    topl_info = basic_topology_info(grid.topology)
    Nqk = grid_info.Nqk
    Nqh = grid_info.Nqh
    npoints = prod(grid_info.Nq)
    nrealelem = topl_info.nrealelem
    nvertelem = topl_info.nvertelem
    nhorzelem = topl_info.nhorzrealelem

    atmos.energy isa EnergyModel || error("EnergyModel only supported")

    # get needed arrays onto the CPU
    if array_device(Q) isa CPU
        state_data = Q.realdata
        aux_data = dg.state_auxiliary.realdata
        vgeo = grid.vgeo
        # ω is the weight vector for the vertical direction
        ω = grid.ω[end]
        gradflux_data = dg.state_gradient_flux.realdata
    else
        state_data = Array(Q.realdata)
        aux_data = Array(dg.state_auxiliary.realdata)
        vgeo = Array(grid.vgeo)
        # ω is the weight vector for the vertical direction
        ω = Array(grid.ω[end])
        gradflux_data = Array(dg.state_gradient_flux.realdata)
    end
    FT = eltype(state_data)

    zvals = AtmosCollected.zvals
    MH_z = AtmosCollected.MH_z

    # Visit each node of the state variables array and:
    # - generate and store the thermo variables,
    # - accumulate the simple horizontal sums, and
    # - determine the cloud fraction, top and base
    #
    thermo_array =
        [zeros(FT, num_thermo(atmos, FT)) for _ in 1:npoints, _ in 1:nrealelem]
    simple_sums = [
        zeros(FT, num_atmos_les_default_simple_vars(atmos, FT))
        for _ in 1:(Nqk * nvertelem)
    ]
    # for liquid, ice, rain and snow water paths
    ρq_liq_z = [zero(FT) for _ in 1:(Nqk * nvertelem)]
    ρq_ice_z = [zero(FT) for _ in 1:(Nqk * nvertelem)]
    ρq_rai_z = [zero(FT) for _ in 1:(Nqk * nvertelem)]
    ρq_sno_z = [zero(FT) for _ in 1:(Nqk * nvertelem)]
    # for cld*
    qc_gt_0_z = [zeros(FT, (Nqh * nhorzelem)) for _ in 1:(Nqk * nvertelem)]
    qc_gt_0_full = zeros(FT, (Nqh * nhorzelem))
    # In honor of PyCLES!
    cld_top = FT(-100000)
    cld_base = FT(100000)
    @traverse_dg_grid grid_info topl_info begin
        state = extract_state(dg, state_data, ijk, e, Prognostic())
        gradflux = extract_state(dg, gradflux_data, ijk, e, GradientFlux())
        aux = extract_state(dg, aux_data, ijk, e, Auxiliary())
        MH = vgeo[ijk, grid.MHid, e]

        thermo = thermo_vars(atmos, thermo_array[ijk, e])
        compute_thermo!(atmos, state, aux, thermo)

        simple = atmos_les_default_simple_vars(atmos, simple_sums[evk])
        atmos_les_default_simple_sums!(
            atmos,
            state,
            gradflux,
            aux,
            thermo,
            currtime,
            MH,
            simple,
        )

        idx = (Nqh * (eh - 1)) + (grid_info.Nq[2] * (j - 1)) + i
        cld_top, cld_base = atmos_les_default_clouds(
            atmos.moisture,
            thermo,
            idx,
            qc_gt_0_z[evk],
            qc_gt_0_full,
            zvals[evk],
            cld_top,
            cld_base,
        )

        # FIXME properly
        if isa(atmos.moisture, EquilMoist) || isa(atmos.moisture, NonEquilMoist)
            # for LWP
            ρq_liq_z[evk] += MH * thermo.moisture.q_liq * state.ρ * state.ρ
            ρq_ice_z[evk] += MH * thermo.moisture.q_ice * state.ρ * state.ρ
        end
        if isa(atmos.precipitation, RainModel)
            # for RWP
            ρq_rai_z[evk] += MH * state.precipitation.ρq_rai * state.ρ
        end
        if isa(atmos.precipitation, RainSnowModel)
            # for RWP
            ρq_rai_z[evk] += MH * state.precipitation.ρq_rai * state.ρ
            # for SWP
            ρq_sno_z[evk] += MH * state.precipitation.ρq_sno * state.ρ
        end
    end

    # reduce horizontal sums and cloud data across ranks and compute averages
    simple_avgs = [
        zeros(FT, num_atmos_les_default_simple_vars(atmos, FT))
        for _ in 1:(Nqk * nvertelem)
    ]
    cld_frac = zeros(FT, Nqk * nvertelem)
    for evk in 1:(Nqk * nvertelem)
        MPI.Allreduce!(simple_sums[evk], simple_avgs[evk], +, mpicomm)
        simple_avgs[evk] .= simple_avgs[evk] ./ MH_z[evk]

        # FIXME properly
        if isa(atmos.moisture, EquilMoist) || isa(atmos.moisture, NonEquilMoist)
            tot_qc_gt_0_z = MPI.Reduce(sum(qc_gt_0_z[evk]), +, 0, mpicomm)
            tot_horz_z = MPI.Reduce(length(qc_gt_0_z[evk]), +, 0, mpicomm)
            if mpirank == 0
                cld_frac[evk] = tot_qc_gt_0_z / tot_horz_z
            end

            # for LWP and IWP
            tot_ρq_liq_z = MPI.Reduce(ρq_liq_z[evk], +, 0, mpicomm)
            tot_ρq_ice_z = MPI.Reduce(ρq_ice_z[evk], +, 0, mpicomm)
            if mpirank == 0
                ρq_liq_z[evk] = tot_ρq_liq_z / MH_z[evk]
                ρq_ice_z[evk] = tot_ρq_ice_z / MH_z[evk]
            end
        end
        if isa(atmos.precipitation, RainModel)
            # for RWP
            tot_ρq_rai_z = MPI.Reduce(ρq_rai_z[evk], +, 0, mpicomm)
            if mpirank == 0
                ρq_rai_z[evk] = tot_ρq_rai_z / MH_z[evk]
            end
        end
        if isa(atmos.precipitation, RainSnowModel)
            # for RWP and SWP
            tot_ρq_rai_z = MPI.Reduce(ρq_rai_z[evk], +, 0, mpicomm)
            tot_ρq_sno_z = MPI.Reduce(ρq_sno_z[evk], +, 0, mpicomm)
            if mpirank == 0
                ρq_rai_z[evk] = tot_ρq_rai_z / MH_z[evk]
                ρq_sno_z[evk] = tot_ρq_sno_z / MH_z[evk]
            end
        end
    end
    # FIXME properly
    if isa(atmos.moisture, EquilMoist) || isa(atmos.moisture, NonEquilMoist)
        cld_top = MPI.Reduce(cld_top, max, 0, mpicomm)
        if cld_top == FT(-100000)
            cld_top = NaN
        end
        cld_base = MPI.Reduce(cld_base, min, 0, mpicomm)
        if cld_base == FT(100000)
            cld_base = NaN
        end
        tot_qc_gt_0_full = MPI.Reduce(sum(qc_gt_0_full), +, 0, mpicomm)
        tot_horz_full = MPI.Reduce(length(qc_gt_0_full), +, 0, mpicomm)
        cld_cover = zero(FT)
        if mpirank == 0
            cld_cover = tot_qc_gt_0_full / tot_horz_full
        end
    end

    simple_varnames = map(
        prefix_filter,
        flattenednames(vars_atmos_les_default_simple(atmos, FT)),
    )

    # complete density averaging
    for evk in 1:(Nqk * nvertelem)
        simple_ha = atmos_les_default_simple_vars(atmos, simple_avgs[evk])
        avg_rho = simple_ha.avg_rho
        for vari in 1:length(simple_varnames)
            if simple_varnames[vari] != "avg_rho"
                simple_avgs[evk][vari] /= avg_rho
            end
        end

        # for all the water paths
        # FIXME properly
        if isa(atmos.moisture, EquilMoist) || isa(atmos.moisture, NonEquilMoist)
            ρq_liq_z[evk] /= avg_rho
            ρq_ice_z[evk] /= avg_rho
        end
        if isa(atmos.precipitation, RainModel)
            ρq_rai_z[evk] /= avg_rho
        end
        if isa(atmos.precipitation, RainSnowModel)
            ρq_rai_z[evk] /= avg_rho
            ρq_sno_z[evk] /= avg_rho
        end
    end

    # compute all the water paths
    lwp = NaN
    iwp = NaN
    rwp = NaN
    swp = NaN
    if mpirank == 0
        JcV = reshape(
            view(vgeo, :, grid.JcVid, grid.topology.realelems),
            Nqh,
            Nqk,
            nvertelem,
            nhorzelem,
        )
        Mvert = (ω .* JcV[1, :, :, 1])[:]
        lwp = FT(sum(ρq_liq_z .* Mvert))
        iwp = FT(sum(ρq_ice_z .* Mvert))
        rwp = FT(sum(ρq_rai_z .* Mvert))
        swp = FT(sum(ρq_sno_z .* Mvert))
    end

    # compute the variances and covariances
    ho_sums = [
        zeros(FT, num_atmos_les_default_ho_vars(atmos, FT))
        for _ in 1:(Nqk * nvertelem)
    ]
    @traverse_dg_grid grid_info topl_info begin
        state = extract_state(dg, state_data, ijk, e, Prognostic())
        thermo = thermo_vars(atmos, thermo_array[ijk, e])
        MH = vgeo[ijk, grid.MHid, e]

        simple_ha = atmos_les_default_simple_vars(atmos, simple_avgs[evk])
        ho = atmos_les_default_ho_vars(atmos, ho_sums[evk])
        atmos_les_default_ho_sums!(atmos, state, thermo, MH, simple_ha, ho)
    end

    # reduce across ranks and compute averages
    ho_avgs = [
        zeros(FT, num_atmos_les_default_ho_vars(atmos, FT))
        for _ in 1:(Nqk * nvertelem)
    ]
    for evk in 1:(Nqk * nvertelem)
        MPI.Reduce!(ho_sums[evk], ho_avgs[evk], +, 0, mpicomm)
        if mpirank == 0
            ho_avgs[evk] .= ho_avgs[evk] ./ MH_z[evk]
        end
    end

    # complete density averaging and prepare output
    if mpirank == 0
        varvals = OrderedDict()
        for (vari, varname) in enumerate(simple_varnames)
            davg = zeros(FT, Nqk * nvertelem)
            for evk in 1:(Nqk * nvertelem)
                davg[evk] = simple_avgs[evk][vari]
            end
            varvals[varname] = davg
        end

        ho_varnames = map(
            prefix_filter,
            flattenednames(vars_atmos_les_default_ho(atmos, FT)),
        )
        for (vari, varname) in enumerate(ho_varnames)
            davg = zeros(FT, Nqk * nvertelem)
            for evk in 1:(Nqk * nvertelem)
                simple_ha =
                    atmos_les_default_simple_vars(atmos, simple_avgs[evk])
                avg_rho = simple_ha.avg_rho
                davg[evk] = ho_avgs[evk][vari] / avg_rho
            end
            varvals[varname] = davg
        end

        if isa(atmos.moisture, EquilMoist) || isa(atmos.moisture, NonEquilMoist)
            varvals["cld_frac"] = cld_frac
            varvals["cld_top"] = cld_top
            varvals["cld_base"] = cld_base
            varvals["cld_cover"] = cld_cover
            varvals["lwp"] = lwp
            varvals["iwp"] = iwp
        end
        if isa(atmos.precipitation, RainModel)
            varvals["rwp"] = rwp
        end
        if isa(atmos.precipitation, RainSnowModel)
            varvals["rwp"] = rwp
            varvals["swp"] = swp
        end

        # write output
        append_data(dgngrp.writer, varvals, currtime)
    end

    MPI.Barrier(mpicomm)
    return nothing
end # function collect

function atmos_les_default_fini(dgngrp::DiagnosticsGroup, currtime) end
