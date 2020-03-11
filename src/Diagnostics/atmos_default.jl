using ..Atmos
using ..Atmos: thermo_state, turbulence_tensors
using ..Mesh.Topologies
using ..Mesh.Grids
using ..SubgridScaleParameters: inv_Pr_turb
using ..PlanetParameters
using ..MoistThermodynamics
using LinearAlgebra

Base.@kwdef mutable struct AtmosCollectedDiagnostics
    zvals::Union{Nothing, Array} = nothing
    repdvsr::Union{Nothing, Array} = nothing
end
const CollectedDiagnostics = AtmosCollectedDiagnostics()

include("diagnostic_vars.jl")

function atmos_default_init(dgngrp::DiagnosticsGroup, currtime)
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    FT = eltype(Q)
    grid = dg.grid
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    if Array ∈ typeof(Q).parameters
        localvgeo = grid.vgeo
    else
        localvgeo = Array(grid.vgeo)
    end

    CollectedDiagnostics.zvals = zeros(FT, Nqk * nvertelem)
    CollectedDiagnostics.repdvsr = zeros(FT, Nqk * nvertelem)

    @visitQ nhorzelem nvertelem Nqk Nq begin
        z = localvgeo[ijk, grid.x3id, e]
        MH = localvgeo[ijk, grid.MHid, e]
        CollectedDiagnostics.zvals[Nqk * (ev - 1) + k] += MH * z
        CollectedDiagnostics.repdvsr[Nqk * (ev - 1) + k] += MH
    end

    # compute the full number of points on a slab
    MPI.Allreduce!(CollectedDiagnostics.repdvsr, +, mpicomm)

    CollectedDiagnostics.zvals ./= CollectedDiagnostics.repdvsr
end

# thermodynamic variables of interest
function vars_thermo(FT)
    @vars begin
        q_liq::FT
        q_ice::FT
        q_vap::FT
        T::FT
        θ_liq_ice::FT
        θ_dry::FT
        θ_v::FT
        e_int::FT
        h_m::FT
        h_t::FT
    end
end
num_thermo(FT) = varsize(vars_thermo(FT))
thermo_vars(array) = Vars{vars_thermo(eltype(array))}(array)

function compute_thermo!(bl, state, aux, ijk, e, thermoQ)
    e_tot = state.ρe / state.ρ
    ts = thermo_state(bl, state, aux)
    e_int = internal_energy(ts)
    Phpart = PhasePartition(ts)

    th = thermo_vars(thermoQ[ijk, e])
    th.q_liq = Phpart.liq
    th.q_ice = Phpart.ice
    th.q_vap = vapor_specific_humidity(ts)
    th.T = air_temperature(ts)
    th.θ_liq_ice = liquid_ice_pottemp(ts)
    th.θ_dry = dry_pottemp(ts)
    th.θ_v = virtual_pottemp(ts)
    th.e_int = e_int

    # Moist and total henthalpy
    R_m = gas_constant_air(ts)
    th.h_m = e_int + R_m * th.T
    th.h_t = e_tot + R_m * th.T

    return nothing
end

# horizontal averages
function vars_horzavg(FT)
    @vars begin
        ρ::FT
        ρu::FT
        ρv::FT
        ρw::FT
        e_tot::FT
        ρq_tot::FT
        q_liq::FT
        q_vap::FT
        θ_liq_ice::FT
        θ_dry::FT
        θ_v::FT
        e_int::FT
        h_m::FT
        h_t::FT
        qt_sgs::FT
        ht_sgs::FT
    end
end
num_horzavg(FT) = varsize(vars_horzavg(FT))
horzavg_vars(array) = Vars{vars_horzavg(eltype(array))}(array)

function compute_horzsums!(
    atmos::AtmosModel,
    state,
    diffusive_flux,
    aux,
    k,
    ijk,
    ev,
    e,
    Nqk,
    nvertelem,
    MH,
    localaux,
    thermoQ,
    horzsums,
    LWP,
    currtime,
)
    th = thermo_vars(thermoQ[ijk, e])
    hs = horzavg_vars(horzsums[Nqk * (ev - 1) + k])
    hs.ρ += MH * state.ρ
    hs.ρu += MH * state.ρu[1]
    hs.ρv += MH * state.ρu[2]
    hs.ρw += MH * state.ρu[3]
    hs.e_tot += MH * state.ρe
    hs.q_liq += MH * th.q_liq
    hs.q_vap += MH * th.q_vap
    hs.θ_liq_ice += MH * th.θ_liq_ice
    hs.θ_dry += MH * th.θ_dry
    hs.θ_v += MH * th.θ_v
    hs.e_int += MH * th.e_int
    hs.h_m += MH * th.h_m
    hs.h_t += MH * th.h_t

    # TODO: temporary fix
    if isa(atmos.moisture, EquilMoist)
        hs.ρq_tot += MH * state.moisture.ρq_tot
    end

    ν, _ = turbulence_tensors(
        atmos.turbulence,
        state,
        diffusive_flux,
        aux,
        currtime,
    )
    D_t = (ν isa Real ? ν : diag(ν)) * inv_Pr_turb

    # TODO: temporary fix
    if isa(atmos.moisture, EquilMoist)
        d_q_tot = (-D_t) .* diffusive_flux.moisture.∇q_tot
        hs.qt_sgs += MH * state.ρ * d_q_tot[end]
    end

    d_h_tot = -D_t .* diffusive_flux.∇h_tot
    hs.ht_sgs += MH * state.ρ * d_h_tot[end]

    # liquid water path
    # this condition is also going to be used to get the number of points that
    # exist on a horizontal plane provided all planes have the same number of
    # points
    # TODO adjust for possibility of non equivalent horizontal slabs
    if ev == floor(nvertelem / 2) && k == floor(Nqk / 2)
        # TODO: uncomment the line below after rewriting the LWP assignment below using aux.∫dz...?
        # aux = extract_aux(dg, localaux, ijk, e)
        LWP[1] += MH * (localaux[ijk, 1, e] + localaux[ijk, 2, e])
    end

    return nothing
end

function compute_diagnosticsums!(
    atmos,
    state,
    k,
    ijk,
    ev,
    e,
    Nqk,
    nvertelem,
    MH,
    thermoQ,
    horzavgs,
    dsums,
)
    zvals = CollectedDiagnostics.zvals
    th = thermo_vars(thermoQ[ijk, e])
    ha = horzavg_vars(horzavgs[Nqk * (ev - 1) + k])
    ds = diagnostic_vars(dsums[Nqk * (ev - 1) + k])

    u = state.ρu[1] / state.ρ
    v = state.ρu[2] / state.ρ
    w = state.ρu[3] / state.ρ
    ũ = ha.ρu / ha.ρ
    ṽ = ha.ρv / ha.ρ
    w̃ = ha.ρw / ha.ρ
    ẽ = ha.e_tot / ha.ρ
    q̃_tot = ha.ρq_tot / ha.ρ
    # TODO: temporary fix
    if isa(atmos.moisture, EquilMoist)
        q_tot = state.moisture.ρq_tot / state.ρ
    end

    # state and functions of state
    ds.u += MH * ũ
    ds.v += MH * ṽ
    ds.w += MH * w̃
    ds.e_tot += MH * ẽ
    ds.q_tot += MH * ha.ρq_tot / ha.ρ
    ds.q_liq += MH * ha.q_liq
    ds.thd += MH * ha.θ_dry
    ds.thl += MH * ha.θ_liq_ice
    ds.thv += MH * ha.θ_v
    ds.e_int += MH * ha.e_int
    ds.h_m += MH * ha.h_m
    ds.h_t += MH * ha.h_t
    ds.qt_sgs += MH * ha.qt_sgs
    ds.ht_sgs += MH * ha.ht_sgs

    # vertical fluxes
    ds.vert_eddy_mass_flux += MH * (w - w̃) * (state.ρ - ha.ρ)
    ds.vert_eddy_u_flux += MH * (w - w̃) * (u - ha.ρu / ha.ρ)
    ds.vert_eddy_v_flux += MH * (w - w̃) * (v - ha.ρv / ha.ρ)
    ds.vert_eddy_ql_flux += MH * (w - w̃) * (th.q_liq - ha.q_liq)
    ds.vert_eddy_qv_flux += MH * (w - w̃) * (th.q_vap - ha.q_vap)
    ds.vert_eddy_thd_flux += MH * (w - w̃) * (th.θ_dry - ha.θ_dry)
    ds.vert_eddy_thv_flux += MH * (w - w̃) * (th.θ_v - ha.θ_v)
    ds.vert_eddy_thl_flux += MH * (w - w̃) * (th.θ_liq_ice - ha.θ_liq_ice)
    # TODO: temporary fix
    if isa(atmos.moisture, EquilMoist)
        ds.vert_eddy_qt_flux += MH * (w - w̃) * (q_tot - q̃_tot)
        ds.vert_qt_flux += MH * w * q_tot
    end

    # variances
    ds.uvariance += MH * (u - ũ)^2
    ds.vvariance += MH * (v - ṽ)^2
    ds.wvariance += MH * (w - w̃)^2

    # skewness
    ds.wskew += MH * (w - w̃)^3

    # turbulent kinetic energy
    ds.TKE = 0.5 * (ds.uvariance + ds.vvariance + ds.wvariance)

    return nothing
end

"""
    atmos_default_collect(bl, currtime)

Perform a global grid traversal to compute various diagnostics.
"""
function atmos_default_collect(dgngrp::DiagnosticsGroup, currtime)
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    mpirank = MPI.Comm_rank(mpicomm)
    current_time = string(currtime)

    # make sure this time step is not already recorded
    dprefix = @sprintf(
        "%s_%s-%s-num%04d",
        dgngrp.out_prefix,
        dgngrp.name,
        Settings.starttime,
        dgngrp.num
    )
    dfilename = joinpath(Settings.output_dir, dprefix)
    docollect = [false]
    if mpirank == 0
        dfullname = full_name(dgngrp.writer, dfilename)
        if isfile(dfullname)
            @warn """
Diagnostics $(dgngrp.name) collection
    output file $dfullname exists
    skipping collection at $current_time"""
        else
            docollect[1] = true
        end
    end
    MPI.Bcast!(docollect, 0, mpicomm)
    if !docollect[1]
        return nothing
    end

    # extract grid information
    bl = dg.balancelaw
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

    nstate = num_state(bl, FT)
    nauxstate = num_aux(bl, FT)
    ndiff = num_diffusive(bl, FT)

    # thermo variables
    thermoQ = [zeros(FT, num_thermo(FT)) for _ in 1:npoints, _ in 1:nrealelem]

    # horizontal sums and the liquid water path
    l_LWP = zeros(FT, 1)
    horzsums = [zeros(FT, num_horzavg(FT)) for _ in 1:(Nqk * nvertelem)]

    # compute thermo variables and horizontal sums in a single pass
    @visitQ nhorzelem nvertelem Nqk Nq begin
        state = extract_state(dg, localQ, ijk, e)
        aux = extract_aux(dg, localaux, ijk, e)

        compute_thermo!(bl, state, aux, ijk, e, thermoQ)

        diffusive_flux = extract_diffusion(dg, localdiff, ijk, e)
        MH = localvgeo[ijk, grid.MHid, e]
        compute_horzsums!(
            bl,
            state,
            diffusive_flux,
            aux,
            k,
            ijk,
            ev,
            e,
            Nqk,
            nvertelem,
            MH,
            localaux,
            thermoQ,
            horzsums,
            l_LWP,
            currtime,
        )
    end

    # compute the horizontal and LWP averages
    repdvsr = CollectedDiagnostics.repdvsr
    horzavgs = [zeros(FT, num_horzavg(FT)) for _ in 1:(Nqk * nvertelem)]
    for ev in 1:nvertelem
        for k in 1:Nqk
            hsum = MPI.Allreduce(horzsums[Nqk * (ev - 1) + k], +, mpicomm)
            horzavgs[Nqk * (ev - 1) + k] .= hsum ./ repdvsr[Nqk * (ev - 1) + k]
        end
    end
    LWP = zero(FT)
    LWP = MPI.Reduce(l_LWP[1], +, 0, mpicomm)
    if mpirank == 0
        LWP /= repdvsr[1]
    end

    # compute the diagnostics using the previous computed values
    dsums = [zeros(FT, num_diagnostic(FT)) for _ in 1:(Nqk * nvertelem)]
    @visitQ nhorzelem nvertelem Nqk Nq begin
        state = extract_state(dg, localQ, ijk, e)
        MH = localvgeo[ijk, grid.MHid, e]
        compute_diagnosticsums!(
            bl,
            state,
            k,
            ijk,
            ev,
            e,
            Nqk,
            nvertelem,
            MH,
            thermoQ,
            horzavgs,
            dsums,
        )
    end
    varvals = OrderedDict()
    varnames = flattenednames(vars_diagnostic(FT))
    for vari in 1:length(varnames)
        davg = zeros(FT, Nqk * nvertelem)
        for evk in 1:(Nqk * nvertelem)
            dsum = MPI.Reduce(dsums[evk][vari], +, 0, mpicomm)
            if mpirank == 0
                davg[evk] = dsum / repdvsr[evk]
            end
        end
        if mpirank == 0
            varvals[varnames[vari]] = davg
        end
    end

    # write diagnostics
    if mpirank == 0
        write_data(
            dgngrp.writer,
            dfilename,
            OrderedDict("z" => CollectedDiagnostics.zvals),
            varvals,
        )
    end

    # write LWP
    if mpirank == 0
        jldopen(
            joinpath(
                Settings.output_dir,
                "liquid_water_path-$(Settings.starttime).jld2",
            ),
            "a+",
        ) do file
            file[current_time] = LWP
        end
    end

    return nothing
end # function collect

function atmos_default_fini(dgngrp::DiagnosticsGroup, currtime) end
