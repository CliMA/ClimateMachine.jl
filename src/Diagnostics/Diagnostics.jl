"""
    Diagnostics

Accumulate mean fields and covariance statistics on the computational grid.

"""
module Diagnostics

using Dates
using FileIO
using JLD2
using MPI
using StaticArrays

using ..Atmos
using ..Atmos: thermo_state, turbulence_tensors
using ..SubgridScaleParameters: inv_Pr_turb
using ..DGmethods: num_state, vars_state, num_aux, vars_aux, vars_diffusive, num_diffusive
using ..Mesh.Topologies
using ..Mesh.Grids
using ..MoistThermodynamics
using ..MPIStateArrays
using ..PlanetParameters
using ..VariableTemplates

export gather_diagnostics

include("diagnostic_vars.jl")

function extract_state(dg, localQ, ijk, e)
    bl = dg.balancelaw
    FT = eltype(localQ)
    nstate = num_state(bl, FT)
    l_Q = MArray{Tuple{nstate},FT}(undef)
    for s in 1:nstate
        l_Q[s] = localQ[ijk,s,e]
    end
    return Vars{vars_state(bl, FT)}(l_Q)
end

function extract_aux(dg, auxstate, ijk, e)
    bl = dg.balancelaw
    FT = eltype(auxstate)
    nauxstate = num_aux(bl, FT)
    l_aux = MArray{Tuple{nauxstate},FT}(undef)
    for s in 1:nauxstate
        l_aux[s] = auxstate[ijk,s,e]
    end
    return Vars{vars_aux(bl, FT)}(l_aux)
end

function extract_diffusion(dg, localdiff, ijk, e)
    bl = dg.balancelaw
    FT = eltype(localdiff)
    ndiff = num_diffusive(bl, FT)
    l_diff = MArray{Tuple{ndiff},FT}(undef)
    for s in 1:ndiff
        l_diff[s] = localdiff[ijk,s,e]
    end
    return Vars{vars_diffusive(bl, FT)}(l_diff)
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

function compute_thermo!(FT, bl, state, k, ijk, ev, e, z, zvals, thermoQ, aux)
    zvals[k,ev] = z

    e_tot = state.ρe / state.ρ
    ts = thermo_state(bl.moisture, bl.orientation, state, aux)
    e_int = internal_energy(ts)
    Phpart = PhasePartition(ts)

    th = thermo_vars(thermoQ[ijk,e])
    th.q_liq     = Phpart.liq
    th.q_ice     = Phpart.ice
    th.q_vap     = Phpart.tot-Phpart.liq-Phpart.ice
    th.T         = air_temperature(ts)
    th.θ_liq_ice = liquid_ice_pottemp(ts)
    th.θ_dry     = dry_pottemp(ts)
    th.θ_v       = virtual_pottemp(ts)
    th.e_int     = e_int

    # Moist and total henthalpy
    R_m          = gas_constant_air(ts)
    th.h_m       = e_int + R_m * th.T
    th.h_t       = e_tot + R_m * th.T

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

function compute_horzsums!(atmos::AtmosModel, state, diffusive_flx, aux, k, ijk,
                           ev, e, Nqk, nvertelem, MH, localaux, thermoQ,
                           horzsums, repdvsr, LWP, t)
    th = thermo_vars(thermoQ[ijk,e])
    hs = horzavg_vars(horzsums[k,ev])
    hs.ρ         += MH * state.ρ
    hs.ρu        += MH * state.ρu[1]
    hs.ρv        += MH * state.ρu[2]
    hs.ρw        += MH * state.ρu[3]
    hs.e_tot     += MH * state.ρe
    hs.ρq_tot    += MH * state.moisture.ρq_tot
    hs.q_liq     += MH * th.q_liq
    hs.q_vap     += MH * th.q_vap
    hs.θ_liq_ice += MH * th.θ_liq_ice
    hs.θ_dry     += MH * th.θ_dry
    hs.θ_v       += MH * th.θ_v
    hs.e_int     += MH * th.e_int
    hs.h_m       += MH * th.h_m
    hs.h_t       += MH * th.h_t

    ν, _ = turbulence_tensors(atmos.turbulence, state, diffusive_flx, aux, t)
    D_t = (ν isa Real ? ν : diag(ν)) * inv_Pr_turb

    d_q_tot = (-D_t) .* diffusive_flx.moisture.∇q_tot
    hs.qt_sgs  += MH * state.ρ * d_q_tot[end]

    d_h_tot = -D_t .* diffusive_flx.∇h_tot
    hs.ht_sgs    += MH * state.ρ * d_h_tot[end]

    repdvsr[Nqk*(ev-1)+k] += MH

    # liquid water path
    # this condition is also going to be used to get the number of points that
    # exist on a horizontal plane provided all planes have the same number of
    # points
    # TODO adjust for possibility of non equivalent horizontal slabs
    if ev == floor(nvertelem/2) && k == floor(Nqk/2)
        # TODO: uncomment the line below after rewriting the LWP assignment below using aux.∫dz...?
        # aux = extract_aux(dg, localaux, ijk, e)
        LWP[1] += MH * (localaux[ijk,1,e] + localaux[ijk,2,e])
    end

    return nothing
end

function compute_diagnosticsums!(state, k, ijk, ev, e, MH, zvals,
                                 thermoQ, horzavgs, dsums)
    th = thermo_vars(thermoQ[ijk,e])
    ha = horzavg_vars(horzavgs[k,ev])
    ds = diagnostic_vars(dsums[k,ev])

    u = state.ρu[1] / state.ρ
    v = state.ρu[2] / state.ρ
    w = state.ρu[3] / state.ρ
    q_tot = state.moisture.ρq_tot / state.ρ
    ũ = ha.ρu / ha.ρ
    ṽ = ha.ρv / ha.ρ
    w̃ = ha.ρw / ha.ρ
    ẽ = ha.e_tot / ha.ρ
    q̃_tot = ha.ρq_tot / ha.ρ

    # vertical coordinate
    ds.z        += MH * zvals[k,ev]

    # state and functions of state
    ds.u        += MH * ũ
    ds.v        += MH * ṽ
    ds.w        += MH * w̃
    ds.e_tot    += MH * ẽ
    ds.q_tot    += MH * ha.ρq_tot / ha.ρ
    ds.q_liq    += MH * ha.q_liq
    ds.thd      += MH * ha.θ_dry
    ds.thl      += MH * ha.θ_liq_ice
    ds.thv      += MH * ha.θ_v
    ds.e_int    += MH * ha.e_int
    ds.h_m      += MH * ha.h_m
    ds.h_t      += MH * ha.h_t
    ds.qt_sgs   += MH * ha.qt_sgs
    ds.ht_sgs   += MH * ha.ht_sgs

    # vertical fluxes
    ds.vert_eddy_mass_flx += MH * (w - w̃) * (state.ρ - ha.ρ)
    ds.vert_eddy_u_flx    += MH * (w - w̃) * (u - ha.ρu / ha.ρ)
    ds.vert_eddy_v_flx    += MH * (w - w̃) * (v - ha.ρv / ha.ρ)
    ds.vert_eddy_qt_flx   += MH * (w - w̃) * (q_tot - q̃_tot)
    ds.vert_qt_flx        += MH * w * q_tot
    ds.vert_eddy_ql_flx   += MH * (w - w̃) * (th.q_liq - ha.q_liq)
    ds.vert_eddy_qv_flx   += MH * (w - w̃) * (th.q_vap - ha.q_vap)
    ds.vert_eddy_thd_flx  += MH * (w - w̃) * (th.θ_dry - ha.θ_dry)
    ds.vert_eddy_thv_flx  += MH * (w - w̃) * (th.θ_v - ha.θ_v)
    ds.vert_eddy_thl_flx  += MH * (w - w̃) * (th.θ_liq_ice - ha.θ_liq_ice)

    # variances
    ds.uvariance += MH * (u - ũ)^2
    ds.vvariance += MH * (v - ṽ)^2
    ds.wvariance += MH * (w - w̃)^2

    # skewness
    ds.wskew     += MH * (w - w̃)^3

    # turbulent kinetic energy
    ds.TKE       += MH * 0.5 * (ds.uvariance + ds.vvariance + ds.wvariance)

    return nothing
end

macro visitQ(nhorzelem, nvertelem, Nqk, Nq, expr)
    return esc(
        quote
            for eh in 1:nhorzelem
                for ev in 1:nvertelem
                    e = ev + (eh - 1) * nvertelem
                    for k in 1:Nqk
                        for j in 1:Nq
                            for i in 1:Nq
                                ijk = i + Nq * ((j-1) + Nq * (k-1))
                                $expr
                            end
                        end
                    end
                end
        end
    end)
end

"""
    gather_diagnostics(mpicomm, dg, Q, current_time_string, out_dir)

Compute various diagnostic variables and write them to JLD2 files in `out_dir`,
indexed by `current_time_string`.
"""
function gather_diagnostics(mpicomm,
                            dg,
                            Q,
                            diagnostics_time_str,
                            sim_time_str,
                            out_dir,
                            t)
    # make sure this time step is not already recorded
    try
        jldopen(joinpath(out_dir,
                "diagnostics-$(diagnostics_time_str).jld2"), "r") do file
            davgs = file[sim_time_str]
            @warn "diagnostics for time step $(sim_time_str) gathered already"
            return nothing
        end
    catch e
    end

    mpirank = MPI.Comm_rank(mpicomm)
    nranks = MPI.Comm_size(mpicomm)

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
        localQ    = Q.realdata
        localaux  = dg.auxstate.realdata
        localvgeo = grid.vgeo
        localdiff = dg.diffstate.realdata
    else
        localQ    = Array(Q.realdata)
        localaux  = Array(dg.auxstate.realdata)
        localvgeo = Array(grid.vgeo)
        localdiff = Array(dg.diffstate.realdata)
    end
    FT = eltype(localQ)

    nstate    = num_state(bl, FT)
    nauxstate = num_aux(bl, FT)
    ndiff     = num_diffusive(bl, FT)

    # vertical coordinates and thermo variables
    zvals = zeros(Nqk, nvertelem)
    thermoQ = [zeros(FT, num_thermo(FT)) for _ in 1:npoints, _ in 1:nrealelem]

    # divisor for horizontal averages
    l_repdvsr = zeros(FT, Nqk * nvertelem)

    # horizontal sums and the liquid water path
    l_LWP = zeros(FT, 1)
    horzsums = [zeros(FT, num_horzavg(FT)) for _ in 1:Nqk, _ in 1:nvertelem]

    # compute thermo variables and horizontal sums in a single pass
    @visitQ nhorzelem nvertelem Nqk Nq begin
        state = extract_state(dg, localQ, ijk, e)
        aux   = extract_aux(dg, localaux, ijk, e)

        z = localvgeo[ijk,grid.x3id,e]
        compute_thermo!(FT, bl, state, k, ijk, ev, e, z, zvals, thermoQ, aux)

        diffusive_flx = extract_diffusion(dg, localdiff, ijk, e)
        MH = localvgeo[ijk,grid.MHid,e]
        compute_horzsums!(bl, state, diffusive_flx, aux, k, ijk, ev, e, Nqk,
                          nvertelem, MH, localaux, thermoQ, horzsums, l_repdvsr,
                          l_LWP, t)
    end

    # compute the full number of points on a slab
    repdvsr = MPI.Allreduce(l_repdvsr, +, mpicomm)

    # compute the horizontal and LWP averages
    horzavgs = [zeros(FT, num_horzavg(FT)) for _ in 1:Nqk, _ in 1:nvertelem]
    for ev in 1:nvertelem
        for k in 1:Nqk
            hsum = MPI.Allreduce(horzsums[k,ev], +, mpicomm)
            horzavgs[k,ev][1:end] = hsum ./ repdvsr[Nqk*(ev-1)+k]
        end
    end
    LWP = zero(FT)
    LWP = MPI.Reduce(l_LWP[1], +, 0, mpicomm)
    if mpirank == 0
        LWP /= repdvsr[1]
    end

    # compute the diagnostics with the previous computed variables
    dsums = [zeros(FT, num_diagnostic(FT)) for _ in 1:Nqk, _ in 1:nvertelem]
    @visitQ nhorzelem nvertelem Nqk Nq begin
        state = extract_state(dg, localQ, ijk, e)
        MH = localvgeo[ijk,grid.MHid,e]
        compute_diagnosticsums!(state, k, ijk, ev, e, MH, zvals,
                                thermoQ, horzavgs, dsums)
    end
    davgs = [zeros(FT, num_diagnostic(FT)) for _ in 1:Nqk, _ in 1:nvertelem]
    for ev in 1:nvertelem
        for k in 1:Nqk
            dsum = MPI.Reduce(dsums[k,ev], +, 0, mpicomm)
            if mpirank == 0
                davgs[k,ev][1:end] = dsum ./ repdvsr[Nqk*(ev-1)+k]
            end
        end
    end

    # write results
    if mpirank == 0
        jldopen(joinpath(out_dir,
                "diagnostics-$(diagnostics_time_str).jld2"), "a+") do file
            file[sim_time_str] = davgs
        end
        jldopen(joinpath(out_dir,
                "liquid_water_path-$(diagnostics_time_str).jld2"), "a+") do file
            file[sim_time_str] = LWP
        end
    end

    return nothing
end # function gather_diagnostics

end # module Diagnostics

