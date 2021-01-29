# # Dry Atmosphere GCM diagnostics
# 
# This file computes selected diagnostics for the GCM and outputs them
# on the spherical interpolated diagnostic grid.
#
# Use it by calling `Diagnostics.setup_atmos_default_diagnostics()`.
#
# TODO:
# - enable zonal means and calculation of covariances using those means
#   - ds.T_zm = mean(.*1., ds.T; dims = 3)
#   - ds.u_zm = mean((ds.u); dims = 3 )
#   - v_zm = mean(ds.v; dims = 3)
#   - w_zm = mean(ds.w; dims = 3)
#   - ds.uvcovariance = (ds.u .- ds.u_zm) * (ds.v .- v_zm)
#   - ds.vTcovariance = (ds.v .- v_zm) * (ds.T .- ds.T_zm)
# - add more variables, including horiz streamfunction from laplacial of vorticity (LN)
# - density weighting
# - maybe change thermo/dyn separation to local/nonlocal vars?

import CUDA
using LinearAlgebra
using Printf
using Statistics

using ..Atmos
using ..Atmos: recover_thermo_state
using ..DGMethods.NumericalFluxes
using ..TurbulenceClosures: turbulence_tensors

include("diagnostic_fields.jl")
include("vorticity_balancelaw.jl")
include("hyperdiffusion_balancelaw.jl")

mutable struct VorticityBLState
    bl::Union{Nothing, VorticityModel}
    dg::Union{Nothing, DGModel}
    state::Union{Nothing, MPIStateArray}
    dQ::Union{Nothing, MPIStateArray}

    VorticityBLState() = new(nothing, nothing, nothing, nothing)
end

mutable struct HyperdiffusionBLState
    bl::Union{Nothing, DryBiharmonicModel}
    dg::Union{Nothing, DGModel}
    state::Union{Nothing, MPIStateArray}
    dQ::Union{Nothing, MPIStateArray}

    HyperdiffusionBLState() = new(nothing, nothing, nothing, nothing)
end

struct AtmosGCMAdditionalDiagnosticsParams{FT} <: DiagnosticsGroupParams
    timescale::FT
    vort_state::VorticityBLState
    hyper_state::HyperdiffusionBLState

    AtmosGCMAdditionalDiagnosticsParams(timescale::FT) where {FT} =
        new{FT}(timescale, VorticityBLState(), HyperdiffusionBLState())
end

"""
    setup_atmos_default_diagnostics(
        ::AtmosGCMConfigType,
        interval::String,
        out_prefix::String;
        timescale = Inf,
        writer = NetCDFWriter(),
        interpol = nothing,
    ) where {FT}

Create the "AtmosGCMDefault" `DiagnosticsGroup` which contains the following
diagnostic variables:
- u: zonal wind
- v: meridional wind
- w: vertical wind
- rho: air density
- temp: air temperature
- pres: air pressure
- thd: dry potential temperature
- et: total specific energy
- ei: specific internal energy
- ht: specific enthalpy based on total energy
- hi: specific enthalpy based on internal energy
- vort: vertical component of relative vorticity
- vort2: vertical component of relative vorticity from DGModel kernels via a mini balance law

When an `EquilMoist` moisture model is used, the following diagnostic
variables are also output:

- qt: mass fraction of total water in air
- ql: mass fraction of liquid water in air
- qv: mass fraction of water vapor in air
- qi: mass fraction of ice in air
- thv: virtual potential temperature
- thl: liquid-ice potential temperature

When `DryBiharmonic` hyperdiffusion is used, the following diagnostic
variables are also output:

- hyper_e: hyperdiffusion tendency for total energy
- hyper_u: hyperdiffusion tendency for zonal velocity
- hyper_v: hyperdiffusion tendency for meridional velocity

All these variables are output with `lat`, `long`, and `level` dimensions
of an interpolated grid (`interpol` _must_ be specified) as well as a
(unlimited) `time` dimension at the specified `interval`.
"""
function setup_atmos_default_diagnostics(
    ::AtmosGCMConfigType,
    interval::String,
    out_prefix::String;
    timescale = Inf,
    writer = NetCDFWriter(),
    interpol = nothing,
) where {FT}
    # TODO: remove this
    @assert !isnothing(interpol)

    return DiagnosticsGroup(
        "AtmosGCMDefault",
        Diagnostics.atmos_gcm_default_init,
        Diagnostics.atmos_gcm_default_fini,
        Diagnostics.atmos_gcm_default_collect,
        interval,
        out_prefix,
        writer,
        interpol,
        AtmosGCMAdditionalDiagnosticsParams(timescale),
    )
end

# Declare all (3D) variables for this diagnostics group
function vars_atmos_gcm_default_simple_3d(atmos::AtmosModel, FT)
    @vars begin
        u::FT
        v::FT
        w::FT
        rho::FT
        temp::FT
        pres::FT
        thd::FT                 # θ_dry
        et::FT                  # e_tot
        ei::FT                  # e_int
        ht::FT
        hi::FT
        vort::FT                # Ω₃
        vort2::FT               # Ω_bl₃

        moisture::vars_atmos_gcm_default_simple_3d(atmos.moisture, FT)
        hyperdiffusion::vars_atmos_gcm_default_simple_3d(
            atmos.hyperdiffusion,
            FT,
        )
    end
end
vars_atmos_gcm_default_simple_3d(::MoistureModel, FT) = @vars()
function vars_atmos_gcm_default_simple_3d(m::EquilMoist, FT)
    @vars begin
        qt::FT                  # q_tot
        ql::FT                  # q_liq
        qv::FT                  # q_vap
        qi::FT                  # q_ice
        thv::FT                 # θ_vir
        thl::FT                 # θ_liq

    end
end
vars_atmos_gcm_default_simple_3d(::HyperDiffusion, FT) = @vars()
function vars_atmos_gcm_default_simple_3d(h::DryBiharmonic, FT)
    @vars begin
        hyper_e::FT
        hyper_u::FT
        hyper_v::FT
    end
end
num_atmos_gcm_default_simple_3d_vars(m, FT) =
    varsize(vars_atmos_gcm_default_simple_3d(m, FT))
atmos_gcm_default_simple_3d_vars(m, array) =
    Vars{vars_atmos_gcm_default_simple_3d(m, eltype(array))}(array)

# Collect all (3D) variables for this diagnostics group
function atmos_gcm_default_simple_3d_vars!(
    atmos::AtmosModel,
    state_prognostic,
    thermo,
    dyn_vort,
    dyn_vort2,
    dyn_hd,
    vars,
)
    vars.u = state_prognostic.ρu[1] / state_prognostic.ρ
    vars.v = state_prognostic.ρu[2] / state_prognostic.ρ
    vars.w = state_prognostic.ρu[3] / state_prognostic.ρ
    vars.rho = state_prognostic.ρ
    vars.et = state_prognostic.ρe / state_prognostic.ρ

    vars.temp = thermo.temp
    vars.pres = thermo.pres
    vars.thd = thermo.θ_dry
    vars.ei = thermo.e_int
    vars.ht = thermo.h_tot
    vars.hi = thermo.h_int

    vars.vort = dyn_vort.Ω₃

    vars.vort2 = dyn_vort2.Ω_bl₃

    atmos_gcm_default_simple_3d_vars!(
        atmos.moisture,
        state_prognostic,
        thermo,
        vars,
    )
    atmos_gcm_default_simple_3d_vars!(
        atmos.hyperdiffusion,
        state_prognostic,
        thermo,
        dyn_hd,
        vars,
    )

    return nothing
end
function atmos_gcm_default_simple_3d_vars!(
    ::MoistureModel,
    state_prognostic,
    thermo,
    vars,
)
    return nothing
end
function atmos_gcm_default_simple_3d_vars!(
    moist::EquilMoist,
    state_prognostic,
    thermo,
    vars,
)
    vars.moisture.qt = state_prognostic.moisture.ρq_tot / state_prognostic.ρ
    vars.moisture.ql = thermo.moisture.q_liq
    vars.moisture.qv = thermo.moisture.q_vap
    vars.moisture.qi = thermo.moisture.q_ice
    vars.moisture.thv = thermo.moisture.θ_vir
    vars.moisture.thl = thermo.moisture.θ_liq_ice

    return nothing
end
function atmos_gcm_default_simple_3d_vars!(
    ::HyperDiffusion,
    state_prognostic,
    thermo,
    dyn_hd,
    vars,
)
    return nothing
end
function atmos_gcm_default_simple_3d_vars!(
    h::DryBiharmonic,
    state_prognostic,
    thermo,
    dyn_hd,
    vars,
)
    vars.hyperdiffusion.hyper_e = dyn_hd.he_bl
    vars.hyperdiffusion.hyper_u = dyn_hd.hu_bl₁
    vars.hyperdiffusion.hyper_v = dyn_hd.hu_bl₂

    return nothing
end

# Dynamic variables
function vars_dyn_vort(FT)
    @vars begin
        Ω₁::FT
        Ω₂::FT
        Ω₃::FT
    end
end
dyn_vort_vars(array) = Vars{vars_dyn_vort(eltype(array))}(array)

function vars_dyn_vort2(FT)
    @vars begin
        Ω_bl₁::FT
        Ω_bl₂::FT
        Ω_bl₃::FT
    end
end
dyn_vort2_vars(array) = Vars{vars_dyn_vort2(eltype(array))}(array)

function vars_dyn_hd(FT)
    @vars begin
        he_bl::FT
        hu_bl₁::FT
        hu_bl₂::FT
        hu_bl₃::FT
    end
end
dyn_hd_vars(array) = Vars{vars_dyn_hd(eltype(array))}(array)


"""
    atmos_gcm_default_init(dgngrp, currtime)

Initialize the GCM default diagnostics group, establishing the output file's
dimensions and variables.
"""
function atmos_gcm_default_init(dgngrp::DiagnosticsGroup, currtime)
    dg = Settings.dg
    grid = dg.grid
    atmos = dg.balance_law
    FT = eltype(Settings.Q)
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)

    if !(dgngrp.interpol isa InterpolationCubedSphere)
        @warn """
            Diagnostics ($dgngrp.name): currently requires `InterpolationCubedSphere`!
            """
        return nothing
    end

    params = dgngrp.params

    # set up the vorticity mini balance law
    vort_state = params.vort_state
    vort_state.bl = VorticityModel()
    vort_state.dg = DGModel(
        vort_state.bl,
        grid,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )
    vort_state.state = init_ode_state(vort_state.dg, FT(0))
    vort_state.dQ = similar(
        vort_state.state;
        vars = @vars(Ω_bl::SVector{3, FT}),
        nstate = 3,
    )

    # set up the hyperdiffusion mini balance law
    hyper_state = params.hyper_state
    if atmos.hyperdiffusion isa DryBiharmonic && isfinite(params.timescale)
        hyper_state.bl = DryBiharmonicModel(
            atmos.param_set,
            atmos.orientation,
            params.timescale,
        )
        hyper_state.dg = DGModel(
            hyper_state.bl,
            grid,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            diffusion_direction = dg.diffusion_direction,
        )
        hyper_state.state = init_ode_state(hyper_state.dg, FT(0))
        hyper_state.dQ = similar(
            hyper_state.state;
            vars = @vars(hyper_e::FT, hyper_u::SVector{3, FT}),
            nstate = 4,
        )
    end

    if mpirank == 0
        # get dimensions for the interpolated grid
        dims = dimensions(dgngrp.interpol)

        # adjust the level dimension for `planet_radius`
        level_val = dims["level"]
        dims["level"] = (
            level_val[1] .- FT(planet_radius(Settings.param_set)),
            level_val[2],
        )

        # set up the variables we're going to be writing
        vars = OrderedDict()
        varnames = map(
            prefix_filter,
            flattenednames(vars_atmos_gcm_default_simple_3d(atmos, FT)),
        )
        for varname in varnames
            var = Variables[varname]
            vars[varname] = (tuple(collect(keys(dims))...), FT, var.attrib)
        end

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
    atmos_gcm_default_collect(dgngrp, currtime)

    Master function that performs a global grid traversal to compute various
    diagnostics using the above functions.
"""
function atmos_gcm_default_collect(dgngrp::DiagnosticsGroup, currtime)
    interpol = dgngrp.interpol
    if !(interpol isa InterpolationCubedSphere)
        @warn """
            Diagnostics ($dgngrp.name): currently requires `InterpolationCubedSphere`!
            """
        return nothing
    end
    params = dgngrp.params
    vort_state = params.vort_state
    hyper_state = params.hyper_state

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

    # get needed arrays onto the CPU
    device = array_device(Q)
    if device isa CPU
        ArrayType = Array
        state_data = Q.realdata
        aux_data = dg.state_auxiliary.realdata
    else
        ArrayType = CUDA.CuArray
        state_data = Array(Q.realdata)
        aux_data = Array(dg.state_auxiliary.realdata)
    end
    FT = eltype(state_data)

    # TODO: can this be done in one pass?
    #
    # Non-local vars, e.g. relative vorticity
    vgrad = VectorGradients(dg, Q)
    vort = Vorticity(dg, vgrad)

    # run the vorticity mini balance law
    vort_state.dg.state_auxiliary.u .= Q.ρu ./ Q.ρ
    vort_state.dg(vort_state.dQ, vort_state.state, nothing, FT(0))

    # run the hyperdiffusion mini balance law
    do_hyperdiffusion =
        atmos.hyperdiffusion isa DryBiharmonic && isfinite(params.timescale)
    if do_hyperdiffusion
        hyper_state.dg.state_auxiliary.ρ .= Q.ρ
        hyper_state.dg.state_auxiliary.ρu .= Q.ρu
        hyper_state.dg.state_auxiliary.ρe .= Q.ρe
        ix_temp = varsindex(vars(dg.state_auxiliary), :moisture, :temperature)
        hyper_state.dg.state_auxiliary.temperature .=
            view(MPIStateArrays.realview(dg.state_auxiliary), :, ix_temp, :)
        hyper_state.dg(hyper_state.dQ, hyper_state.state, nothing, FT(0))
    end

    # Compute thermo variables element-wise
    thermo_array = Array{FT}(undef, npoints, num_thermo(atmos, FT), nrealelem)
    @traverse_dg_grid grid_info topl_info begin
        state = extract_state(dg, state_data, ijk, e, Prognostic())
        aux = extract_state(dg, aux_data, ijk, e, Auxiliary())

        thermo = thermo_vars(atmos, view(thermo_array, ijk, :, e))
        compute_thermo!(atmos, state, aux, thermo)
    end

    # Interpolate the state, thermo, dgdiags and dyn vars to sphere (u and vorticity
    # need projection to zonal, merid). All this may happen on the GPU.
    istate =
        ArrayType{FT}(undef, interpol.Npl, number_states(atmos, Prognostic()))
    interpolate_local!(interpol, Q.realdata, istate)

    ithermo = ArrayType{FT}(undef, interpol.Npl, num_thermo(atmos, FT))
    interpolate_local!(interpol, ArrayType(thermo_array), ithermo)

    idyn_vort = ArrayType{FT}(undef, interpol.Npl, size(vort.data, 2))
    interpolate_local!(interpol, vort.data, idyn_vort)

    idyn_vort2 = ArrayType{FT}(undef, interpol.Npl, size(vort_state.dQ.data, 2))
    interpolate_local!(interpol, vort_state.dQ.data, idyn_vort2)

    idyn_hd = nothing
    if do_hyperdiffusion
        idyn_hd =
            ArrayType{FT}(undef, interpol.Npl, size(hyper_state.dQ.data, 2))
        interpolate_local!(interpol, hyper_state.dQ.data, idyn_hd)
    end

    # TODO: get indices here without hard-coding them
    _ρu, _ρv, _ρw = 2, 3, 4
    project_cubed_sphere!(interpol, istate, (_ρu, _ρv, _ρw))
    _Ω₁, _Ω₂, _Ω₃ = 1, 2, 3
    project_cubed_sphere!(interpol, idyn_vort, (_Ω₁, _Ω₂, _Ω₃))
    project_cubed_sphere!(interpol, idyn_vort2, (_Ω₁, _Ω₂, _Ω₃))
    if do_hyperdiffusion
        _u₁, _u₂, _u₃ = 2, 3, 4
        project_cubed_sphere!(interpol, idyn_hd, (_u₁, _u₂, _u₃))
    end

    # FIXME: accumulating to rank 0 is not scalable
    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)
    all_thermo_data = accumulate_interpolated_data(mpicomm, interpol, ithermo)
    all_dyn_vort_data =
        accumulate_interpolated_data(mpicomm, interpol, idyn_vort)
    all_dyn_vort2_data =
        accumulate_interpolated_data(mpicomm, interpol, idyn_vort2)
    all_dyn_hd_data = nothing
    if do_hyperdiffusion
        all_dyn_hd_data =
            accumulate_interpolated_data(mpicomm, interpol, idyn_hd)
    end

    if mpirank == 0
        # get dimensions for the interpolated grid
        dims = dimensions(dgngrp.interpol)

        # set up the array for the diagnostic variables based on the interpolated grid
        nlong = length(dims["long"][1])
        nlat = length(dims["lat"][1])
        nlevel = length(dims["level"][1])

        simple_3d_vars_array = Array{FT}(
            undef,
            nlong,
            nlat,
            nlevel,
            num_atmos_gcm_default_simple_3d_vars(atmos, FT),
        )

        @traverse_interpolated_grid nlong nlat nlevel begin
            statei = Vars{vars_state(atmos, Prognostic(), FT)}(view(
                all_state_data,
                lo,
                la,
                le,
                :,
            ))
            thermoi = thermo_vars(atmos, view(all_thermo_data, lo, la, le, :))
            dyni_vort = dyn_vort_vars(view(all_dyn_vort_data, lo, la, le, :))
            dyni_vort2 = dyn_vort2_vars(view(all_dyn_vort2_data, lo, la, le, :))
            dyni_hd = nothing
            if do_hyperdiffusion
                dyni_hd = dyn_hd_vars(view(all_dyn_hd_data, lo, la, le, :))
            end
            simple_3d_vars = atmos_gcm_default_simple_3d_vars(
                atmos,
                view(simple_3d_vars_array, lo, la, le, :),
            )

            atmos_gcm_default_simple_3d_vars!(
                atmos,
                statei,
                thermoi,
                dyni_vort,
                dyni_vort2,
                dyni_hd,
                simple_3d_vars,
            )
        end

        # assemble the diagnostics for writing
        varvals = OrderedDict()
        varnames = map(
            prefix_filter,
            flattenednames(vars_atmos_gcm_default_simple_3d(atmos, FT)),
        )
        for (vari, varname) in enumerate(varnames)
            varvals[varname] = simple_3d_vars_array[:, :, :, vari]
        end

        # write output
        append_data(dgngrp.writer, varvals, currtime)
    end

    MPI.Barrier(mpicomm)
    return nothing
end # function collect

function atmos_gcm_default_fini(dgngrp::DiagnosticsGroup, currtime) end
