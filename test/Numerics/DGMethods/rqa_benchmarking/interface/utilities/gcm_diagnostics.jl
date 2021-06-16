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

# mutable struct VorticityBLState <: DiagnosticsGroupParams
#     bl::Union{Nothing, VorticityModel}
#     dg::Union{Nothing, DGModel}
#     state::Union{Nothing, MPIStateArray}
#     dQ::Union{Nothing, MPIStateArray}

#     VorticityBLState() = new(nothing, nothing, nothing, nothing)
# end

"""
    setup_atmos_default_diagnostics(
        ::AtmosGCMConfigType,
        interval::String,
        out_prefix::String;
        writer::AbstractWriter,
        interpol = nothing,
    )

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

All these variables are output with `lat`, `long`, and `level` dimensions
of an interpolated grid (`interpol` _must_ be specified) as well as a
(unlimited) `time` dimension at the specified `interval`.
"""
function setup_atmos_default_diagnostics(
    simulation::Simulation,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
    # TODO: remove this
    @assert !isnothing(interpol)

    return DiagnosticsGroup(
        "AtmosGCMDefault",
        atmos_gcm_default_init,
        atmos_gcm_default_fini,
        atmos_gcm_default_collect,
        interval,
        out_prefix,
        writer,
        interpol,
        # VorticityBLState(),
    )
end

# Declare all (3D) variables for this diagnostics group
function vars_atmos_gcm_default_simple_3d(::DryAtmosModel, FT)
    @vars begin
        u::FT
        v::FT
        w::FT
        ρ::FT
        # temp::FT
        # pres::FT
        q::FT
      	e::FT
        # thd::FT                 # θ_dry
        # et::FT                  # e_tot
        # ei::FT                  # e_int
        # ht::FT
        # hi::FT
        # vort::FT                # Ω₃
        # vort2::FT               # Ω_bl₃

        # moisture::vars_atmos_gcm_default_simple_3d(moisture_model(atmos), FT)
    end
end

num_atmos_gcm_default_simple_3d_vars(m, FT) =
    varsize(vars_atmos_gcm_default_simple_3d(m, FT))
atmos_gcm_default_simple_3d_vars(m, array) =
    Vars{vars_atmos_gcm_default_simple_3d(m, eltype(array))}(array)

# Collect all (3D) variables for this diagnostics group
function atmos_gcm_default_simple_3d_vars!(
    ::DryAtmosModel,
    state_prognostic,
    vars,
)
    vars.u = state_prognostic.ρu[1] / state_prognostic.ρ
    vars.v = state_prognostic.ρu[2] / state_prognostic.ρ
    vars.w = state_prognostic.ρu[3] / state_prognostic.ρ
    vars.ρ = state_prognostic.ρ
    vars.q = state_prognostic.ρq / state_prognostic.ρ
    vars.e = state_prognostic.ρe / state_prognostic.ρ

    return nothing
end

"""
    atmos_gcm_default_init(simulation, interval, interpol, currtime)

Initialize the GCM default diagnostics group, establishing the output file's
dimensions and variables.
"""
function atmos_gcm_default_init(simulation, interval, interpol, currtime)
    if simulation.rhs isa Tuple
	if simulation.rhs[1] isa AbstractRate 
            model = simulation.rhs[1].model
	else
	    model = simulation.rhs[1]
	end
    else
	model = simulation.rhs
    end 
    dg = model
    grid = simulation.grid.numerical
    atmos = dg.balance_law
    FT = eltype(simulation.state)

    # TODO: make mpicomm an input arg
    mpicomm = MPI.COMM_WORLD
    mpirank = MPI.Comm_rank(mpicomm)

    if !(interpol isa InterpolationCubedSphere)
        @warn """
            Diagnostics ($dgngrp.name): currently requires `InterpolationCubedSphere`!
            """
        return nothing
    end

    if mpirank == 0
        # get dimensions for the interpolated grid
        dims = dimensions(interpol)

        # adjust the level dimension for `planet_radius`
        level_val = dims["level"]
        dims["level"] = (
            level_val[1] .- FT(simulation.grid.domain.radius),
            level_val[2],
        )

        # set up the variables we're going to be writing
        vars = OrderedDict()
        varnames = map(
            s -> startswith(s, "moisture.") ? s[10:end] : s,
            flattenednames(vars_atmos_gcm_default_simple_3d(atmos, FT)),
        )
        for varname in varnames
            var = Variables[varname]
            vars[varname] = (tuple(collect(keys(dims))...), FT, var.attrib)
        end

        # create the output file
        dprefix = @sprintf("%s_%s", dgngrp.out_prefix, dgngrp.name)
        dfilename = joinpath(Settings.output_dir, dprefix)
        noov = Settings.no_overwrite
        init_data(dgngrp.writer, dfilename, noov, dims, vars)
    end

    return nothing
end

"""
    atmos_gcm_default_collect(simulation, interval, interpol, currtime)

    Master function that performs a global grid traversal to compute various
    diagnostics using the above functions.
"""
function atmos_gcm_default_collect(simulation, interval, interpol, currtime)
    interpol = interpol
    if !(interpol isa InterpolationCubedSphere)
        @warn """
            Diagnostics ($dgngrp.name): currently requires `InterpolationCubedSphere`!
            """
        return nothing
    end
 #   vort_state = dgngrp.params
# TODO: verify how comm info is passed 
    mpicomm = MPI.COMM_WORLD
    if simulation.rhs isa Tuple
	if simulation.rhs[1] isa AbstractRate 
            model = simulation.rhs[1].model
	else
	    model = simulation.rhs[1]
	end
    else
	model = simulation.rhs
    end 
    dg = model
    Q = simulation.state
    mpirank = MPI.Comm_rank(mpicomm)
    atmos = dg.balance_law
    grid = simulation.grid.numerical
    grid_info = basic_grid_info(grid)
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

    @traverse_dg_grid grid_info topl_info begin
        state = extract_state(dg, state_data, ijk, e, Prognostic())
        aux = extract_state(dg, aux_data, ijk, e, Auxiliary())

        # thermo = thermo_vars(atmos, view(thermo_array, ijk, :, e))
        # compute_thermo!(atmos, state, aux, thermo)
    end

    # Interpolate the state, thermo, dgdiags and dyn vars to sphere (u and vorticity
    # need projection to zonal, merid). All this may happen on the GPU.
    istate =
        ArrayType{FT}(undef, interpol.Npl, number_states(atmos, Prognostic()))
    interpolate_local!(interpol, Q.realdata, istate)

#     ithermo = ArrayType{FT}(undef, interpol.Npl, num_thermo(atmos, FT))
#     interpolate_local!(interpol, ArrayType(thermo_array), ithermo)

#     idyn = ArrayType{FT}(undef, interpol.Npl, size(vort.data, 2))
#     interpolate_local!(interpol, vort.data, idyn)

#     idyn_bl = ArrayType{FT}(undef, interpol.Npl, size(vort_state.dQ.data, 2))
#     interpolate_local!(interpol, vort_state.dQ.data, idyn_bl)

    # TODO: get indices here without hard-coding them
    _ρu, _ρv, _ρw = 2, 3, 4
    project_cubed_sphere!(interpol, istate, (_ρu, _ρv, _ρw))
#     _Ω₁, _Ω₂, _Ω₃ = 1, 2, 3
#     project_cubed_sphere!(interpol, idyn, (_Ω₁, _Ω₂, _Ω₃))
#     project_cubed_sphere!(interpol, idyn_bl, (_Ω₁, _Ω₂, _Ω₃))


    # FIXME: accumulating to rank 0 is not scalable
    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)
#     all_thermo_data = accumulate_interpolated_data(mpicomm, interpol, ithermo)
#     all_dyn_data = accumulate_interpolated_data(mpicomm, interpol, idyn)
#     all_dyn_bl_data = accumulate_interpolated_data(mpicomm, interpol, idyn_bl)

    if mpirank == 0
        # get dimensions for the interpolated grid
        dims = dimensions(interpol)

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
        #     thermoi = thermo_vars(atmos, view(all_thermo_data, lo, la, le, :))
        #     dyni = dyn_vars(view(all_dyn_data, lo, la, le, :))
        #     dyn_bli = dyn_bl_vars(view(all_dyn_bl_data, lo, la, le, :))
            simple_3d_vars = atmos_gcm_default_simple_3d_vars(
                atmos,
                view(simple_3d_vars_array, lo, la, le, :),
            )

            atmos_gcm_default_simple_3d_vars!(
                atmos,
                statei,
                simple_3d_vars,
            )
        end

        # assemble the diagnostics for writing
        varvals = OrderedDict()
        varnames = map(
            # s -> startswith(s, "moisture.") ? s[10:end] : s, # Depends on atmos.moisture which rqa interface doesn't have
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

# TODO: Fix function signature to be consistent with init / collect 
function atmos_gcm_default_fini(dgngrp::DiagnosticsGroup, currtime) end
