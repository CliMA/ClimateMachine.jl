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
using OrderedCollections

using ClimateMachine.Diagnostics: 
    extract_state, 
    @traverse_dg_grid, 
    @traverse_interpolated_grid,
    DiagnosticVariable,
    var_attrib

using ClimateMachine.Interpolation

using ClimateMachine.Writers


### 
### Define Global RQAVars (RQA Specific - Following current Diagnostics but perhaps change this ??? TODO)
###
const RQAVars = OrderedDict{String,DiagnosticVariable}()

function init_diag_vars()
    RQAVars["u"] = DiagnosticVariable(
        "u",
        var_attrib("m s^-1", "zonal wind", "eastward_wind"),
    )
    RQAVars["v"] = DiagnosticVariable(
        "v",
        var_attrib("m s^-1", "meridional wind", "northward_wind"),
    )
    RQAVars["w"] = DiagnosticVariable(
        "w",
        var_attrib("m s^-1", "vertical wind", "upward_air_velocity"),
    )
    RQAVars["rho"] = DiagnosticVariable(
        "rho",
        var_attrib("kg m^-3", "air density", "air_density"),
    )
    RQAVars["q"] = DiagnosticVariable(
        "q",
        var_attrib("kg kg^-1", "specific total moisture", "specific_total_moisture"),
    )
    RQAVars["e"] = DiagnosticVariable(
        "e",
        var_attrib("kg m^2 s^2", "specific total energy", "specific_total_energy"),
    )
end


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

abstract type DiagnosticsGroupParams end

"""
    DiagnosticsGroup

Holds a set of diagnostics that share a collection interval, a filename
prefix, an output writer, an interpolation, and any extra parameters.
"""
mutable struct DiagnosticsGroup{DGP <: Union{Nothing, DiagnosticsGroupParams}}
    name::String
    init::Function
    fini::Function
    collect::Function
    interval::Int
    out_prefix::String
    out_dir::String
    writer::AbstractWriter
    interpol::Union{Nothing, InterpolationTopology}
    params::DGP

    DiagnosticsGroup(
        name,
        init,
        fini,
        collect,
        interval,
        out_prefix,
        out_dir,
        writer,
        interpol,
        params = nothing,
    ) = new{typeof(params)}(
        name,
        init,
        fini,
        collect,
        interval,
        out_prefix,
        out_dir,
        writer,
        interpol,
        params,
    )
end



function setup_atmos_default_diagnostics(
    ::Simulation,
    interval::Int,
    out_prefix::String,
    out_dir::String;
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
        out_dir,
        writer,
        interpol,
    )
end

# Declare all (3D) variables for this diagnostics group
function vars_atmos_gcm_default_simple_3d(::DryAtmosModel, FT)
    @vars begin
        u::FT
        v::FT
        w::FT
        rho::FT
        q::FT
      	e::FT
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
    vars.rho = state_prognostic.ρ
    vars.q = state_prognostic.ρq / state_prognostic.ρ
    vars.e = state_prognostic.ρe / state_prognostic.ρ

    return nothing
end

"""
    atmos_gcm_default_init(simulation, interval, interpol, currtime)

Initialize the GCM default diagnostics group, establishing the output file's
dimensions and variables.
"""
function atmos_gcm_default_init(dgngrp::DiagnosticsGroup, simulation::Simulation, currtime)
    @warn "Entered NetCDF init" maxlog = 1
    interpol = dgngrp.interpol
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
        init_diag_vars()
        vars = OrderedDict()
        varnames = flattenednames(vars_atmos_gcm_default_simple_3d(atmos, FT))
        @show(varnames, RQAVars)
        for varname in varnames
            var = RQAVars[varname] 
            vars[varname] = (tuple(collect(keys(dims))...), FT, var.attrib)
        end

        # create the output file
        dprefix = @sprintf("%s_%s", dgngrp.out_prefix, dgngrp.name)
        # TODO: Fix pointer to output_dir here ...
        dfilename = joinpath(dgngrp.out_dir, dprefix)

        ## TODO: access Settings in Driver.jl
        noov = false # Settings.no_overwrite
        init_data(dgngrp.writer, dfilename, noov, dims, vars)
    end

    @warn "Debug: Completed NETCDF Callback Initialisation" maxlog = 1

    return nothing
end

"""
    atmos_gcm_default_collect(dgngrp::DiagnosticsGroup, simulation::Simulation, currtime)

    Master function that performs a global grid traversal to compute various
    diagnostics using the above functions. Modified for compatibility with RQA interface.
"""
function atmos_gcm_default_collect(dgngrp::DiagnosticsGroup, simulation::Simulation, currtime)
    interpol = dgngrp.interpol
    interval = dgngrp.interval
    @warn "Entered collect function" maxlog = 1
    if !(interpol isa InterpolationCubedSphere)
        @warn """
            Diagnostics ($dgngrp.name): currently requires `InterpolationCubedSphere`!
            """
        return nothing
    end
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
    end

    # Interpolate the state, thermo, dgdiags and dyn vars to sphere (u and vorticity
    # need projection to zonal, merid). All this may happen on the GPU.
    istate =
        ArrayType{FT}(undef, interpol.Npl, number_states(atmos, Prognostic()))
    interpolate_local!(interpol, Q.realdata, istate)

    # TODO: get indices here without hard-coding them
    _ρu, _ρv, _ρw = 2, 3, 4
    project_cubed_sphere!(interpol, istate, (_ρu, _ρv, _ρw))

    # FIXME: accumulating to rank 0 is not scalable
    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)

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
        varnames = flattenednames(vars_atmos_gcm_default_simple_3d(atmos, FT))
        for (vari, varname) in enumerate(varnames)
            varvals[varname] = simple_3d_vars_array[:, :, :, vari]
        end
        # write output
        append_data(dgngrp.writer, varvals, currtime)
    end
    
    @warn "Debug: executed collect operation" maxlog = 1

    MPI.Barrier(mpicomm)
    return nothing
end # function collect

function atmos_gcm_default_fini(dgngrp::DiagnosticsGroup, simulation::Simulation, currtime) 
    @warn "Finishing NETCDF callback operation" maxlog = 1
end
