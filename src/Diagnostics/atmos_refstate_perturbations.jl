# AtmosRefStatePerturbations
#
# Computes perturbations from the reference state and outputs them
# on the specified interpolated grid.

import CUDA
using ..Atmos
using ..Mesh.Topologies
using ..Mesh.Grids
using ..Thermodynamics

"""
    setup_atmos_refstate_perturbations(
        ::ClimateMachineConfigType,
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
    )

Create the "AtmosRefStatePerturbations" `DiagnosticsGroup` which contains
perturbations from the (hydrostatic) reference state for:

- rho: air density
- pres: air pressure
- temp: air temperature
- et: total specific energy
- qt: mass fraction of total water in air (NaN when not an `EquilMoist` moisture model)

The perturbations are computed from variables on an interpolated grid
(`interpol` _must_ be specified) and output on `x`, `y`, `z` or `lat`,
`long`, `level` dimensions as well as a (unlimited) `time` dimension
at the specified `interval`.
"""
function setup_atmos_refstate_perturbations(
    ::ClimateMachineConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
    # TODO: remove this
    @assert !isnothing(interpol)

    return DiagnosticsGroup(
        "AtmosRefStatePerturbations",
        Diagnostics.atmos_refstate_perturbations_init,
        Diagnostics.atmos_refstate_perturbations_fini,
        Diagnostics.atmos_refstate_perturbations_collect,
        interval,
        out_prefix,
        writer,
        interpol,
    )
end

function vars_atmos_refstate_perturbations(m::AtmosModel, FT)
    @vars begin
        ref_state::vars_atmos_refstate_perturbations(m.ref_state, FT)
    end
end
vars_atmos_refstate_perturbations(::ReferenceState, FT) = @vars()
function vars_atmos_refstate_perturbations(rs::HydrostaticState, FT)
    @vars begin
        rho::FT
        pres::FT
        temp::FT
        et::FT
        qt::FT
    end
end
num_atmos_refstate_perturbation_vars(m, FT) =
    varsize(vars_atmos_refstate_perturbations(m, FT))
atmos_refstate_perturbation_vars(m, array) =
    Vars{vars_atmos_refstate_perturbations(m, eltype(array))}(array)

function atmos_refstate_perturbations!(
    atmos::AtmosModel,
    state,
    aux,
    thermo,
    vars,
)
    atmos_refstate_perturbations!(
        atmos.ref_state,
        atmos,
        state,
        aux,
        thermo,
        vars,
    )
    return nothing
end
function atmos_refstate_perturbations!(
    ::ReferenceState,
    ::AtmosModel,
    state,
    aux,
    thermo,
    vars,
)
    return nothing
end
function atmos_refstate_perturbations!(
    rs::HydrostaticState,
    atmos::AtmosModel,
    state,
    aux,
    thermo,
    vars,
)
    vars.ref_state.rho = state.ρ - aux.ref_state.ρ
    vars.ref_state.pres = thermo.pres - aux.ref_state.p
    vars.ref_state.temp = thermo.temp - aux.ref_state.T
    vars.ref_state.et =
        (state.energy.ρe / state.ρ) - (aux.ref_state.ρe / aux.ref_state.ρ)
    # FIXME properly
    if atmos.moisture isa EquilMoist
        vars.ref_state.qt =
            (thermo.moisture.ρq_tot / state.ρ) -
            (aux.ref_state.ρq_tot / aux.ref_state.ρ)
    else
        vars.ref_state.qt = NaN
    end

    return nothing
end

function atmos_refstate_perturbations_init(dgngrp::DiagnosticsGroup, currtime)
    FT = eltype(Settings.Q)
    atmos = Settings.dg.balance_law
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)

    if mpirank == 0
        # get dimensions for the interpolated grid
        dims = dimensions(dgngrp.interpol)

        # adjust the level dimension for `planet_radius` on 'CubedSphere's
        if dgngrp.interpol isa InterpolationCubedSphere
            level_val = dims["level"]
            dims["level"] = (
                level_val[1] .- FT(planet_radius(Settings.param_set)),
                level_val[2],
            )
        end

        # set up the variables we're going to be writing
        vars = OrderedDict()
        varnames = map(
            s -> startswith(s, "ref_state.") ? s[11:end] : s,
            flattenednames(vars_atmos_refstate_perturbations(atmos, FT)),
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
    atmos_refstate_perturbations_collect(dgngrp, currtime)

Perform a global grid traversal to compute various diagnostics.
"""
function atmos_refstate_perturbations_collect(
    dgngrp::DiagnosticsGroup,
    currtime,
)
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    mpirank = MPI.Comm_rank(mpicomm)
    atmos = dg.balance_law
    if !isa(atmos.ref_state, HydrostaticState)
        @warn """
            Diagnostics $(dgngrp.name): has useful output only for `HydrostaticState`
            """
    end
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
    if array_device(Q) isa CPU
        ArrayType = Array
        state_data = Q.realdata
        aux_data = dg.state_auxiliary.realdata
    else
        ArrayType = CUDA.CuArray
        state_data = Array(Q.realdata)
        aux_data = Array(dg.state_auxiliary.realdata)
    end
    FT = eltype(state_data)

    # Compute thermo variables
    thermo_array = Array{FT}(undef, npoints, num_thermo(atmos, FT), nrealelem)
    @traverse_dg_grid grid_info topl_info begin
        state = extract_state(dg, state_data, ijk, e, Prognostic())
        aux = extract_state(dg, aux_data, ijk, e, Auxiliary())

        thermo = thermo_vars(atmos, view(thermo_array, ijk, :, e))
        compute_thermo!(atmos, state, aux, thermo)
    end

    # Interpolate the state and thermo variables.
    interpol = dgngrp.interpol
    istate =
        ArrayType{FT}(undef, interpol.Npl, number_states(atmos, Prognostic()))
    interpolate_local!(interpol, Q.realdata, istate)

    if interpol isa InterpolationCubedSphere
        # TODO: get indices here without hard-coding them
        _ρu, _ρv, _ρw = 2, 3, 4
        project_cubed_sphere!(interpol, istate, (_ρu, _ρv, _ρw))
    end

    iaux = ArrayType{FT}(undef, interpol.Npl, number_states(atmos, Auxiliary()))
    interpolate_local!(interpol, dg.state_auxiliary.realdata, iaux)

    ithermo = ArrayType{FT}(undef, interpol.Npl, num_thermo(atmos, FT))
    interpolate_local!(interpol, ArrayType(thermo_array), ithermo)

    # FIXME: accumulating to rank 0 is not scalable
    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)
    all_aux_data = accumulate_interpolated_data(mpicomm, interpol, iaux)
    all_thermo_data = accumulate_interpolated_data(mpicomm, interpol, ithermo)

    if mpirank == 0
        # get dimensions for the interpolated grid
        dims = dimensions(dgngrp.interpol)

        # set up the array for the diagnostic variables based on the interpolated grid
        nlong = length(dims["long"][1])
        nlat = length(dims["lat"][1])
        nlevel = length(dims["level"][1])

        perturbations_array = Array{FT}(
            undef,
            nlong,
            nlat,
            nlevel,
            num_atmos_refstate_perturbation_vars(atmos, FT),
        )

        @traverse_interpolated_grid nlong nlat nlevel begin
            statei = Vars{vars_state(atmos, Prognostic(), FT)}(view(
                all_state_data,
                lo,
                la,
                le,
                :,
            ))
            auxi = Vars{vars_state(atmos, Auxiliary(), FT)}(view(
                all_aux_data,
                lo,
                la,
                le,
                :,
            ))
            thermoi = thermo_vars(atmos, view(all_thermo_data, lo, la, le, :))

            perturbations = atmos_refstate_perturbation_vars(
                atmos,
                view(perturbations_array, lo, la, le, :),
            )
            atmos_refstate_perturbations!(
                atmos,
                statei,
                auxi,
                thermoi,
                perturbations,
            )
        end

        varvals = OrderedDict()
        varnames = map(
            s -> startswith(s, "ref_state.") ? s[11:end] : s,
            flattenednames(vars_atmos_refstate_perturbations(atmos, FT)),
        )
        for (vari, varname) in enumerate(varnames)
            varvals[varname] = perturbations_array[:, :, :, vari]
        end

        # write output
        append_data(dgngrp.writer, varvals, currtime)
    end

    MPI.Barrier(mpicomm)
    return nothing
end # function collect

function atmos_refstate_perturbations_fini(dgngrp::DiagnosticsGroup, currtime) end
