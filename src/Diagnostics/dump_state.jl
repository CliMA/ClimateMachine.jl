"""
    setup_dump_state_diagnostics(
        ::ClimateMachineConfigType,
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
    )

Create and return a `DiagnosticsGroup` containing a diagnostic that
simply dumps the conservative state variables at the specified
`interval` after being interpolated, into NetCDF files prefixed by
`out_prefix`.
"""
function setup_dump_state_diagnostics(
    ::ClimateMachineConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
    # TODO: remove this
    @assert !isnothing(interpol)

    return DiagnosticsGroup(
        "DumpState",
        Diagnostics.dump_state_init,
        Diagnostics.dump_state_fini,
        Diagnostics.dump_state_collect,
        interval,
        out_prefix,
        writer,
        interpol,
    )
end

dump_state_init(dgngrp, currtime) = dump_init(dgngrp, currtime, Prognostic())

function dump_state_collect(dgngrp, currtime)
    interpol = dgngrp.interpol
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    FT = eltype(Q.data)
    bl = dg.balance_law
    mpirank = MPI.Comm_rank(mpicomm)

    istate = similar(Q.data, interpol.Npl, number_states(bl, Prognostic()))
    interpolate_local!(interpol, Q.data, istate)

    if interpol isa InterpolationCubedSphere
        # TODO: get indices here without hard-coding them
        _ρu, _ρv, _ρw = 2, 3, 4
        project_cubed_sphere!(interpol, istate, (_ρu, _ρv, _ρw))
    end

    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)

    if mpirank == 0
        statenames = flattenednames(vars_state(bl, Prognostic(), FT))
        varvals = OrderedDict()
        for (vari, varname) in enumerate(statenames)
            varvals[varname] = all_state_data[:, :, :, vari]
        end
        append_data(dgngrp.writer, varvals, currtime)
    end

    MPI.Barrier(mpicomm)
    return nothing
end

function dump_state_fini(dgngrp, currtime) end
