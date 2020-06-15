"""
    DiagnosticsGroup

Holds a set of diagnostics that share a collection interval, a filename
prefix, and an interpolation.

TODO: will be restructured to be defined as groups of `DiagnosticVariable`s.
"""
mutable struct DiagnosticsGroup
    name::String
    init::Function
    fini::Function
    collect::Function
    interval::String
    out_prefix::String
    writer::AbstractWriter
    interpol::Union{Nothing, InterpolationTopology}

    DiagnosticsGroup(
        name,
        init,
        fini,
        collect,
        interval,
        out_prefix,
        writer,
        interpol,
    ) = new(name, init, fini, collect, interval, out_prefix, writer, interpol)
end


function GenericCallbacks.init!(dgngrp::DiagnosticsGroup, solver, Q, param, t0)
    @info @sprintf(
        """
    Diagnostics: %s
        %s at %8.2f""",
        dgngrp.name,
        "initializing",
        t0,
    )
    dgngrp.init(dgngrp, t0)
    dgngrp.collect(dgngrp, t0)
    return nothing
end
function GenericCallbacks.call!(dgngrp::DiagnosticsGroup, solver, Q, param, t)
    @tic diagnostics
    @info @sprintf(
        """
    Diagnostics: %s
        %s at %8.2f""",
        dgngrp.name,
        "collecting",
        t,
    )
    dgngrp.collect(dgngrp, t)
    @toc diagnostics
    return nothing
end


include("atmos_les_default.jl")
"""
    setup_atmos_default_diagnostics(
        ::AtmosLESConfigType,
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
    )

Create and return a `DiagnosticsGroup` containing the "AtmosDefault"
diagnostics for LES configurations. All the diagnostics in the group will
run at the specified `interval`, be interpolated to the specified boundaries
and resolution, and will be written to files prefixed by `out_prefix` using
`writer`.
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

include("atmos_gcm_default.jl")
"""
    setup_atmos_default_diagnostics(
        ::AtmosGCMConfigType,
        interval::String,
        out_prefix::String;
        writer::AbstractWriter,
        interpol = nothing,
    )

Create and return a `DiagnosticsGroup` containing the "AtmosDefault"
diagnostics for GCM configurations. All the diagnostics in the group will run
at the specified `interval`, be interpolated to the specified boundaries and
resolution, and will be written to files prefixed by `out_prefix` using
`writer`.
"""
function setup_atmos_default_diagnostics(
    ::AtmosGCMConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
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
    )
end

include("atmos_les_core.jl")
"""
    setup_atmos_core_diagnostics(
        ::AtmosLESConfigType,
        interval::String,
        out_prefix::String;
        writer::AbstractWriter,
        interpol = nothing,
    )

Create and return a `DiagnosticsGroup` containing the "AtmosLESCore"
diagnostics for LES configurations. All the diagnostics in the group
will run at the specified `interval`, be interpolated to the specified
boundaries and resolution, and will be written to files prefixed by
`out_prefix` using `writer`.
"""
function setup_atmos_core_diagnostics(
    ::AtmosLESConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
    # TODO: remove this
    @assert isnothing(interpol)

    return DiagnosticsGroup(
        "AtmosLESCore",
        Diagnostics.atmos_les_core_init,
        Diagnostics.atmos_les_core_fini,
        Diagnostics.atmos_les_core_collect,
        interval,
        out_prefix,
        writer,
        interpol,
    )
end

include("atmos_les_default_perturbations.jl")
"""
    setup_atmos_default_perturbations(
        ::AtmosLESConfigType,
        interval::String,
        out_prefix::String;
        writer::AbstractWriter,
        interpol = nothing,
    )

Create and return a `DiagnosticsGroup` containing the
"AtmosLESDefaultPerturbations" diagnostics for the LES configuration.
All the diagnostics in the group will run at the specified `interval`,
and written to files prefixed by `out_prefix` using `writer`.
"""
function setup_atmos_default_perturbations(
    ::AtmosLESConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
    # TODO: remove this
    @assert !isnothing(interpol)

    return DiagnosticsGroup(
        "AtmosLESDefaultPerturbations",
        Diagnostics.atmos_les_default_perturbations_init,
        Diagnostics.atmos_les_default_perturbations_fini,
        Diagnostics.atmos_les_default_perturbations_collect,
        interval,
        out_prefix,
        writer,
        interpol,
    )
end

include("atmos_refstate_perturbations.jl")
"""
    setup_atmos_refstate_perturbations(
        ::ClimateMachineConfigType,
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
    )

Create and return a `DiagnosticsGroup` containing the
"AtmosRefStatePerturbations" diagnostics for both LES and GCM
configurations. All the diagnostics in the group will run at the
specified `interval`, be interpolated to the specified boundaries
and resolution, and written to files prefixed by `out_prefix`
using `writer`.
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

include("dump_state.jl")
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

include("dump_aux.jl")
"""
    setup_dump_aux_diagnostics(
        ::ClimateMachineConfigType,
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
    )

Create and return a `DiagnosticsGroup` containing a diagnostic that
simply dumps the auxiliary state variables at the specified
`interval` after being interpolated, into NetCDF files prefixed by
`out_prefix`.
"""
function setup_dump_aux_diagnostics(
    ::ClimateMachineConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
    # TODO: remove this
    @assert !isnothing(interpol)

    return DiagnosticsGroup(
        "DumpAux",
        Diagnostics.dump_aux_init,
        Diagnostics.dump_aux_fini,
        Diagnostics.dump_aux_collect,
        interval,
        out_prefix,
        writer,
        interpol,
    )
end
