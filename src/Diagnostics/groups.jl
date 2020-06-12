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
    num::Int

    DiagnosticsGroup(
        name,
        init,
        fini,
        collect,
        interval,
        out_prefix,
        writer,
        interpol,
    ) = new(
        name,
        init,
        fini,
        collect,
        interval,
        out_prefix,
        writer,
        interpol,
        0,
    )
end

function (dgngrp::DiagnosticsGroup)(currtime; init = false, fini = false)
    if init
        dgngrp.init(dgngrp, currtime)
    end
    dgngrp.collect(dgngrp, currtime)
    if fini
        dgngrp.fini(dgngrp, currtime)
    end
    dgngrp.num += 1
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
        interval::Int,
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
        interval::Int,
        out_prefix::String;
        writer::AbstractWriter,
        interpol = nothing,
    )

Create and return a `DiagnosticsGroup` containing the "AtmosLESCore" diagnostics
for LES configurations. All the diagnostics in the group will run at the
specified `interval`, be interpolated to the specified boundaries and
resolution, and will be written to files prefixed by `out_prefix` using
`writer`.
"""
function setup_atmos_core_diagnostics(
    ::AtmosLESConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
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

include("dump_state_and_aux.jl")
"""
    setup_dump_state_and_aux_diagnostics(
        ::ClimateMachineConfigType,
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
    )

Create and return a `DiagnosticsGroup` containing a diagnostic that
simply dumps the state and aux variables at the specified `interval`
after being interpolated, into NetCDF files prefixed by `out_prefix`.
"""
function setup_dump_state_and_aux_diagnostics(
    ::ClimateMachineConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
    return DiagnosticsGroup(
        "DumpStateAndAux",
        Diagnostics.dump_state_and_aux_init,
        Diagnostics.dump_state_and_aux_fini,
        Diagnostics.dump_state_and_aux_collect,
        interval,
        out_prefix,
        writer,
        interpol,
    )
end
