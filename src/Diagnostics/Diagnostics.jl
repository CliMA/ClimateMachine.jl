"""
    Diagnostics

Accumulate mean fields and covariance statistics on the computational grid.

"""
module Diagnostics

export DiagnosticsGroup,
    setup_atmos_default_diagnostics,
    setup_atmos_core_diagnostics,
    setup_dump_state_and_aux_diagnostics,
    VecGrad,
    compute_vec_grad,
    SphericalCoord,
    compute_vec_grad_spherical,
    Vorticity,
    compute_vorticity

using Dates
using FileIO
using JLD2
using MPI
using OrderedCollections
using Printf
using StaticArrays

using ClimateMachine
using ..DGmethods
using ..DGmethods:
    number_state_conservative,
    vars_state_conservative,
    number_state_auxiliary,
    vars_state_auxiliary,
    vars_state_gradient_flux,
    number_state_gradient_flux
using ..Mesh.Interpolation
using ..MPIStateArrays
using ..VariableTemplates
using ..Writers

Base.@kwdef mutable struct Diagnostic_Settings
    mpicomm::MPI.Comm = MPI.COMM_WORLD
    dg::Union{Nothing, DGModel} = nothing
    Q::Union{Nothing, MPIStateArray} = nothing
    starttime::Union{Nothing, String} = nothing
    output_dir::Union{Nothing, String} = nothing
end
const Settings = Diagnostic_Settings()

"""
    init(mpicomm, dg, Q, starttime, output_dir)

Initialize the diagnostics collection module -- save the parameters into `Settings`.
"""
function init(
    mpicomm::MPI.Comm,
    dg::DGModel,
    Q::MPIStateArray,
    starttime::String,
    output_dir::String,
)
    Settings.mpicomm = mpicomm
    Settings.dg = dg
    Settings.Q = Q
    Settings.starttime = starttime
    Settings.output_dir = output_dir
end

"""
    DiagnosticsGroup

Holds a set of diagnostics that share a collection interval, a filename prefix,
and an interpolation.

TODO: to be completed; will be restructured.
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
    project::Bool
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
        project,
    ) = new(
        name,
        init,
        fini,
        collect,
        interval,
        out_prefix,
        writer,
        interpol,
        project,
        0,
    )
end

function (dgngrp::DiagnosticsGroup)(currtime; init = false, fini = false)
    if init
        dgngrp.init(dgngrp, currtime)
    else
        dgngrp.collect(dgngrp, currtime)
    end
    if fini
        dgngrp.fini(dgngrp, currtime)
    end
    dgngrp.num += 1
end

"""
    setup_atmos_default_diagnostics(
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
        project = true,
    )

Create and return a `DiagnosticsGroup` containing the "AtmosDefault"
diagnostics. All the diagnostics in the group will run at the specified
`interval`, be interpolated to the specified boundaries and resolution,
and will be written to files prefixed by `out_prefix` using `writer`.
"""
function setup_atmos_default_diagnostics(
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
    project = true,
)
    return DiagnosticsGroup(
        "AtmosDefault",
        Diagnostics.atmos_default_init,
        Diagnostics.atmos_default_fini,
        Diagnostics.atmos_default_collect,
        interval,
        out_prefix,
        writer,
        interpol,
        project,
    )
end

"""
    setup_atmos_core_diagnostics(
            interval::Int,
            out_prefix::String;
            writer::AbstractWriter,
            interpol = nothing,
            project  = true)

Create and return a `DiagnosticsGroup` containing the "AtmosCore"
diagnostics. All the diagnostics in the group will run at the
specified `interval`, be interpolated to the specified boundaries
and resolution, and will be written to files prefixed by `out_prefix`
using `writer`.
"""
function setup_atmos_core_diagnostics(
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
    project = true,
)
    return DiagnosticsGroup(
        "AtmosCore",
        Diagnostics.atmos_core_init,
        Diagnostics.atmos_core_fini,
        Diagnostics.atmos_core_collect,
        interval,
        out_prefix,
        writer,
        interpol,
        project,
    )
end

"""
    setup_dump_state_and_aux_diagnostics(
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
        project = true,
    )

Create and return a `DiagnosticsGroup` containing a diagnostic that
simply dumps the state and aux variables at the specified `interval`
after being interpolated, into NetCDF files prefixed by `out_prefix`.
"""
function setup_dump_state_and_aux_diagnostics(
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
    project = true,
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
        project,
    )
end

"""
    visitQ()

Helper macro to iterate over the DG grid. Generates the needed loops
and indices: `eh`, `ev`, `e`, `k,`, `j`, `i`, `ijk`.
"""
macro visitQ(nhorzelem, nvertelem, Nqk, Nq, expr)
    return esc(quote
        for eh in 1:nhorzelem
            for ev in 1:nvertelem
                e = ev + (eh - 1) * nvertelem
                for k in 1:Nqk
                    for j in 1:Nq
                        for i in 1:Nq
                            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
                            $expr
                        end
                    end
                end
            end
        end
    end)
end

# Helpers to extract data from the various state arrays
function extract_state_conservative(dg, state_conservative, ijk, e)
    bl = dg.balance_law
    FT = eltype(state_conservative)
    num_state_conservative = number_state_conservative(bl, FT)
    local_state_conservative = MArray{Tuple{num_state_conservative}, FT}(undef)
    for s in 1:num_state_conservative
        local_state_conservative[s] = state_conservative[ijk, s, e]
    end
    return Vars{vars_state_conservative(bl, FT)}(local_state_conservative)
end
function extract_state_auxiliary(dg, state_auxiliary, ijk, e)
    bl = dg.balance_law
    FT = eltype(state_auxiliary)
    num_state_auxiliary = number_state_auxiliary(bl, FT)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
    for s in 1:num_state_auxiliary
        local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
    end
    return Vars{vars_state_auxiliary(bl, FT)}(local_state_auxiliary)
end
function extract_state_gradient_flux(dg, state_gradient_flux, ijk, e)
    bl = dg.balance_law
    FT = eltype(state_gradient_flux)
    num_state_gradient_flux = number_state_gradient_flux(bl, FT)
    local_state_gradient_flux =
        MArray{Tuple{num_state_gradient_flux}, FT}(undef)
    for s in 1:num_state_gradient_flux
        local_state_gradient_flux[s] = state_gradient_flux[ijk, s, e]
    end
    return Vars{vars_state_gradient_flux(bl, FT)}(local_state_gradient_flux)
end

include("atmos_common.jl")
include("thermo.jl")
include("atmos_default.jl")
include("atmos_core.jl")
include("dump_state_and_aux.jl")
include("diagnostic_fields.jl")

end # module Diagnostics
