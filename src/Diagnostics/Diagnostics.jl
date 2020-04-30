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

using CLIMA
using ..DGmethods
using ..DGmethods:
    number_state_conservative, vars_state_conservative, number_state_auxiliary, vars_state_auxiliary, vars_state_gradient_flux, number_state_gradient_flux
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

# Helpers to extract data from `Q`, etc.
function extract_state(dg, localQ, ijk, e)
    bl = dg.balancelaw
    FT = eltype(localQ)
    nstate = number_state_conservative(bl, FT)
    l_Q = MArray{Tuple{nstate}, FT}(undef)
    for s in 1:nstate
        l_Q[s] = localQ[ijk, s, e]
    end
    return Vars{vars_state_conservative(bl, FT)}(l_Q)
end
function extract_aux(dg, state_auxiliary, ijk, e)
    bl = dg.balancelaw
    FT = eltype(state_auxiliary)
    nauxstate = number_state_auxiliary(bl, FT)
    l_aux = MArray{Tuple{nauxstate}, FT}(undef)
    for s in 1:nauxstate
        l_aux[s] = state_auxiliary[ijk, s, e]
    end
    return Vars{vars_state_auxiliary(bl, FT)}(l_aux)
end
function extract_diffusion(dg, localdiff, ijk, e)
    bl = dg.balancelaw
    FT = eltype(localdiff)
    ndiff = number_state_gradient_flux(bl, FT)
    l_diff = MArray{Tuple{ndiff}, FT}(undef)
    for s in 1:ndiff
        l_diff[s] = localdiff[ijk, s, e]
    end
    return Vars{vars_state_gradient_flux(bl, FT)}(l_diff)
end

include("atmos_common.jl")
include("thermo.jl")
include("atmos_default.jl")
include("atmos_core.jl")
include("dump_state_and_aux.jl")
include("diagnostic_fields.jl")

end # module Diagnostics
