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

using ..DGmethods
using ..DGmethods: num_state, vars_state, num_aux, vars_aux, vars_diffusive, num_diffusive
using ..Mesh.Topologies
using ..Mesh.Grids
using ..MPIStateArrays
using ..VariableTemplates

Base.@kwdef mutable struct Diagnostic_Settings
    mpicomm::MPI.Comm = MPI.COMM_WORLD
    dg::Union{Nothing,DGModel} = nothing
    Q::Union{Nothing,MPIStateArray} = nothing
    starttime::Union{Nothing,String} = nothing
    outdir::Union{Nothing,String} = nothing
end

const Settings = Diagnostic_Settings()

include("diagnostic_vars.jl")

"""
    visitQ()

Helper macro to iterate over the DG grid. Generates the needed loops
and indices: `eh`, `ev`, `e`, `k,`, `j`, `i`, `ijk`.
"""
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

# Helpers to extract data from `Q`, etc.
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

include("AtmosDiagnostics.jl")

"""
    init(mpicomm, dg, Q, starttime, outdir)

Initialize the diagnostics collection module.
"""
function init(mpicomm::MPI.Comm,
              dg::DGModel,
              Q::MPIStateArray,
              starttime::String,
              outdir::String)
    Settings.mpicomm = mpicomm
    Settings.dg = dg
    Settings.Q = Q
    Settings.starttime = starttime
    Settings.outdir = outdir

    init(dg.balancelaw, mpicomm, dg, Q, starttime)
end

"""
    collect(currtime)

Collect various diagnostics specific to the balance law and write them to a
JLD2 file, indexed by `currtime`, in the output directory specified in
`init()`.
"""
function collect(currtime)
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)

    current_time = string(currtime)

    # make sure this time step is not already recorded
    docollect = [false]
    if mpirank == 0
        try
            jldopen(joinpath(Settings.outdir,
                    "diagnostics-$(Settings.starttime).jld2"), "r") do file
                diagnostics = file[current_time]
                @warn "diagnostics for time step $(current_time) already collected"
        end
        catch e
            docollect[1] = true
        end
    end
    MPI.Bcast!(docollect, 0, mpicomm)
    if !docollect[1]
        return nothing
    end

    # collect diagnostics
    diagnostics = collect(Settings.dg.balancelaw, currtime)

    # write results
    if mpirank == 0 && diagnostics !== nothing
        jldopen(joinpath(Settings.outdir,
                "diagnostics-$(Settings.starttime).jld2"), "a+") do file
            file[current_time] = diagnostics
        end
    end

    return nothing
end # function collect

end # module Diagnostics

