"""
    DiagnosticsMachine

This module provides the infrastructure to extract diagnostics from a
ClimateMachine simulation. Two key abstractions are defined: diagnostic
variables and diagnostic groups. The `StdDiagnostics` module makes use of
these to define many standard variables and groups which may be used
directly by experiments. `DiagnosticsMachine` may be used by experiments
to define specialized variables and groups.
"""
module DiagnosticsMachine

export DiagnosticVar,
    PointwiseDiagnostic,
    @pointwise_diagnostic,
    dv_PointwiseDiagnostic,
    HorizontalAverage,
    @horizontal_average,
    dv_HorizontalAverage,
    States,
    #DiagnosticsGroup,
    @diagnostics_group
#DiagnosticsGroupParams

using CUDA
using Dates
using InteractiveUtils
using KernelAbstractions
using MacroTools
using MacroTools: prewalk
using MPI
using OrderedCollections
using Printf
using StaticArrays

using ..Diagnostics # until old diagnostics groups are removed
using ..Atmos
using ..BalanceLaws
using ..ConfigTypes
using ..DGMethods
using ..GenericCallbacks
using ..Mesh.Grids
using ..Mesh.Interpolation
using ..Mesh.Topologies
using ..MPIStateArrays
using ..Spectra
using ..TicToc
using ..VariableTemplates
using ..Writers

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius

# Container to store simulation information necessary for all
# diagnostics groups.
Base.@kwdef mutable struct DiagnosticSettings
    mpicomm::MPI.Comm = MPI.COMM_WORLD
    param_set::Union{Nothing, AbstractParameterSet} = nothing
    dg::Union{Nothing, SpaceDiscretization} = nothing
    Q::Union{Nothing, MPIStateArray} = nothing
    starttime::Union{Nothing, String} = nothing
    output_dir::Union{Nothing, String} = nothing
    no_overwrite::Bool = false
end
const Settings = DiagnosticSettings()

include("onetime.jl")
include("variables.jl")
include("groups.jl")

const AllDiagnosticVars = OrderedDict{
    Type{<:ClimateMachineConfigType},
    OrderedDict{String, DiagnosticVar},
}()
function add_all_dvar_dicts(T::DataType)
    AllDiagnosticVars[T] = OrderedDict{String, DiagnosticVar}()
    for t in subtypes(T)
        add_all_dvar_dicts(t)
    end
end
add_all_dvar_dicts(ClimateMachineConfigType)

"""
    init(mpicomm, param_set, dg, Q, starttime, output_dir)

Save the parameters into `Settings`, a container for simulation
information necessary for all diagnostics groups.
"""
function init(
    mpicomm::MPI.Comm,
    param_set::AbstractParameterSet,
    dg::SpaceDiscretization,
    Q::MPIStateArray,
    starttime::String,
    output_dir::String,
    no_overwrite::Bool,
)
    Settings.mpicomm = mpicomm
    Settings.param_set = param_set
    Settings.dg = dg
    Settings.Q = Q
    Settings.starttime = starttime
    Settings.output_dir = output_dir
    Settings.no_overwrite = no_overwrite
end

include("atmos_diagnostic_funs.jl")

end # module DiagnosticsMachine
