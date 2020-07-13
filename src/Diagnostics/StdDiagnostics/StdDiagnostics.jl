"""
    StdDiagnostics

This module defines many standard diagnostic variables and groups that may
be used directly by experiments.
"""
module StdDiagnostics

using CLIMAParameters
using CLIMAParameters.Planet
using CLIMAParameters.Atmos
using CLIMAParameters.SubgridScale

using KernelAbstractions
using MPI
using OrderedCollections
using Printf
using StaticArrays

using ..Diagnostics # until old diagnostics groups are removed
using ..Atmos
using ..BalanceLaws
using ..ConfigTypes
using ..DGMethods
using ..DiagnosticsMachine
import ..DiagnosticsMachine:
    Settings,
    dv_name,
    dv_attrib,
    dv_args,
    dv_project,
    dv_scale,
    dv_PointwiseDiagnostic,
    dv_HorizontalAverage
using ..Mesh.Grids
using ..Mesh.Interpolation
using ..Mesh.Topologies
using ..MPIStateArrays
using ..Thermodynamics
using ..TurbulenceClosures
using ..VariableTemplates
using ..Writers


# Pre-defined diagnostic variables

# Atmos
include("atmos_les_diagnostic_vars.jl")
include("atmos_gcm_diagnostic_vars.jl")


# Pre-defined diagnostics groups

# Atmos
include("atmos_les_default.jl")
include("atmos_gcm_default.jl")

end # module StdDiagnostics
