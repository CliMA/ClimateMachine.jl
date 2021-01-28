"""
    Diagnostics

Accumulate mean fields and covariance statistics on the computational grid.
"""
module Diagnostics

export DiagnosticsGroup,
    setup_atmos_default_diagnostics,
    setup_atmos_core_diagnostics,
    setup_atmos_default_perturbations,
    setup_atmos_refstate_perturbations,
    setup_atmos_turbulence_stats,
    setup_atmos_mass_energy_loss,
    setup_atmos_spectra_diagnostics,
    setup_dump_state_diagnostics,
    setup_dump_aux_diagnostics,
    setup_dump_tendencies_diagnostics

using CUDA
using Dates
using FileIO
using JLD2
using KernelAbstractions
using MPI
using OrderedCollections
using Printf
using StaticArrays
import KernelAbstractions: CPU

using ..BalanceLaws
using ..ConfigTypes
using ..DGMethods
using ..Mesh.Interpolation
using ..MPIStateArrays
using ..Spectra
using ..TicToc
using ..VariableTemplates
using ..Writers
import ..GenericCallbacks

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius

Base.@kwdef mutable struct Diagnostic_Settings
    mpicomm::MPI.Comm = MPI.COMM_WORLD
    param_set::Union{Nothing, AbstractParameterSet} = nothing
    dg::Union{Nothing, SpaceDiscretization} = nothing
    Q::Union{Nothing, MPIStateArray} = nothing
    starttime::Union{Nothing, String} = nothing
    output_dir::Union{Nothing, String} = nothing
end
const Settings = Diagnostic_Settings()

"""
    init(mpicomm, param_set, dg, Q, starttime, output_dir)

Initialize the diagnostics collection module -- save the parameters into
`Settings`.
"""
function init(
    mpicomm::MPI.Comm,
    param_set::AbstractParameterSet,
    dg::SpaceDiscretization,
    Q::MPIStateArray,
    starttime::String,
    output_dir::String,
)
    Settings.mpicomm = mpicomm
    Settings.param_set = param_set
    Settings.dg = dg
    Settings.Q = Q
    Settings.starttime = starttime
    Settings.output_dir = output_dir
end

include("variables.jl")
include("helpers.jl")
include("atmos_common.jl")
include("thermo.jl")
include("groups.jl")

"""
    __init()__

Module initialization function. Currently, only fills in all currently
defined diagnostic variables.
"""
function __init__()
    setup_variables()
end

end # module Diagnostics
