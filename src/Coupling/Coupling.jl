"""
    Coupling

Primitive coupling module sufficient for initial atmos-ocean-land coupled simulation.
"""
module Coupling

export CplTestModel
export CplState

using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.ODESolvers
using ClimateMachine.ODESolvers: AbstractODESolver

import ClimateMachine.Ocean.Domains: DiscontinuousSpectralElementGrid

include("CplTestModel.jl")
include("CplState.jl")

end
