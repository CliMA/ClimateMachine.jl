module SuperModels

using MPI

using ClimateMachine

using ClimateMachine: Settings

using ...DGMethods.NumericalFluxes

using ..OceanProblems: InitialValueProblem, InitialConditions
using ..Domains: array_type
using ..Ocean: FreeSlip, Impenetrable, Insulating, OceanBC, Penetrable
using ..Ocean.Fields: SpectralElementField

using ...Mesh.Filters: CutoffFilter, ExponentialFilter
using ...Mesh.Grids: polynomialorders, DiscontinuousSpectralElementGrid

include("hydrostatic_boussinesq_super_model.jl")

end # module
