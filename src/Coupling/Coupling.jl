module Coupling

using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.ODESolvers

import ClimateMachine.Ocean.Domains: DiscontinuousSpectralElementGrid

include("CplTestingBL.jl")

struct CplTestModel{G,E,S,TS}
         grid::G
         equations::E
         state::S
         stepper::TS
end

"""
    CplTestModel(;domain=nothing)

Builds an instance of the coupling experimentation test model.
"""
function CplTestModel(;
                      domain=nothing,
                     )

        FT = eltype(domain)

        grid = nothing
        grid = DiscontinuousSpectralElementGrid(
         domain;
        )

        equations=nothing
        bl_prop=CplTestingBL.prop_defaults()
        equations=CplTestingBL.CplTestBL{FT}(;bl_prop=bl_prop)

        discretization=nothing
        discretization = DGModel(equations,grid,RusanovNumericalFlux(),CentralNumericalFluxSecondOrder(),CentralNumericalFluxGradient())

        state=nothing
        state=init_ode_state(discretization, FT(0); init_on_cpu = true)

        stepper=nothing
        stepper=LSRK54CarpenterKennedy(discretization,state,
                                  dt=1.,
                                  t0=0.)

        return CplTestModel(
                             grid,
                             equations,
                             state,
                             stepper,
                            )
end

end
