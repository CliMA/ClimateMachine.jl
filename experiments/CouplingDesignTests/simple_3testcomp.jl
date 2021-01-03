using Test
using MPI

using ClimateMachine

# To test coupling
using ClimateMachine.Coupling

# To create meshes (borrowed from Ocean for now!)
using ClimateMachine.Ocean.Domains

ClimateMachine.init()

# Use toy balance law for now
include("CplTestingBL.jl")

# Make some meshes covering same space laterally.
domainA = RectangularDomain(
    Ne = (10, 10, 5),
    Np = 4,
    x = (0, 1e6),
    y = (0, 1e6),
    z = (0, 1e5),
    periodicity = (true, true, false),
)
domainO = RectangularDomain(
    Ne = (10, 10, 4),
    Np = 4,
    x = (0, 1e6),
    y = (0, 1e6),
    z = (-4e3, 0),
    periodicity = (true, true, false),
)
domainL = RectangularDomain(
    Ne = (10, 10, 1),
    Np = 4,
    x = (0, 1e6),
    y = (0, 1e6),
    z = (0, 1),
    periodicity = (true, true, false),
)

# Create 3 components - one on each domain, for now all are instances
# of the same balance law
mA=Coupling.CplTestModel(;domain=domainA,BL_module=CplTestingBL)
mO=Coupling.CplTestModel(;domain=domainO,BL_module=CplTestingBL)
mL=Coupling.CplTestModel(;domain=domainL,BL_module=CplTestingBL)


# Instantiate a coupled timestepper that steps forward the components and
# maps imports and exports between components.
