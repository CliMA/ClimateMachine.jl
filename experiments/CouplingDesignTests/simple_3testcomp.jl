using Test
using MPI

using ClimateMachine

# To test coupling
using ClimateMachine.Coupling

# To create meshes (borrowed from Ocean for now!)
using ClimateMachine.Ocean.Domains

# To setup some callbacks
using ClimateMachine.GenericCallbacks

# To invoke timestepper
using ClimateMachine.ODESolvers

import ClimateMachine.Mesh.Grids: _x3

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
    periodicity = (true, true, true),
)
domainO = RectangularDomain(
    Ne = (10, 10, 4),
    Np = 4,
    x = (0, 1e6),
    y = (0, 1e6),
    z = (-4e3, 0),
    periodicity = (true, true, true),
)
domainL = RectangularDomain(
    Ne = (10, 10, 1),
    Np = 4,
    x = (0, 1e6),
    y = (0, 1e6),
    z = (0, 1),
    periodicity = (true, true, true),
)

# Create 3 components - one on each domain, for now all are instances
# of the same balance law
mA=Coupling.CplTestModel(;domain=domainA,BL_module=CplTestingBL, nsteps=5)
mO=Coupling.CplTestModel(;domain=domainO,BL_module=CplTestingBL, nsteps=2)
#mL=Coupling.CplTestModel(;domain=domainL,BL_module=CplTestingBL)
 
# Create a Coupler State object for holding imort/export fields.
cState=CplState( Dict(:AtmosAirSeaHeatFlux=>[ ], :OceanSST=>[ ] ) )

# I think each BL can have a pre- and post- couple function?
function postatmos(_)
    println(" Ocean component finished stepping...")
    println("Atmos export fill callback")
    # Pass atmos exports to "coupler" namespace
    # For now we use deepcopy here.
    cState.CplStateBlob[:AtmosAirSeaHeatFlux]=deepcopy(mA.discretization.state_auxiliary.boundary_out[mA.discretization.grid.vgeo[:,_x3:_x3,:] .== 0] )
    # mO.discretization.state_auxiliary.boundary_in[mO.discretization.grid.vgeo[:,_x3:_x3,:] .== 0] .= mA.discretization.state_auxiliary.boundary_out[mA.discretization.grid.vgeo[:,_x3:_x3,:] .== 0]
end

function postocean(_)
    println(" Ocean component finished stepping...")
    println("Ocean export fill callback")
    # Pass ocean exports to "coupler" namespace
    cState.CplStateBlob[:OceanSST]=deepcopy( mO.discretization.state_auxiliary.boundary_out[mO.discretization.grid.vgeo[:,_x3:_x3,:] .== 0] )
    # mA.discretization.state_auxiliary.boundary_in[mA.discretization.grid.vgeo[:,_x3:_x3,:] .== 0] .= mO.discretization.state_auxiliary.boundary_out[mO.discretization.grid.vgeo[:,_x3:_x3,:] .== 0]
end

function preatmos(_)
        println("Atmos import fill callback")
        mA.discretization.state_auxiliary.boundary_in[mA.discretization.grid.vgeo[:,_x3:_x3,:] .== 0] .= cState.CplStateBlob[:OceanSST]
        println(" Atmos component start stepping...")
        nothing
end
function preocean(_)
        println("Ocean import fill callback")
        println(" Ocean component start stepping...")
        nothing
end

# Instantiate a coupled timestepper that steps forward the components and
# implements mapings between components export bondary states and
# other components imports.

compA=(pre_step=preatmos,component_model=mA,post_step=postatmos)
compO=(pre_step=preocean,component_model=mO,post_step=postocean)
component_list=( atmosphere=compA,ocean=compO,)
cC=Coupling.CplSolver(component_list=component_list,
                      coupling_dt=5.,t0=0.)

# We also need to initialize the imports so they can be read.
cState.CplStateBlob[:OceanSST]=deepcopy( mO.discretization.state_auxiliary.boundary_out[mO.discretization.grid.vgeo[:,_x3:_x3,:] .== 0] )
cState.CplStateBlob[:AtmosAirSeaHeatFlux]=deepcopy(mA.discretization.state_auxiliary.boundary_out[mA.discretization.grid.vgeo[:,_x3:_x3,:] .== 0] )

# Invoke solve! with coupled timestepper and callback list.
solve!(nothing,cC;numberofsteps=2)
