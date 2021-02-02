using Test
using MPI
using Statistics
using ClimateMachine

# To test coupling
using ClimateMachine.Coupling

# To create meshes (borrowed from Ocean for now!)
using ClimateMachine.Ocean.Domains

# To setup some callbacks
using ClimateMachine.GenericCallbacks

# To invoke timestepper
using ClimateMachine.ODESolvers
using ClimateMachine.MPIStateArrays

import ClimateMachine.Mesh.Grids: _x3

import ClimateMachine.DGMethods.NumericalFluxes:
           NumericalFluxSecondOrder

using LinearAlgebra

ClimateMachine.init()
const FT = Float64;

# Use toy balance law for now
include("CplTestingBL.jl")
using .CplTestingBL


couple_dt = 300.0
nstepsA = 5
nstepsO = 5

# Make some meshes covering same space laterally.
Np=4
domainA = RectangularDomain(
    Ne = (10, 10, 5),
    Np = Np,
    x = (0, 1e6),
    y = (0, 1e6),
    z = (0, 1e5),
    periodicity = (true, true, false),
)
domainO = RectangularDomain(
    Ne = (10, 10, 4),
    Np = Np,
    x = (0, 1e6),
    y = (0, 1e6),
    z = (-4e3, 0),
    periodicity = (true, true, false),
)
domainL = RectangularDomain(
    Ne = (10, 10, 1),
    Np = Np,
    x = (0, 1e6),
    y = (0, 1e6),
    z = (0, 1),
    periodicity = (true, true, true),
)

# Set some paramters
#  Haney like relaxation 60-day time scale.
const τ_airsea=FT(60*86400)
#  Background atmos diffusivities
const κᵃʰ=FT(1e4)
const κᵃᶻ=FT(1e-2)
const κᵒʰ=FT(1e3)
const κᵒᶻ=FT(1e-4)

# Create 3 components - one on each domain, for now all are instances
# of the same balance law

# Atmos component
## Set atmosphere initial state function
function atmos_init_theta(xc,yc,zc,npt,el)
  return 30.
end
## Set atmosphere shadow boundary flux function
function atmos_theta_shadow_boundary_flux(θᵃ,θᵒ,npt,el,xc,yc,zc)
 if zc == 0.
  tflux=(1. / τ_airsea)*( θᵃ-θᵒ )
 else
  tflux=0.
 end
 return tflux
end
## Set atmsophere diffusion coeffs
function atmos_calc_kappa_diff(_...)
  return κᵃʰ,κᵃʰ,κᵃᶻ
end
## Set atmos source!
function atmos_source_theta(θᵃ,npt,el,xc,yc,zc,θᵒ)
  tsource = 0.
  if zc == 0.
    #tsource = -(1. / τ_airsea)*( θᵃ-θᵒ )
  end
  return tsource
end
## Create atmos component
bl_prop=CplTestingBL.prop_defaults()
bl_prop=(bl_prop..., init_theta=atmos_init_theta)
bl_prop=(bl_prop..., theta_shadow_boundary_flux=atmos_theta_shadow_boundary_flux)
bl_prop=(bl_prop..., calc_kappa_diff=atmos_calc_kappa_diff)
bl_prop=(bl_prop..., source_theta=atmos_source_theta)
mA=Coupling.CplTestModel(;domain=domainA,
    equations=CplTestBL(bl_prop, (CoupledPrimaryBoundary(), ExteriorBoundary())),
    nsteps=nstepsA, dt=couple_dt/nstepsA, NFSecondOrder=CplTestingBL.PenaltyNumFluxDiffusive() )

# Ocean component
## Set initial temperature profile
function ocean_init_theta(xc,yc,zc,npt,el)
  return 20.
end
## Set boundary source imported from atmos
function ocean_source_theta(θ,npt,el,xc,yc,zc,air_sea_flux_import)
  sval=0.
  if zc == 0.
   #sval=air_sea_flux_import
  end
  return sval
end
## Set ocean diffusion coeffs
function ocean_calc_kappa_diff(_...)
  # return κᵒʰ,κᵒʰ,κᵒᶻ*FT(100.)
  return κᵒʰ,κᵒʰ,κᵒᶻ # m^2 s^-1
end
## Set penalty term tau (for debugging)
function ocean_get_penalty_tau(_...)
  return FT(0.5)
end
## Create ocean component
bl_prop=CplTestingBL.prop_defaults()
bl_prop=(bl_prop..., init_theta=ocean_init_theta)
bl_prop=(bl_prop..., source_theta=ocean_source_theta)
bl_prop=(bl_prop..., calc_kappa_diff=ocean_calc_kappa_diff)
bl_prop=(bl_prop..., get_penalty_tau=ocean_get_penalty_tau)
mO=Coupling.CplTestModel(;domain=domainO,
    equations=CplTestBL(bl_prop, (ExteriorBoundary(), CoupledSecondaryBoundary())),
    nsteps=nstepsO, dt=couple_dt/nstepsO, NFSecondOrder=CplTestingBL.PenaltyNumFluxDiffusive() )

# No Land for now
#mL=Coupling.CplTestModel(;domain=domainL,BL_module=CplTestingBL)

# Create a Coupler State object for holding imort/export fields.
# Try using Dict here - not sure if that will be OK with GPU
cState=CplState( Dict(:Atmos_MeanAirSeaθFlux=>[ ], :Ocean_SST=>[ ] ) )

# I think each BL can have a pre- and post- couple function?

const boundaryA = mA.discretization.grid.vgeo[:,_x3:_x3,:] .== 0
const boundaryO = mO.discretization.grid.vgeo[:,_x3:_x3,:] .== 0

function preatmos(csolver)

  # Set boundary SST used in atmos to SST of ocean surface at start of coupling cycle.
  mA.discretization.state_auxiliary.θ_secondary[boundaryA] .= cState.CplStateBlob[:Ocean_SST]
  # Set atmos boundary flux accumulator to 0.
  mA.state.F_accum.=0

  @info("preatmos",
    time = csolver.t,
    total_θ_atmos = weightedsum(mA.state,1),
    total_θ_ocean = weightedsum(mO.state,1),
    total_θ = weightedsum(mA.state,1) + weightedsum(mO.state,1),
    atmos_θ_surface_max = maximum(mA.state.θ[boundaryA]),
    ocean_θ_surface_max = maximum(mO.state.θ[boundaryO]),
  )

end

function postatmos(csolver)

    # Pass atmos exports to "coupler" namespace
    # For now we use deepcopy here.
    # 1. Save mean θ flux at the Atmos boundary during the coupling period
    cState.CplStateBlob[:Atmos_MeanAirSeaθFlux] = mA.state.F_accum[boundaryA] ./ csolver.dt


    @info("postatmos",
    time = time = csolver.t + csolver.dt,
    total_θ_atmos = weightedsum(mA.state,1),
    total_θ_ocean = weightedsum(mO.state,1),
    total_F_accum = mean(mA.state.F_accum[boundaryA]) * 1e6*1e6,
    total_θ = weightedsum(mA.state,1) + weightedsum(mO.state,1) + mean(mA.state.F_accum[boundaryA]) * 1e6*1e6,
    F_accum_max = maximum(mA.state.F_accum[boundaryA]),
    F_avg_max = maximum(mA.state.F_accum[boundaryA] ./ csolver.dt),
    atmos_θ_surface_max = maximum(mA.state.θ[boundaryA]),
    ocean_θ_surface_max = maximum(mO.state.θ[boundaryO]),
  )

end


function preocean(csolver)

  # Set mean air-sea theta flux
  mO.discretization.state_auxiliary.F_prescribed[boundaryO] .= cState.CplStateBlob[:Atmos_MeanAirSeaθFlux]
  # Set ocean boundary flux accumulator to 0. (this isn't used)
  mO.state.F_accum.=0

  @info("preocean",
  time = csolver.t,
  F_prescribed_max = maximum(mO.discretization.state_auxiliary.F_prescribed[boundaryO]),
  F_prescribed_min = maximum(mO.discretization.state_auxiliary.F_prescribed[boundaryO]),
  ocean_θ_surface_max = maximum(mO.state.θ[boundaryO]),
  ocean_θ_surface_min = maximum(mO.state.θ[boundaryO]),
)

end
function postocean(csolver)
  @info("postocean",
  time = csolver.t + csolver.dt,
  ocean_θ_surface_max = maximum(mO.state.θ[boundaryO]),
  ocean_θ_surface_min = maximum(mO.state.θ[boundaryO]),
)

    # Pass ocean exports to "coupler" namespace
    #  1. Ocean SST (value of θ at z=0)
    cState.CplStateBlob[:Ocean_SST]=deepcopy( mO.state.θ[boundaryO] )
end


# Instantiate a coupled timestepper that steps forward the components and
# implements mapings between components export bondary states and
# other components imports.

compA=(pre_step=preatmos,component_model=mA,post_step=postatmos)
compO=(pre_step=preocean,component_model=mO,post_step=postocean)
component_list=( atmosphere=compA,ocean=compO,)
cC=Coupling.CplSolver(component_list=component_list,
                      coupling_dt=couple_dt,t0=0.)

# If this is run from t=0 we also need to initialize the imports so they can be read
# (for restart we need to add logic to JLD2 save/restore cState.CplStateBlob ).
cState.CplStateBlob[:Ocean_SST]=deepcopy( mO.state.θ[boundaryO] )
cState.CplStateBlob[:Atmos_MeanAirSeaθFlux]=deepcopy(mA.state.F_accum[boundaryA] )

# Invoke solve! with coupled timestepper and callback list.
solve!(nothing,cC;numberofsteps=4)
