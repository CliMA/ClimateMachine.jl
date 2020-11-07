#!/usr/bin/env julia --project

function get_parm( plist, pkey, def)
  hk = haskey(plist,pkey)
  if ( hk ) 
    return plist[pkey]
  end
  return def
end

include("../../../experiments/OceanBoxGCM/simple_box.jl")
ClimateMachine.init(parse_clargs = true)

const FT = Float64

#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    include("../refvals/test_ocean_gyre_refvals.jl")

    ### # tinit(x,y,z,p) = ( 20. * ( 1 - z/p.H ) )
    ### tinit(x::FT,y::FT,z::FT,p::HomogeneousBox) = ( 20. )

    # Set a bunch of parameters 
    myparams = (

      # Equation set type, selects core equations and timestepper defaults
      etype=HydrostaticBoussinesq,

      # Setup type for selecting defaults around initialization, boundary conditions, forcing options and parameters.
      # HomogenousBox - multi-layer, constant initial T, no surface T forcing only wind, solid walls all round
      #                 noslip side and bottom.
      stype=HomogeneousBox,

      # Name and major type (used in output file prefix)
      cname="barotropic_nlayer_short",

      # Spatial discretization parameters
      # Polynomial order and element counts in x,y and z.
      N_dgp = Int(4),
      Nˣ = Int(10),
      Nʸ = Int(15),
      Nᶻ = Int(2),

      # Domain extents
      Lˣ = 4e6,   # m
      Lʸ = 6e6,   # m
      H = 3000,  # m

      # Duration
      timestart = FT(0),   # s
      timeend = FT(3600), # s

      # Timestepper
      ## solver_method = LSRK144NiegemannDiehlBusch
      solver_method = LS3NRK33Heuns,

      # Overrides some defaults
      # Boundary conditions 
      # sides, bottom, top
      # overrides "setup type"
      BC = (
        OceanBC(Impenetrable(NoSlip()), Insulating()),
        OceanBC(Impenetrable(FreeSlip()), Insulating()),
        OceanBC(Penetrable(KinematicStress()), Insulating()),
           ),

      # Set VTK non-default range for use when sampling selected.
      # - range must fall in biunit limits -1 to 1.
      my_vtk_range_lo = -0.99,
      my_vtk_range_hi =  0.99,

      # Physical parameters
      αᵀ = 0.,
      νʰ = 1e2*5,

      # Initializers
      # tinit(x,y,z,p) = ( 20. * ( 1 - z/p.H ) )
      tinit_f = tinit(x::FT,y::FT,z::FT,p::HomogeneousBox) = ( 20. ),
      # tinit_f = tinit,
    )




  
    # simulation time
    timestart = get_parm(myparams, :timestart, FT(0)    )    # s
    timeend   = get_parm(myparams, :timeend,   FT(3600) )    # s
    timespan = (timestart, timeend)

    # DG polynomial order
    N = get_parm(myparams, :N_dgp, Int(4) )

    # Domain resolution
    Nˣ = get_parm(myparams, :Nˣ, Int(5) )
    Nʸ = get_parm(myparams, :Nʸ, Int(5) )
    Nᶻ = get_parm(myparams, :Nᶻ, Int(2) )
    resolution = (N, Nˣ, Nʸ, Nᶻ)

    # Domain size
    Lˣ = 4e6    # m
    Lˣ = get_parm(myparams, :Lˣ, Lˣ )
    Lʸ = 6e6    # m
    Lʸ = get_parm(myparams, :Lʸ, Lʸ )
    H = 3000   # m
    H = get_parm(myparams, :H, H )
    dimensions = (Lˣ, Lʸ, H)

    BC = (
        OceanBC(Impenetrable(NoSlip()), Insulating()),
        OceanBC(Impenetrable(FreeSlip()), Insulating()),
        OceanBC(Penetrable(KinematicStress()), Insulating()),
    )
    BC = get_parm(myparams, :BC, BC )
    stype = get_parm(myparams,  :stype, HomogeneousBox)

    run_simple_box(
        "barotropic_nlayer_short",
        resolution,
        dimensions,
        timespan,
        stype,
        imex = false,
        BC = BC,
        Δt = 120,
        refDat = refVals.short,
        mymodeldata = myparams,
    )
end
