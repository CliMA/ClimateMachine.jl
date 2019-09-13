
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK

using CLIMA.Atmos: vars_state, vars_aux

using Random 
const seed = MersenneTwister(0)

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray,) 
else
  const ArrayTypes = (Array,)
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

"""
  Initial Condition for ChannelFlow_RF01 LES
"""
function Initialise_ChannelFlow!(state::Vars, aux::Vars, (x,y,z), t)
  DT         = eltype(state)
  state.ρ    = DT(1.22)
  u          = sin(1000*x*y*z)/100
  v          = sin(1000*x*y*z)/1000
  w          = DT(0)
  state.ρu   = SVector(u,v,w)
  state.ρe   = 1/2 * state.ρ * (u^2 + v^2 + w^2)
  state.moisture.ρq_tot = DT(0)
end   


function run(mpicomm, ArrayType, dim, topl, N, timeend, DT, dt, C_smag)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  model = AtmosModel(NoOrientation(),
                     NoReferenceState(),
                     ConstantViscosityWithDivergence{DT}(1.22e-4),
                     EquilMoist(),
                     NoRadiation(),
                     (Gravity(),ConstPG{DT}(-(1.22*180^2*(1e-4)^2))),
                     ChannelFlowBC(),
                     Initialise_ChannelFlow!)

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DT(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  step = [0]
    cbvtk = GenericCallbacks.EveryXSimulationSteps(20000) do (init=false)
    vtkdir = "./vtk-channel"
    mkpath(vtkdir)
    outprefix = @sprintf("%s/channel_%dD_mpirank%04d_step%04d",vtkdir, dim,
                           MPI.Comm_rank(mpicomm), step[1])
    writevtk(outprefix, Q, dg, flattenednames(vars_state(model,DT)), 
             param[1], flattenednames(vars_aux(model,DT)))
    #=
    pvtuprefix = @sprintf("channel_%dD_step%04d", dim, step[1])
      prefixes = ntuple(i->
      @sprintf("%s/channel_%dD_mpirank%04d_step%04d", vtkdir,
               dim, i-1, step[1]),
               MPI.Comm_size(mpicomm))
      writepvtu(pvtuprefix, prefixes, statenames)
    =#
    step[1] += 1
    nothing
  end

  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo, cbvtk))

  # Print some end of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, param, DT(timeend))

  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
  engf/eng0
end

using Test
let
  MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  @static if haspkg("CUDAnative")
      device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end
  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    # Problem type
    DT = Float32
    # DG polynomial order 
    polynomialorder = 4
    # SGS Filter constants
    C_smag = DT(0.15)
    # User defines the grid size:
    Ne = (25, 25, 25)
    # Physical domain extents 
    (xmin, xmax) = (0, 4)
    (ymin, ymax) = (0, 2)
    (zmin, zmax) = (0, 2)
    # User defined domain parameters
    brickrange = (range(DT(xmin), length=Ne[1]+1, DT(xmax)),
                  range(DT(ymin), length=Ne[2]+1, DT(ymax)),
                  range(DT(zmin), length=Ne[3]+1, DT(zmax)))
    topl = StackedBrickTopology(mpicomm, brickrange,periodicity = (true, true, false), boundary=((0,0),(0,0),(1,2)))
    dt = (2/25/16/330 * 0.5)
    timeend = 3600*10
    dim = 3
    @info (ArrayType, DT, dim)
    result = run(mpicomm, ArrayType, dim, topl, 
                 polynomialorder, timeend, DT, dt, C_smag)
  end
end
