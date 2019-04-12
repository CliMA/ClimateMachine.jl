using MPI

using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.AtmosDycore.VanillaAtmosDiscretizations
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.GenericCallbacks
using CLIMA.AtmosDycore
using CLIMA.MoistThermodynamics
using LinearAlgebra
using Logging, Printf, Dates

using CLIMA.ParametersType
using CLIMA.PlanetParameters: R_d, cp_d, grav, cv_d
@parameter γ_d cp_d/cv_d "Heat capacity ratio of dry air"
@parameter gdm1 R_d/cv_d "(equivalent to gamma_d-1)"

const halfperiod = 5

function isentropic_vortex(t, x...; ntrace=0,nmoist=0,dim=3)
  # Standard isentropic vortex test case.
  # For a more complete description of
  # the setup see for Example 3 of:
  #
  # @article{ZHOU2003159,
  #   author = {Y.C. Zhou and G.W. Wei},
  #   title = {High resolution conjugate filters for the simulation of flows},
  #   journal = {Journal of Computational Physics},
  #   volume = {189},
  #   number = {1},
  #   pages = {159--179},
  #   year = {2003},
  #   doi = {10.1016/S0021-9991(03)00206-7},
  #   url = {https://doi.org/10.1016/S0021-9991(03)00206-7},
  # }

  DFloat        = eltype(x)

  γ::DFloat     = γ_d
  uinf::DFloat  = 1
  vinf::DFloat  = 1
  Tinf::DFloat  = 1
  λ::DFloat     = 5 # Vortex strength
  η::DFloat     = 1 # Solution gradient parameter

  xs = x[1] - uinf*t
  ys = x[2] - vinf*t

  # make the function periodic
  xtn = floor((xs+halfperiod)/(2halfperiod))
  ytn = floor((ys+halfperiod)/(2halfperiod))
  xp = xs - xtn*2*halfperiod
  yp = ys - ytn*2*halfperiod

  rsq = xp^2 + yp^2

  u = uinf - λ*(1//2)*exp(1-rsq)*yp/π
  v = vinf + λ*(1//2)*exp(1-rsq)*xp/π
  w = zero(DFloat)
  ρ = (Tinf - ((γ-1)*λ^2*exp(2*(1-rsq))/(γ*16*π*π)))^(1/(γ-1))

  p = ρ^γ
  U = ρ*u
  V = ρ*v
  W = ρ*w
  E = p/(γ-1) + ρ * internal_energy(0) +  (1//2)*ρ*(u^2 + v^2 + w^2)
  # TODO generalise non-dimensionalisation to the PlanetParameters file for moist cases
  (ρ=ρ, U=U, V=V, W=W, E=E)
end

function main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N,
              timeend, bricktopo; dt=nothing,
              exact_timeend=true, timeinitial=0)
  dim = length(brickrange)
  topl = bricktopo(# MPI communicator to connect elements/partition
                   mpicomm,
                   # tuple of point element edges in each dimension
                   # (dim is inferred from this)
                   brickrange,
                   periodicity=ntuple(j->true, dim))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          # Compute floating point type
                                          FloatType = DFloat,
                                          # This is the array type to store
                                          # data: CuArray = GPU, Array = CPU
                                          DeviceArray = ArrayType,
                                          # polynomial order for LGL grid
                                          polynomialorder = N,
                                          # how to skew the mesh degrees of
                                          # freedom (for instance spherical
                                          # or topography maps)
                                          # warp = warpgridfun
                                         )

  # spacedisc = data needed for evaluating the right-hand side function
  spacedisc = VanillaAtmosDiscretization(grid; gravity=false,
                                        ntrace=ntrace,
                                        nmoist=nmoist)

  # This is a actual state/function that lives on the grid
  initialcondition(x...) = isentropic_vortex(DFloat(timeinitial), x...;
                                            ntrace=ntrace,
                                            nmoist=nmoist,
                                            dim=dim)
  Q = MPIStateArray(spacedisc, initialcondition)

  # Determine the time step
  (dt == nothing) && (dt = VanillaAtmosDiscretizations.estimatedt(spacedisc, Q))
  if exact_timeend
    nsteps = ceil(Int64, timeend / dt)
    dt = timeend / nsteps
  end

  # Initialize the Method (extra needed buffers created here)
  # Could also add an init here for instance if the ODE solver has some
  # state and reading from a restart file

  # TODO: Should we use get property to get the rhs function?
  lsrk = LowStorageRungeKutta(getrhsfunction(spacedisc), Q; dt = dt, t0 = 0)

  # Get the initial energy
  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      @info @sprintf """Update
  simtime = %.16e
  runtime = %s
  norm(Q) = %.16e""" ODESolvers.gettime(lsrk) Dates.format(convert(Dates.DateTime, Dates.now()-starttime[]), Dates.dateformat"HH:MM:SS") norm(Q)
    end
  end

  #= Paraview calculators:
  P = (0.4) * (E  - (U^2 + V^2 + W^2) / (2*ρ) - 9.81 * ρ * coordsZ)
  theta = (100000/287.0024093890231) * (P / 100000)^(1/1.4) / ρ
  =#
  step = [0]
  mkpath("vtk")
  cbvtk = GenericCallbacks.EveryXSimulationSteps(100) do (init=false)
    outprefix = @sprintf("vtk/IS_%dD_rank_%d_of_%d_step%04d", dim,
                         MPI.Comm_rank(mpicomm)+1, MPI.Comm_size(mpicomm),
                         step[1])
    @debug "doing VTK output" outprefix
    step[1] == 0 &&
      VanillaAtmosDiscretizations.writevtk(outprefix, Q, spacedisc)
    step[1] += 1
    nothing
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

  # Print some end of the simulation information
  engf = norm(Q)
  @info @sprintf """Finished
  norm(Q)            = %.16e
  norm(Q) / norm(Q₀) = %.16e
  norm(Q) - norm(Q₀) = %.16e""" engf engf/eng0 engf-eng0

  # TODO: Add error check!
  engf
end

using Test
let
  MPI.Initialized() || MPI.Init()

  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())
  mpicomm = MPI.COMM_WORLD

  if MPI.Comm_rank(mpicomm) == 0
    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
               ll == "WARN"  ? Logging.Warn  :
               ll == "ERROR" ? Logging.Error : Logging.Info
    global_logger(ConsoleLogger(stderr, loglevel))
  else
    global_logger(NullLogger())
  end

  Ne = (10, 10, 1)
  N = 4
  timeend = 0.1
  nmoist = 0
  ntrace = 0
  expected_energy = Dict()
  expected_energy[Float64, Array, BrickTopology, 2] =        1.9409969604595271e+06
  expected_energy[Float64, Array, BrickTopology, 3] =        6.1379713265154157e+06
  expected_energy[Float64, Array, StackedBrickTopology, 2] = 1.9409969604595273e+06
  expected_energy[Float64, Array, StackedBrickTopology, 3] = 6.1379713265154026e+06

  for DFloat in (Float64,)
    for ArrayType in (Array,)
      for bricktopo in (BrickTopology, StackedBrickTopology)
        for dim in 2:3
          brickrange = ntuple(j->range(DFloat(-halfperiod); length=Ne[j]+1,
                                       stop=halfperiod), dim)
          @test expected_energy[DFloat, ArrayType, bricktopo, dim] ≈
          main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N,
               DFloat(timeend), bricktopo, dt = 1e-3)
        end
      end
    end
  end

end

isinteractive() || MPI.Finalize()
