using CLIMAAtmosDycore
const AD = CLIMAAtmosDycore
using Canary
using MPI

using ParametersType
using PlanetParameters: R_d, cp_d, grav, cv_d
@parameter gamma_d cp_d/cv_d "Heat capcity ratio of dry air"
@parameter gdm1 R_d/cv_d "(equivalent to gamma_d-1)"

const HAVE_CUDA = try
  using CUDAdrv
  using CUDAnative
  true
catch
  false
end

macro hascuda(ex)
  return HAVE_CUDA ? :($(esc(ex))) : :(nothing)
end

meshgenerator(part, numparts, Ne, dim, DFloat) =
brickmesh(ntuple(j->range(DFloat(-5); length=Ne+1, stop=5), dim),
          ntuple(j->true, dim), part=part, numparts=numparts)

function isentropicvortex(t, x...)
  # Standard isentropic vortex test case.  For a more complete description of
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
  DFloat = eltype(x)

  γ::DFloat    = gamma_d
  uinf::DFloat = 1
  vinf::DFloat = 1
  Tinf::DFloat = 1
  xo::DFloat   = 0
  yo::DFloat   = 0
  λ::DFloat    = 5

  xs = x[1] - uinf*t - xo
  ys = x[2] - vinf*t - yo
  rsq = xs^2 + ys^2

  u = uinf - λ*(1//2)*exp(1-rsq)*ys/π
  v = vinf + λ*(1//2)*exp(1-rsq)*xs/π
  w = zero(DFloat)

  ρ = (Tinf - ((γ-1)*λ^2*exp(2*(1-rsq))/(γ*16*π*π)))^(1/(γ-1))
  p = ρ^γ
  U = ρ*u
  V = ρ*v
  W = ρ*w
  E = p/(γ-1) + (1//2)*ρ*(u^2 + v^2 + w^2)

  ρ, U, V, W, E
end

function main()
  MPI.Initialized() || MPI.Init()
  MPI.finalize_atexit()

  mpicomm = MPI.COMM_WORLD
  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # FIXME: query via hostname
  @hascuda device!(mpirank % length(devices()))

  timeinitial = 0.0
  timeend = 10.0
  Ne = 10
  N  = 4

  for DFloat in (Float64, Float32)
    for dim in (2, 3)
      for backend in (HAVE_CUDA ? (CuArray, Array) : (Array,))

        runner = AD.Runner(mpicomm,
                           #Space Discretization and Parameters
                           :VanillaEuler,
                           (DFloat = DFloat,
                            DeviceArray = backend,
                            meshgenerator = (part, numparts) ->
                            meshgenerator(part, numparts, Ne, dim,
                                          DFloat),
                            dim = dim,
                            gravity = false,
                            N = N,
                           ),
                           # Time Discretization and Parameters
                           :LSRK,
                           (),
                          )

        # Set the initial condition with a function
        AD.initspacestate!(runner, host=true) do (x...)
          isentropicvortex(DFloat(timeinitial), x...)
        end

        base_dt = 1e-3
        nsteps = ceil(Int64, timeend / base_dt)
        dt = timeend / nsteps

        # Set the time step
        AD.inittimestate!(runner, dt)

        eng0 = AD.L2solutionnorm(runner; host=true)
        # mpirank == 0 && @show eng0

        # Setup the info callback
        io = mpirank == 0 ? stdout : open("/dev/null", "w")
        show(io, "text/plain", runner[:spacerunner])
        cbinfo = AD.GenericCallbacks.EveryXWallTimeSecondsCallback(10) do
          println(io, runner[:spacerunner])
        end

        # Setup the vtk callback
        mkpath("viz")
        dump_vtk(step) = AD.writevtk(runner,
                                     "viz/IV"*
                                     "_dim_$(dim)"*
                                     "_DFloat_$(DFloat)"*
                                     "_backend_$(backend)"*
                                     "_mpirank_$(mpirank)"*
                                     "_step_$(step)")
        step = 0
        cbvtk = AD.GenericCallbacks.EveryXSimulationSteps(10) do
          # TODO: We should add queries back to time stepper for this
          step += 1
          dump_vtk(step)
          nothing
        end

        dump_vtk(0)
        AD.run!(runner; numberofsteps=nsteps, callbacks=(cbinfo, cbvtk))
        dump_vtk(nsteps)

        engf = AD.L2solutionnorm(runner; host=true)

        mpirank == 0 && @show engf
        mpirank == 0 && @show eng0 - engf
        mpirank == 0 && @show engf/eng0
        mpirank == 0 && println()
      end
    end
  end
  nothing
end

main()
