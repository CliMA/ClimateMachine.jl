# TODO:
# - Switch to logging
# - Add vtk
# - timestep calculation clean
# - Move stuff to device (to kill transfers back from GPU)
# - Check that Float32 is really being used in all the kernels properly

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
brickmesh(ntuple(j->range(DFloat(0); length=Ne+1, stop=1000), dim),
          (true, ntuple(j->false, dim-1)...),
          part=part, numparts=numparts)

function main()
  MPI.Initialized() || MPI.Init()
  MPI.finalize_atexit()

  mpicomm = MPI.COMM_WORLD
  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # FIXME: query via hostname
  @hascuda device!(mpirank % length(devices()))

  timeinitial = 0.0
  timeend = 0.1
  Ne = 10
  N  = 4

  for DFloat in (Float64, Float32)
    for dim in (2,3)
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
                            gravity = true,
                            N = N,
                           ),
                           # Time Discretization and Parameters
                           :LSRK,
                           (),
                          )

        # Set the initial condition with a function
        AD.initspacestate!(runner, host=true) do (x...)
          DFloat = eltype(x)
          γ::DFloat       = gamma_d
          p0::DFloat      = 100000
          R_gas::DFloat   = R_d
          c_p::DFloat     = cp_d
          c_v::DFloat     = cv_d
          gravity::DFloat = grav

          r = sqrt((x[1] - 500)^2 + (x[dim] - 350)^2)
          rc::DFloat = 250
          θ_ref::DFloat = 300
          θ_c::DFloat = 0.5
          Δθ::DFloat = 0
          if r <= rc
            Δθ = θ_c * (1 + cos(π * r / rc)) / 2
          end
          θ_k = θ_ref + Δθ
          π_k = 1 - gravity / (c_p * θ_k) * x[dim]
          c = c_v / R_gas
          ρ_k = p0 / (R_gas * θ_k) * (π_k)^c
          ρ = ρ_k
          u = zero(DFloat)
          v = zero(DFloat)
          w = zero(DFloat)
          U = ρ * u
          V = ρ * v
          W = ρ * w
          Θ = ρ * θ_k
          P = p0 * (R_gas * Θ / p0)^(c_p / c_v)
          T = P / (ρ * R_gas)
          E = ρ * (c_v * T + (u^2 + v^2 + w^2) / 2 + gravity * x[dim])
          ρ, U, V, W, E
        end

        # Compute a (bad guess) for the time step
        base_dt = AD.estimatedt(runner, host=true)
        nsteps = ceil(Int64, timeend / base_dt)
        dt = timeend / nsteps

        # Set the time step
        AD.inittimestate!(runner, dt)

        eng0 = AD.L2solutionnorm(runner; host=true)
        # mpirank == 0 && @show eng0

        # Setup the info callback
        io = mpirank == 0 ? stdout : open("/dev/null", "w")
        show(io, "text/plain", runner[:spacerunner])
        cbinfo =
          AD.GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do
            println(io, runner[:spacerunner])
          end

        # Setup the vtk callback
        mkpath("viz")
        dump_vtk(step) = AD.writevtk(runner,
                                     "viz/RTB"*
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
