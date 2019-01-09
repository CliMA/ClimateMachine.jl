# TODO:
# - Switch to logging
# - Add vtk
# - timestep calculation clean
# - Move stuff to device (to kill transfers back from GPU)
# - Check that Float32 is really being used in all the kernels properly
# - Move kernels into vanilla_euler.jl?
using CLIMAAtmosDycore
using Canary
using MPI
using PlanetParameters: R_d, cp_d, grav
using ParametersType
@parameter cv_d cp_d-R_d "Isochoric specific heat dry air"
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


#Initial Conditions
function initialcondition(dim, x...)
  # FIXME: Type generic?
  DFloat = eltype(x)
  γ::DFloat       = gamma_d
  p0::DFloat      = 100000
  R_gas::DFloat   = R_d
  c_p::DFloat     = cp_d
  c_v::DFloat     = cv_d
  gravity::DFloat = grav

  u0 = 0
  r = sqrt((x[1]-500)^2 + (x[dim]-350)^2 )
  rc = 250.0
  θ_ref=300.0
  θ_c=0.5
  Δθ=0.0
  if r <= rc
    Δθ = 0.5 * θ_c * (1.0 + cos(π * r/rc))
  end
  θ_k=θ_ref + Δθ
  π_k=1.0 - gravity/(c_p*θ_k)*x[dim]
  c=c_v/R_gas
  ρ_k=p0/(R_gas*θ_k)*(π_k)^c
  ρ = ρ_k
  u = u0
  v = 0
  w = 0
  U = ρ*u
  V = ρ*v
  W = ρ*w
  Θ = ρ*θ_k
  P = p0 * (R_gas * Θ / p0)^(c_p / c_v)
  T = P/(ρ*R_gas)
  E = ρ*(c_v*T + (u^2 + v^2 + w^2)/2 + gravity*x[dim])
  ρ, U, V, W, E
end


# {{{ main
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
        @show (DFloat, dim, backend)

        meshgenerator(part, numparts) =
          brickmesh(ntuple(j->range(DFloat(0); length=Ne+1, stop=1000), dim),
                    (true, ntuple(j->false, dim-1)...),
                    part=part, numparts=numparts)
        meshwarp(x...) = identity(x)

        parameters = CLIMAAtmosDycore.Parameters(DFloat, backend, dim,
                                                 meshgenerator, meshwarp, N,
                                                 :vanilla, true)

        # generate all the static data for the dycore from the input parameters
        configuration = CLIMAAtmosDycore.Configuration(parameters, mpicomm)

        # generate all the state variables
        state = CLIMAAtmosDycore.State(parameters, configuration, timeinitial)

        nelem = size(configuration.mesh.elemtoelem, 2)
        vgeo = backend == Array ? configuration.vgeo :
                Array(configuration.vgeo)
        Q = backend == Array ? state.Q : Array(state.Q)
        _x, _y, _z = CLIMAAtmosDycore._x, CLIMAAtmosDycore._y, CLIMAAtmosDycore._z
        @inbounds for e = 1:nelem, i = 1:(N+1)^dim
          x, y, z = vgeo[i, _x, e], vgeo[i, _y, e], vgeo[i, _z, e]
          ρ, U, V, W, E = initialcondition(dim, x, y, z)
          Q[i, CLIMAAtmosDycore._ρ, e] = ρ
          Q[i, CLIMAAtmosDycore._U, e] = U
          Q[i, CLIMAAtmosDycore._V, e] = V
          Q[i, CLIMAAtmosDycore._W, e] = W
          Q[i, CLIMAAtmosDycore._E, e] = E
        end
        if backend != Array
          state.Q .= Q
        end

        # Set up the time stepper
        # FIXME: Set timestep in another way
        mpirank == 0 && println("computing dt (CPU)...")
        base_dt = CLIMAAtmosDycore.cfl(Val(dim), Val(N), vgeo, Q, mpicomm)/N^√2
        mpirank == 0 && @show base_dt
        nsteps = ceil(Int64, timeend / base_dt)
        dt = timeend / nsteps
        mpirank == 0 && @show (dt, nsteps, dt * nsteps, timeend)
        timestepstate = CLIMAAtmosDycore.LSRKState(dt, state)

        timestepinfo = CLIMAAtmosDycore.timestepinfocallback(timestepstate, 1)

        stats = zeros(DFloat, 2)
        mpirank == 0 && println("computing initial energy...")
        stats[1] = CLIMAAtmosDycore.L2energysquared(Val(dim), Val(N),
                                                    Q, vgeo,
                                                    configuration.mesh.realelems)

        CLIMAAtmosDycore.run!(state, timestepstate, parameters, configuration,
                              timeend, (timestepinfo,))

        mpirank == 0 && println("computing final energy...")
        if backend != Array
          Q .= state.Q
        end
        stats[2] = CLIMAAtmosDycore.L2energysquared(Val(dim), Val(N),
                                                    Q, vgeo,
                                                    configuration.mesh.realelems)
        stats = sqrt.(MPI.allreduce(stats, MPI.SUM, mpicomm))
        if  mpirank == 0
          @show eng0 = stats[1]
          @show engf = stats[2]
          @show Δeng = engf - eng0
        end
      end
    end
  end
  nothing
end
# }}}

main()
