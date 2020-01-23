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
using Dierckx
using DelimitedFiles
using Logging, Printf, Dates
using CLIMA.VTK

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

using CLIMA.Atmos
using CLIMA.Atmos: internal_energy, get_phase_partition, thermo_state
import CLIMA.Atmos: MoistureModel, temperature, pressure, soundspeed, update_aux!

# function pressure(m::MMSDryModel, state::Vars, aux::Vars)
#   T = eltype(state)
#   γ = T(7)/T(5)
#   ρinv = 1 / state.ρ
#   return (γ-1)*(state.ρe - ρinv/2 * sum(abs2, state.ρu))

# end

# function soundspeed(m::MMSDryModel, state::Vars, aux::Vars)
#   T = eltype(state)
#   γ = T(7)/T(5)
#   ρinv = 1 / state.ρ
#   p = pressure(m, state, aux)
#   sqrt(ρinv * γ * p)
# end

# TODO: Get this from geometry
const (xmin, xmax) = (-30000,30000)
const (ymin, ymax) = (0,  5000)
const (zmin, zmax) = (0, 24000)

function init_state!(state::Vars, aux::Vars, (x1,x2,x3), args...)
  spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit = args
  FT         = eltype(state)

  x = x1
  y = x2
  z = x3
  xvert          = z
  datat          = FT(spl_tinit(xvert))
  dataq          = FT(spl_qinit(xvert))
  datau          = FT(spl_uinit(xvert))
  datav          = FT(spl_vinit(xvert))
  datap          = FT(spl_pinit(xvert))
  dataq          = dataq / 1000

  if xvert >= 14000
      dataq = 0.0
  end

  θ_c =     3.0
  rx  = 10000.0
  ry  =  1500.0
  rz  =  1500.0
  xc  = 0.5*(xmax + xmin)
  yc  = 0.5*(ymax + ymin)
  zc  = 2000.0

  cylinder_flg = 0.0
  r   = sqrt( (x - xc)^2/rx^2 + cylinder_flg*(y - yc)^2/ry^2 + (z - zc)^2/rz^2)
  Δθ  = 0.0
  if r <= 1.0
      Δθ = θ_c * (cospi(0.5*r))^2
  end
  θ_liq = datat + Δθ
  q_tot = dataq
  p     = datap
  T     = air_temperature_from_liquid_ice_pottemp(θ_liq, p, PhasePartition(q_tot))
  ρ     = air_density(T, p)

  # energy definitions
  u, v, w     = datau, datav, zero(FT) #geostrophic. TO BE BUILT PROPERLY if Coriolis is considered
  ρu          = ρ * u
  ρv          = ρ * v
  ρw          = ρ * w
  e_kin       = (u^2 + v^2 + w^2) / 2
  e_pot       = grav * xvert
  ρe_tot      = ρ * total_energy(e_kin, e_pot, T, PhasePartition(q_tot))
  ρq_tot      = ρ * q_tot

  state.ρ = ρ
  state.ρu = SVector(ρu, ρv, ρw)
  state.ρe = ρe_tot
  state.moisture.ρq_tot = ρq_tot
  state.moisture.ρq_liq = FT(0)
  state.moisture.ρq_ice = FT(0)
  state.precipitation.ρq_rain = FT(0)
  nothing
end

function read_sounding()
    #read in the original squal sounding
    fsounding  = open(joinpath(@__DIR__, "../soundings/sounding_gabersek.dat"))
    #fsounding  = open(joinpath(@__DIR__, "../soundings/sounding_gabersek_3deg_warmer.dat"))
    sounding = readdlm(fsounding)
    close(fsounding)
    (nzmax, ncols) = size(sounding)
    if nzmax == 0
        error("SOUNDING ERROR: The Sounding file is empty!")
    end
    return (sounding, nzmax, ncols)
end

function spline_int()

  # ----------------------------------------------------
  # GET DATA FROM INTERPOLATED ARRAY ONTO VECTORS
  # This driver accepts data in 6 column format
  # ----------------------------------------------------
  (sounding, _, ncols) = read_sounding()

  # WARNING: Not all sounding data is formatted/scaled
  # the same. Care required in assigning array values
  # height theta qv    u     v     pressure
  zinit, tinit, qinit, uinit, vinit, pinit  =
      sounding[:, 1], sounding[:, 2], sounding[:, 3], sounding[:, 4], sounding[:, 5], sounding[:, 6]
  #------------------------------------------------------
  # GET SPLINE FUNCTION
  #------------------------------------------------------
  spl_tinit    = Spline1D(zinit, tinit; k=1)
  spl_qinit    = Spline1D(zinit, qinit; k=1)
  spl_uinit    = Spline1D(zinit, uinit; k=1)
  spl_vinit    = Spline1D(zinit, vinit; k=1)
  spl_pinit    = Spline1D(zinit, pinit; k=1)
  return spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit
end

source!(m, source, state, aux, t) = nothing

using CLIMA.Atmos: vars_state, vars_aux

function run(mpicomm, ArrayType, dim, topl, N, timeend, FT, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N
                                         )

  model = AtmosModel(FlatOrientation(),
                     ConstantViscosityWithDivergence(FT(1/100)),
                     NonEquilMoist(),
                     Rain(),
                     NoRadiation(),
                     source!,
                     NoFluxBC(),
                     init_state!)

  dg = DGModel(model,
               grid,
               Rusanov(),
               DefaultGradNumericalFlux())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, FT(0), spline_int())

  mkpath("vtk")
  mkpath(joinpath("vtk","squal_line"))
  outprefix = "vtk/squal_line/state_init_$(dim)D_mpirank$(MPI.Comm_rank(mpicomm))_step$(FT(0))"
  writevtk(outprefix, Q, dg, flattenednames(vars_state(model, FT)))

  outprefix = "vtk/squal_line/aux_init_$(dim)D_mpirank$(MPI.Comm_rank(mpicomm))_step$(FT(0))"
  writevtk(outprefix, param[1], dg, flattenednames(vars_aux(model, FT)))


  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXSimulationSteps(1) do (s=false)
    if s
      starttime[] = now()
    else
      outprefix = "vtk/squal_line/state_$(dim)D_mpirank$(MPI.Comm_rank(mpicomm))_step$(FT(ODESolvers.gettime(lsrk)))"
      writevtk(outprefix, Q, dg, flattenednames(vars_state(model, FT)))

      energy = norm(Q)
      @info @sprintf("""Update, simtime = %.16e, runtime = %s,  norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
      nothing


    end
  end

  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo, ))

  outprefix = "vtk/squal_line/state_init_$(dim)D_mpirank$(MPI.Comm_rank(mpicomm))_step$(FT(0))"
  writevtk(outprefix, Q, dg, flattenednames(vars_state(model, FT)))

  # Print some end of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, param, FT(timeend), spline_int())

  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  return [engf,engf/eng0,engf-eng0,errf,errf/engfe]
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

  polynomialorder = 4
  base_num_elem = 4

  expected_result = [1.5606226382564500e-01 5.3302790086802504e-03 2.2574728860707139e-04;
                     2.5803100360042141e-02 1.1794776908545315e-03 6.1785354745749247e-05]

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
  n_runs = 2
    for FT in (Float64,) #Float32)
      results = []
      for nth_run in 1:n_runs
        dim = 3
        l = 2
        Ne = (2^(l-1) * base_num_elem, 2^(l-1) * base_num_elem)
        brickrange = (range(FT(xmin); length=Ne[1]+1, stop=FT(xmax)),
                      range(FT(ymin); length=Ne[2]+1, stop=FT(ymax)),
                      range(FT(zmin); length=Ne[2]+1, stop=FT(zmax)))
        topl = BrickTopology(mpicomm, brickrange,
                             periodicity = (false, false, false))
        dt = 5e-3 / Ne[1]
        timeend = dt*2
        nsteps = ceil(Int64, timeend / dt)
        dt = timeend / nsteps

        result_n = run(mpicomm, ArrayType, dim, topl, polynomialorder, timeend, FT, dt)
        push!(results, result_n)
      end
      results_diff = abs.(first(diff(results, dims=1)))
      println("------------------------------------------------------------ results")
      @show results
      println("------------------------------------------------------------ results diff")
      @show results_diff
      println("------------------------------------------------------------")
      @test all([x<eps(FT) for x in results_diff])
    end
  end
end


#nothing
