using MPI
using CLIMA
using Logging
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using LinearAlgebra
using Printf
using Dates
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.DGmethods: EveryDirection, HorizontalDirection, VerticalDirection

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

if !@isdefined integration_testing
  if length(ARGS) > 0
    const integration_testing = parse(Bool, ARGS[1])
  else
    const integration_testing =
      parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
  end
end

const output = parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_OUTPUT","false")))

include("advection_diffusion_model.jl")

struct Pseudo1D{n, α, β, μ, δ} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(::Pseudo1D{n, α, β}, aux::Vars,
                                  geom::LocalGeometry) where {n, α, β}
  # Direction of flow is n with magnitude α
  aux.u = α * n

  # diffusion of strength β in the n direction
  aux.D = β * n * n'
end

function initial_condition!(::Pseudo1D{n, α, β, μ, δ}, state, aux, x,
                            t) where {n, α, β, μ, δ}
  ξn = dot(n, x)
  # ξT = SVector(x) - ξn * n
  state.ρ = exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model, testname)
  ## name of the file that this MPI rank will write
  filename = @sprintf("%s/%s_mpirank%04d_step%04d",
                      vtkdir, testname, MPI.Comm_rank(mpicomm), vtkstep)

  statenames = flattenednames(vars_state(model, eltype(Q)))
  exactnames = statenames .* "_exact"

  writevtk(filename, Q, dg, statenames, Qe, exactnames)

  ## Generate the pvtu file for these vtk files
  if MPI.Comm_rank(mpicomm) == 0
    ## name of the pvtu file
    pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

    ## name of each of the ranks vtk files
    prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
      @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
    end

    writepvtu(pvtuprefix, prefixes, (statenames..., exactnames...))

    @info "Done writing VTK: $pvtuprefix"
  end
end


function run(mpicomm, ArrayType, dim, topl, N, timeend, FT, direction, dt,
             n, α, β, μ, δ, vtkdir, outputtime)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  model = AdvectionDiffusion{dim}(Pseudo1D{n, α, β, μ, δ}())
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty(),
               direction=direction())

  Q = init_ode_state(dg, FT(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end
  callbacks = (cbinfo,)
  if ~isnothing(vtkdir)
    # create vtk dir
    mkpath(vtkdir)

    vtkstep = 0
    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model, "advection_diffusion")

    # setup the output callback
    cbvtk = EveryXSimulationSteps(floor(outputtime/dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, gettime(lsrk))
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model,
                "advection_diffusion")
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=callbacks)

  # Print some end of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, FT(timeend))

  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
  errf
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

  expected_result = Dict()
  expected_result[2, 1, Float64, EveryDirection] = 1.2228434091602128e-02
  expected_result[2, 2, Float64, EveryDirection] = 8.8037798002260420e-04
  expected_result[2, 3, Float64, EveryDirection] = 4.8828676920661276e-05
  expected_result[2, 4, Float64, EveryDirection] = 2.0105646643725454e-06
  expected_result[3, 1, Float64, EveryDirection] = 9.5425450102548364e-03
  expected_result[3, 2, Float64, EveryDirection] = 5.9769045240778518e-04
  expected_result[3, 3, Float64, EveryDirection] = 4.0081798525590592e-05
  expected_result[3, 4, Float64, EveryDirection] = 2.9803558844543670e-06
  expected_result[2, 1, Float32, EveryDirection] = 1.2228445149958134e-02
  expected_result[2, 2, Float32, EveryDirection] = 8.8042858988046646e-04
  expected_result[2, 3, Float32, EveryDirection] = 4.8848683945834637e-05
  expected_result[2, 4, Float32, EveryDirection] = 2.1814605588588165e-06
  expected_result[3, 1, Float32, EveryDirection] = 9.5424978062510490e-03
  expected_result[3, 2, Float32, EveryDirection] = 5.9770536608994007e-04
  expected_result[3, 3, Float32, EveryDirection] = 4.0205955883720890e-05
  expected_result[3, 4, Float32, EveryDirection] = 5.1562137741711922e-06

  expected_result[2, 1, Float64, HorizontalDirection] = 4.6773313437233156e-02
  expected_result[2, 2, Float64, HorizontalDirection] = 4.0665907382118234e-03
  expected_result[2, 3, Float64, HorizontalDirection] = 5.3141853450218684e-05
  expected_result[2, 4, Float64, HorizontalDirection] = 3.9767488072903428e-07
  expected_result[3, 1, Float64, HorizontalDirection] = 1.7293617338929253e-02
  expected_result[3, 2, Float64, HorizontalDirection] = 1.2450424793625284e-03
  expected_result[3, 3, Float64, HorizontalDirection] = 6.9054177134198605e-05
  expected_result[3, 4, Float64, HorizontalDirection] = 2.8433678163945639e-06
  expected_result[2, 1, Float32, HorizontalDirection] = 4.6773292124271393e-02
  expected_result[2, 2, Float32, HorizontalDirection] = 4.0663531981408596e-03
  expected_result[2, 3, Float32, HorizontalDirection] = 5.3235678933560848e-05
  expected_result[2, 4, Float32, HorizontalDirection] = 3.5185137221560581e-06
  expected_result[3, 1, Float32, HorizontalDirection] = 1.7293667420744896e-02
  expected_result[3, 2, Float32, HorizontalDirection] = 1.2450854992493987e-03
  expected_result[3, 3, Float32, HorizontalDirection] = 7.0057700213510543e-05
  expected_result[3, 4, Float32, HorizontalDirection] = 5.5491593229817227e-05

  expected_result[2, 1, Float64, VerticalDirection] = 4.6773313437233136e-02
  expected_result[2, 2, Float64, VerticalDirection] = 4.0665907382118321e-03
  expected_result[2, 3, Float64, VerticalDirection] = 5.3141853450244739e-05
  expected_result[2, 4, Float64, VerticalDirection] = 3.9767488073407439e-07
  expected_result[3, 1, Float64, VerticalDirection] = 6.6147454220062032e-02
  expected_result[3, 2, Float64, VerticalDirection] = 5.7510277746000895e-03
  expected_result[3, 3, Float64, VerticalDirection] = 7.5153929878994988e-05
  expected_result[3, 4, Float64, VerticalDirection] = 5.6239720970531199e-07
  expected_result[2, 1, Float32, VerticalDirection] = 4.6773284673690796e-02
  expected_result[2, 2, Float32, VerticalDirection] = 4.0663606487214565e-03
  expected_result[2, 3, Float32, VerticalDirection] = 5.3235460654832423e-05
  expected_result[2, 4, Float32, VerticalDirection] = 3.8958251025178470e-06
  expected_result[3, 1, Float32, VerticalDirection] = 6.6147454082965851e-02
  expected_result[3, 2, Float32, VerticalDirection] = 5.7507432065904140e-03
  expected_result[3, 3, Float32, VerticalDirection] = 8.3523096691351384e-05
  expected_result[3, 4, Float32, VerticalDirection] = 2.3923123080749065e-04

  numlevels = integration_testing ? 4 : 1

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for FT in (Float64, Float32)
      result = zeros(FT, numlevels)
      for dim = 2:3
        for direction in (EveryDirection, HorizontalDirection,
                          VerticalDirection)
          if direction <: EveryDirection
            n = dim == 2 ? SVector{3, FT}(1/sqrt(2), 1/sqrt(2), 0) :
                           SVector{3, FT}(1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
          elseif direction <: HorizontalDirection
            n = dim == 2 ? SVector{3, FT}(1, 0, 0) :
                           SVector{3, FT}(1/sqrt(2), 1/sqrt(2), 0)
          elseif direction <: VerticalDirection
            n = dim == 2 ? SVector{3, FT}(0, 1, 0) : SVector{3, FT}(0, 0, 1)
          end
          α = FT(1)
          β = FT(1 // 100)
          μ = FT(-1 // 2)
          δ = FT(1 // 10)
          for l = 1:numlevels
            Ne = 2^(l-1) * base_num_elem
            brickrange = ntuple(j->range(FT(-1); length=Ne+1, stop=1), dim)
            periodicity = ntuple(j->false, dim)
            topl = StackedBrickTopology(mpicomm, brickrange;
                                        periodicity = periodicity)
            dt = (α/4) / (Ne * polynomialorder^2)
            @info "time step" dt

            timeend = 1
            outputtime = 1

            dt = outputtime / ceil(Int64, outputtime / dt)

            @info (ArrayType, FT, dim, direction)
            vtkdir = output ? "vtk_advection" *
                              "_poly$(polynomialorder)" *
                              "_dim$(dim)_$(ArrayType)_$(FT)_$(direction)" *
                              "_level$(l)" : nothing
            result[l] = run(mpicomm, ArrayType, dim, topl, polynomialorder,
                            timeend, FT, direction, dt, n, α, β, μ, δ, vtkdir,
                            outputtime)
            @test result[l] ≈ FT(expected_result[dim, l, FT, direction])
          end
          @info begin
            msg = ""
            for l = 1:numlevels-1
              rate = log2(result[l]) - log2(result[l+1])
              msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
            end
            msg
          end
        end
      end
    end
  end
end

nothing

