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
using CLIMA.Mesh.Grids: EveryDirection, HorizontalDirection, VerticalDirection, min_node_distance

const ArrayType = CLIMA.array_type()

if !@isdefined integration_testing
  if length(ARGS) > 0
    const integration_testing = parse(Bool, ARGS[1])
  else
    const integration_testing =
      parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
  end
end

const output = parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_OUTPUT","false")))

include("hyperdiffusion_model.jl")

struct ConstantHyperDiffusion{dim, dir, FT} <: HyperDiffusionProblem
  D::SMatrix{3, 3, FT, 9}
end

function init_hyperdiffusion_tensor!(problem::ConstantHyperDiffusion, aux::Vars,
                                     geom::LocalGeometry)
  aux.D = problem.D
end

function initial_condition!(problem::ConstantHyperDiffusion{dim, dir}, state, aux, x, t) where {dim, dir}
  @inbounds begin 
    if dir === EveryDirection()
      tD = sum(problem.D[SOneTo(dim), SOneTo(dim)])
      c = dim
    elseif dir === HorizontalDirection()
      tD = sum(problem.D[SOneTo(dim - 1), SOneTo(dim - 1)])
      c = dim - 1
    elseif dir === VerticalDirection()
      tD = problem.D[dim, dim]
      c = 1
    end
    state.ρ = sin(sum(x[SOneTo(dim)]))  * exp(-c * tD * t)
  end
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


function run(mpicomm, dim, topl, N, timeend, FT, direction,
             D, vtkdir, outputtime)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  dx = min_node_distance(grid)
  dt = dx ^ 4 / 25 / sum(D)
  @info "time step" dt
  dt = outputtime / ceil(Int64, outputtime / dt)

  model = HyperDiffusion{dim}(ConstantHyperDiffusion{dim, direction(), FT}(D))
  dg = DGModel(model,
               grid,
               CentralNumericalFluxNonDiffusive(),
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
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model, "hyperdiffusion")

    # setup the output callback
    cbvtk = EveryXSimulationSteps(floor(outputtime/dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, gettime(lsrk))
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model,
                "hyperdiffusion")
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
  CLIMA.init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  polynomialorder = 4
  base_num_elem = 4

  expected_result = Dict()
  expected_result[2, 1, Float64, EveryDirection] = 1.2811603422252390e-03
  expected_result[2, 2, Float64, EveryDirection] = 6.2595782972243863e-05
  expected_result[2, 3, Float64, EveryDirection] = 1.7123779415917798e-06
  expected_result[2, 1, Float64, HorizontalDirection] = 8.4650675812621937e-04
  expected_result[2, 2, Float64, HorizontalDirection] = 4.4626814794977747e-05
  expected_result[2, 3, Float64, HorizontalDirection] = 1.2193396280948004e-06
  expected_result[2, 1, Float64, VerticalDirection] = 7.8773419106930231e-04
  expected_result[2, 2, Float64, VerticalDirection] = 4.4504128011922683e-05
  expected_result[2, 3, Float64, VerticalDirection] = 1.2198274613245607e-06

  expected_result[3, 1, Float64, EveryDirection] = 4.6856494647676715e-03
  expected_result[3, 2, Float64, EveryDirection] = 1.8817234231940995e-04
  expected_result[3, 3, Float64, EveryDirection] = 5.1466916036801814e-06
  expected_result[3, 1, Float64, HorizontalDirection] = 3.2113927381582875e-03
  expected_result[3, 2, Float64, HorizontalDirection] = 1.5690435947374426e-04
  expected_result[3, 3, Float64, HorizontalDirection] = 4.2922949674095959e-06
  expected_result[3, 1, Float64, VerticalDirection] = 2.1979218115440783e-03
  expected_result[3, 2, Float64, VerticalDirection] = 1.1190507836283780e-04
  expected_result[3, 3, Float64, VerticalDirection] = 3.0558199635985687e-06

  numlevels = 3
  for FT in (Float64,)
    D = 1 // 100 * SMatrix{3, 3, FT}(9 // 50, 3 // 50, 5  // 50,
                                     3 // 50, 7 // 50, 4  // 50,
                                     5 // 50, 4 // 50, 10 // 50)

    result = zeros(FT, numlevels)
    for dim in (2, 3)
      for direction in (EveryDirection, HorizontalDirection, VerticalDirection)
        for l = 1:numlevels
          Ne = 2^(l-1) * base_num_elem
          xrange = range(FT(0); length=Ne+1, stop=FT(2pi))
          brickrange = ntuple(j->xrange, dim)
          periodicity = ntuple(j->true, dim)
          topl = StackedBrickTopology(mpicomm, brickrange;
                                      periodicity = periodicity)
          timeend = 1
          outputtime = 1

          @info (ArrayType, FT, dim, direction)
          vtkdir = output ? "vtk_hyperdiffusion" *
                            "_poly$(polynomialorder)" *
                            "_dim$(dim)_$(ArrayType)_$(FT)_$(direction)" *
                            "_level$(l)" : nothing
          result[l] = run(mpicomm, dim, topl, polynomialorder,
                          timeend, FT, direction, D, vtkdir,
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

nothing

