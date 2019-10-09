using CLIMA: haspkg
using CLIMA.Mesh.Topologies: StackedCubedSphereTopology, cubedshellwarp, grid1d
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid
using CLIMA.Mesh.Filters
using CLIMA.DGmethods: DGModel, init_ode_state
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralGradPenalty,
                                       CentralNumericalFluxDiffusive,
                                       CentralNumericalFluxNonDiffusive
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.LowStorageRungeKuttaMethod: LSRK144NiegemannDiehlBusch
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.MPIStateArrays: euclidean_distance
using CLIMA.PlanetParameters: R_d, grav, MSLP, planet_radius, cp_d, cv_d
using CLIMA.MoistThermodynamics: air_density, total_energy, soundspeed_air, internal_energy, air_temperature
using CLIMA.Atmos: AtmosModel, SphericalOrientation, NoReferenceState,
                   DryModel, NoRadiation, PeriodicBC, NoFluxBC,
                   ConstantViscosityWithDivergence, vars_state, Gravity, Coriolis,
                   HydrostaticState, IsothermalProfile
using CLIMA.VariableTemplates: flattenednames

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test
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

const output_vtk = false

function main()
  MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD

  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = Dict("DEBUG" => Logging.Debug,
                  "WARN"  => Logging.Warn,
                  "ERROR" => Logging.Error,
                  "INFO"  => Logging.Info)[ll]

  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  polynomialorder = 4
  numlevels = integration_testing ? 4 : 1

  expected_error = Dict()

  # just to make it shorter and aligning
  Central = CentralNumericalFluxNonDiffusive

  expected_error[Float64, 2, Rusanov, 1] = 1.1990999506538110e+01
  expected_error[Float64, 2, Rusanov, 2] = 2.0813000228865612e+00
  expected_error[Float64, 2, Rusanov, 3] = 6.3752572004789149e-02
  expected_error[Float64, 2, Rusanov, 4] = 2.0984975076420455e-03
  
  expected_error[Float64, 2, Central, 1] = 2.0840574601661153e+01
  expected_error[Float64, 2, Central, 2] = 2.9255455365299827e+00
  expected_error[Float64, 2, Central, 3] = 3.6935849488949657e-01
  expected_error[Float64, 2, Central, 4] = 8.3528804679907434e-03

  expected_error[Float64, 3, Rusanov, 1] = 3.7918869862613858e+00
  expected_error[Float64, 3, Rusanov, 2] = 6.5816485664822677e-01
  expected_error[Float64, 3, Rusanov, 3] = 2.0160333422867591e-02
  expected_error[Float64, 3, Rusanov, 4] = 6.6360317881818034e-04
  
  expected_error[Float64, 3, Central, 1] = 6.5903683487905749e+00
  expected_error[Float64, 3, Central, 2] = 9.2513872939749997e-01
  expected_error[Float64, 3, Central, 3] = 1.1680141169828175e-01
  expected_error[Float64, 3, Central, 4] = 2.6414127301659534e-03

  expected_error[Float32, 2, Rusanov, 1] = 1.1990854263305664e+01
  expected_error[Float32, 2, Rusanov, 2] = 2.0812149047851563e+00
  expected_error[Float32, 2, Rusanov, 3] = 6.7652329802513123e-02
  expected_error[Float32, 2, Rusanov, 4] = 3.6849677562713623e-02
  
  expected_error[Float32, 2, Central, 1] = 2.0840496063232422e+01
  expected_error[Float32, 2, Central, 2] = 2.9250388145446777e+00
  expected_error[Float32, 2, Central, 3] = 3.7026408314704895e-01
  expected_error[Float32, 2, Central, 4] = 6.7625500261783600e-02

  expected_error[Float32, 3, Rusanov, 1] = 3.7918324470520020e+00
  expected_error[Float32, 3, Rusanov, 2] = 6.5811443328857422e-01
  expected_error[Float32, 3, Rusanov, 3] = 2.1280560642480850e-02
  expected_error[Float32, 3, Rusanov, 4] = 9.8376255482435226e-03
  
  expected_error[Float32, 3, Central, 1] = 6.5902600288391113e+00
  expected_error[Float32, 3, Central, 2] = 9.2505264282226563e-01
  expected_error[Float32, 3, Central, 3] = 1.1701638251543045e-01
  expected_error[Float32, 3, Central, 4] = 1.2930640019476414e-02

  @testset "$(@__FILE__)" begin
    for ArrayType in ArrayTypes, FT in (Float64,), dims in (3,)
      for NumericalFlux in (Rusanov, )
        @info @sprintf """Configuration
                          ArrayType     = %s
                          FT            = %s
                          NumericalFlux = %s
                          dims          = %d
                          """ "$ArrayType" "$FT" "$NumericalFlux" dims

        setup = HeldSuarezSetup{FT}()
        errors = Vector{FT}(undef, numlevels)

        for level in 1:numlevels
          numelems = ntuple(dim -> dim == 3 ? 1 : 2 ^ (level - 1) * 5, dims)
          errors[level] =
            run(mpicomm, polynomialorder, numelems,
                NumericalFlux, setup, ArrayType, FT, dims, level)

          # rtol = sqrt(eps(FT))
          # # increase rtol for comparing with GPU results using Float32
          # if FT === Float32 && !(ArrayType === Array)
          #   rtol *= 10 # why does this factor have to be so big :(
          # end
          # @test isapprox(errors[level],
          #                expected_error[FT, dims, NumericalFlux, level]; rtol = rtol)
        end

        rates = @. log2(first(errors[1:numlevels-1]) / first(errors[2:numlevels]))
        numlevels > 1 && @info "Convergence rates\n" *
          join(["rate for levels $l → $(l + 1) = $(rates[l])" for l in 1:numlevels-1], "\n")
      end
    end
  end
end

function run(mpicomm, polynomialorder, numelems,
             NumericalFlux, setup, ArrayType, FT, dims, level)


  polynomialorder = 5
  num_elem_horz = 6
  num_elem_vert = 8

  vert_range = grid1d(FT(planet_radius), FT(planet_radius + setup.domain_height), nelem = num_elem_vert)
  topology = StackedCubedSphereTopology(mpicomm, num_elem_horz, vert_range)

  grid = DiscontinuousSpectralElementGrid(topology,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder,
                                          meshwarp = cubedshellwarp)

  model = AtmosModel(SphericalOrientation(),
                     HydrostaticState(IsothermalProfile(setup.T_initial), 0.0),
                     ConstantViscosityWithDivergence(0.0),
                     DryModel(),
                     NoRadiation(),
                     (Gravity(), Coriolis(), held_suarez_forcing!), 
                     NoFluxBC(),
                     setup)

  dg = DGModel(model, grid, NumericalFlux(),
               CentralNumericalFluxDiffusive(), CentralGradPenalty())

  timeend = 60 # FT(2 * setup.domain_halflength / 10 / setup.translation_speed)

  # determine the time step
  element_size = (setup.domain_height / num_elem_vert)
  acoustic_speed = soundspeed_air(FT(315))
  lucas_magic_factor = 14
  dt = lucas_magic_factor * element_size / acoustic_speed / polynomialorder ^ 2

  Q = init_ode_state(dg, FT(0))
  lsrk = LSRK144NiegemannDiehlBusch(dg, Q; dt = dt, t0 = 0)


  filter = ExponentialFilter(grid, 0, 14)
  cbfilter = EveryXSimulationSteps(1) do
    Filters.apply!(Q, 1:size(Q, 2), grid, filter)
    nothing
  end

  eng0 = norm(Q)
  dims == 2 && (numelems = (numelems..., 0))
  @info @sprintf """Starting
                    numelems  = (%d, %d, %d)
                    dt        = %.16e
                    norm(Q₀)  = %.16e
                    """ numelems... dt eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      runtime = Dates.format(convert(DateTime, now() - starttime[]), dateformat"HH:MM:SS")
      @info @sprintf """Update
                        simtime = %.16e
                        runtime = %s
                        norm(Q) = %.16e
                        """ gettime(lsrk) runtime energy
    end
  end
  callbacks = (cbinfo,cbfilter)

  #if output_vtk
  #  # create vtk dir
  #  vtkdir = "vtk_heldsuarez" *
  #    "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(FT)_level$(level)"
  #  mkpath(vtkdir)
  #  
  #  vtkstep = 0
  #  # output initial step
  #  do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

  #  # setup the output callback
  #  outputtime = timeend
  #  cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
  #    vtkstep += 1
  #    Qe = init_ode_state(dg, gettime(lsrk))
  #    do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
  #  end
  #  callbacks = (callbacks..., cbvtk)
  #end

  solve!(Q, lsrk; timeend=timeend, callbacks=callbacks)

  # final statistics
  engf = norm(Q)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  """ engf engf/eng0 engf-eng0
end

Base.@kwdef struct HeldSuarezSetup{FT}
  p_ground::FT = Float64(MSLP)
  T_initial::FT = 255.0
  domain_height::FT = 30e3
end

function (setup::HeldSuarezSetup)(state, aux, coords, t) 
  # callable to set initial conditions
  FT = eltype(state)

  r = norm(coords, 2)
  h = r - FT(planet_radius)

  scale_height = R_d * setup.T_initial / grav

  P_ref = MSLP * exp(-h / scale_height)
  state.ρ = air_density(setup.T_initial, P_ref)
  state.ρu = SVector{3, FT}(0, 0, 0)
  state.ρe = state.ρ * (internal_energy(setup.T_initial) + aux.orientation.Φ)
  nothing
end



function held_suarez_forcing!(source, state, aux, t::Real)
  FT = eltype(state)

  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  coord = aux.coord
  Φ = aux.orientation.Φ
  e = ρe / ρ
  u = ρu / ρ
  e_int = e - u' * u / 2 - Φ
  T = air_temperature(e_int)
  # Held-Suarez constants
  k_a = FT(1 / 40 / 86400)
  k_f = FT(1 / 86400)
  k_s = FT(1 / 4 / 86400)
  ΔT_y = FT(60)
  Δθ_z = FT(10)
  T_equator = FT(315)
  T_min = FT(200)
  scale_height = FT(7000) #from Smolarkiewicz JAS 2001 paper
  σ_b = FT(7 / 10)
  r = norm(coord, 2)
  @inbounds λ = atan(coord[2], coord[1])
  @inbounds φ = asin(coord[3] / r)
  h = r - FT(planet_radius)
  σ = exp(-h / scale_height)
  #p = air_pressure(T, ρ)
#  σ = p/p0
  exner_p = σ ^ (R_d / cp_d)
  Δσ = (σ - σ_b) / (1 - σ_b)
  height_factor = max(0, Δσ)
  T_equil = (T_equator - ΔT_y * sin(φ) ^ 2 - Δθ_z * log(σ) * cos(φ) ^ 2 ) * exner_p
  T_equil = max(T_min, T_equil)
  k_T = k_a + (k_s - k_a) * height_factor * cos(φ) ^ 4
  k_v = k_f * height_factor
  source.ρu += -k_v * ρu
  source.ρe += -k_T * ρ * cv_d * (T - T_equil) - k_v * ρu' * ρu / ρ
end


function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model, testname = "heldsuarez")
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

main()
