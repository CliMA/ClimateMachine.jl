using CLIMA: haspkg
using CLIMA.Mesh.Topologies: StackedCubedSphereTopology, cubedshellwarp, grid1d
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid
using CLIMA.Mesh.Geometry: LocalGeometry
using CLIMA.Mesh.Filters
using CLIMA.DGmethods: DGModel, init_ode_state
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralGradPenalty,
                                       CentralNumericalFluxDiffusive
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.MPIStateArrays: euclidean_distance
using CLIMA.PlanetParameters: R_d, grav, MSLP, planet_radius, cp_d, cv_d, day, kappa_d
using CLIMA.MoistThermodynamics: air_density, total_energy, soundspeed_air, internal_energy, air_temperature
using CLIMA.Atmos: AtmosModel, SphericalOrientation, ReferenceState,
                   DryModel, NoRadiation, NoFluxBC,
                   ConstantViscosityWithDivergence,
                   vars_state, vars_aux,
                   Gravity, Coriolis,
                   HydrostaticState, IsothermalProfile
using CLIMA.VariableTemplates: Vars, flattenednames, @vars
import CLIMA.Atmos: atmos_init_aux!
import CLIMA.DGmethods: vars_aux

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test
@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayType = CuArray
else
  const ArrayType = Array
end

const output_vtk = true

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
  numelem_horz = 20
  numelem_vert = 2
  timeend = 3600
  outputtime = 100
  
  for FT in (Float64,)

    run(mpicomm, polynomialorder, numelem_horz, numelem_vert,
        timeend, outputtime, ArrayType, FT)
  end
end

function run(mpicomm, polynomialorder, numelem_horz, numelem_vert,
             timeend, outputtime, ArrayType, FT)

  setup = GravityWaveSetup{FT}()

  a = FT(planet_radius / setup.X)
  vert_range = grid1d(a, a + setup.z_top, nelem = numelem_vert)
  topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

  grid = DiscontinuousSpectralElementGrid(topology,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder,
                                          meshwarp = cubedshellwarp)

  model = AtmosModel(SphericalOrientation(),
                     GravityWaveRefState(setup),
                     ConstantViscosityWithDivergence(FT(0)),
                     DryModel(),
                     NoRadiation(),
                     (Gravity(),),
                     NoFluxBC(),
                     setup)

  dg = DGModel(model, grid, Rusanov(),
               CentralNumericalFluxDiffusive(), CentralGradPenalty())
  Q = init_ode_state(dg, FT(0))

  # determine the time step
  element_size = (setup.z_top / numelem_vert)
  acoustic_speed = soundspeed_air(FT(315))
  dt = element_size / acoustic_speed / polynomialorder ^ 2
  @info "Acoustic dt estimate = $dt"
  dt = 0.001

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  #filterorder = 14
  #filter = ExponentialFilter(grid, 0, filterorder)
  #cbfilter = EveryXSimulationSteps(1) do
  #  Filters.apply!(Q, 1:size(Q, 2), grid, filter)
  #  nothing
  #end

  eng0 = norm(Q)
  @info @sprintf """Starting
                    ArrayType       = %s
                    FT              = %s
                    polynomialorder = %d
                    numelem_horz    = %d
                    numelem_vert    = %d
                    dt              = %.16e
                    norm(Q₀)        = %.16e
                    """ "$ArrayType" "$FT" polynomialorder numelem_horz numelem_vert dt eng0

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
  callbacks = (cbinfo,)

  if output_vtk
    # create vtk dir
    vtkdir = "vtk_gravitywave" *
      "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
      "_$(ArrayType)_$(FT)"
    mkpath(vtkdir)

    vtkstep = 0
    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)

    # setup the output callback
    cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, gettime(lsrk))
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=callbacks)

  # final statistics
  engf = norm(Q)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  """ engf engf/eng0 engf-eng0
end

Base.@kwdef struct GravityWaveSetup{FT}
  z_top::FT = 10e3
  p_top::FT = 273.919e2
  X::FT = 125
  Ω::FT = 0
  u_0::FT = 20
  z_s::FT = 0
  N::FT = 0.01
  T_eq::FT = 300
  p_eq::FT = 1e5
  d::FT = 5e3
  λ_c::FT = 2pi / 3
  φ_c::FT = 0
  ΔΘ::FT = 1
  L_z::FT = 20e3
end

struct GravityWaveRefState{FT} <: ReferenceState
  setup::GravityWaveSetup{FT}
end
vars_aux(::GravityWaveRefState, FT) = @vars(ρ::FT, p::FT, T::FT, ρe::FT)
function atmos_init_aux!(ref::GravityWaveRefState, atmos::AtmosModel, aux::Vars, geom::LocalGeometry)
  FT = eltype(aux)
  coord = geom.coord
  u_0 = ref.setup.u_0
  N = ref.setup.N
  T_eq = ref.setup.T_eq
  p_eq = ref.setup.p_eq
  Ω = ref.setup.Ω
  
  a_ref = FT(planet_radius)
  a = a_ref / ref.setup.X
  r = norm(coord, 2)
  @inbounds λ = atan(coord[2], coord[1])
  @inbounds φ = asin(coord[3] / r)
  z = r - a

  G = grav ^ 2 / (N ^ 2 * cp_d)
  T_s = G + (T_eq - G) * exp(-u_0 * N ^ 2 / (4grav ^ 2) * (u_0 + 2Ω * a) * (cos(2φ) - 1))
  T_b = G * (1 - exp(N ^ 2 / grav * z)) + T_s * exp(N ^ 2 / grav * z)
  p_s = p_eq * exp(u_0 / (4G * R_d) * (u_0 + 2Ω * a) * (cos(2φ) - 1)) * (T_s / T_eq) ^ (1 / kappa_d)
  p = p_s * (G / T_s * exp(-N ^ 2 / grav * z) + 1 - G / T_s) ^ (1 / kappa_d)
  ρ = p / (R_d * T_b) 
  
  aux.ref_state.T = T_b
  aux.ref_state.p = p
  aux.ref_state.ρ = ρ

  e_kin = FT(0)
  e_pot = aux.orientation.Φ
  aux.ref_state.ρe = ρ * total_energy(e_kin, e_pot, T_b)
end

function (setup::GravityWaveSetup)(state, aux, coord, t) 
  FT = eltype(state)
  u_0 = setup.u_0
  N = setup.N
  T_eq = setup.T_eq
  Ω = setup.Ω
  φ_c = setup.φ_c
  λ_c = setup.λ_c
  d = setup.d
  L_z = setup.L_z
  ΔΘ = setup.ΔΘ
  
  a_ref = FT(planet_radius)
  a = a_ref / setup.X
  r = norm(coord, 2) 
  @inbounds λ = atan(coord[2], coord[1])
  @inbounds φ = asin(coord[3] / r)
  z = r - a

  r = a * (acos(sin(φ_c) * sin(φ) + cos(φ_c) * cos(φ) * cos(λ - λ_c)))
  s = d ^ 2 / (d ^ 2 + r ^ 2)
  δΘ = ΔΘ * s * sin(2pi * z / L_z)
  δT = δΘ * (aux.ref_state.p / MSLP) ^ kappa_d
  T = aux.ref_state.T + δT
  ρ = air_density(T, aux.ref_state.p)
  
  u_sphere = SVector{3, FT}(u_0 * cos(φ), 0, 0)
  u = SVector{3, FT}(
    -u_sphere[1] * sin(λ) - u_sphere[2] *sin(φ) * cos(λ),
     u_sphere[1] * cos(λ) - u_sphere[2] *sin(φ) * sin(λ),
     u_sphere[2] * cos(φ))

  state.ρ = ρ
  state.ρu = ρ * u
  e_kin = u' * u / 2
  e_pot = aux.orientation.Φ
  state.ρe = ρ * total_energy(e_kin, e_pot, T)
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, model, testname = "gravitywave")
  ## name of the file that this MPI rank will write
  filename = @sprintf("%s/%s_mpirank%04d_step%04d",
                      vtkdir, testname, MPI.Comm_rank(mpicomm), vtkstep)

  statenames = flattenednames(vars_state(model, eltype(Q)))
  auxnames = flattenednames(vars_aux(model, eltype(Q))) 
  writevtk(filename, Q, dg, statenames, dg.auxstate, auxnames)

  ## Generate the pvtu file for these vtk files
  if MPI.Comm_rank(mpicomm) == 0
    ## name of the pvtu file
    pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

    ## name of each of the ranks vtk files
    prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
      @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
    end

    writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...))

    @info "Done writing VTK: $pvtuprefix"
  end
end

main()
