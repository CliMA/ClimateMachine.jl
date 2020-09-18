using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.DGMethods
using ClimateMachine.NumericalFluxes
using StaticArrays
using Test


# FIXME: How to do this in a less hacky way ?
using ClimateMachine.VTK, ClimateMachine.BalanceLaws
using Printf, MPI
import ClimateMachine.Callbacks
function Callbacks.vtk(vtk_opt, solver_config, output_dir, number_sample_points)
  cb_constr = Callbacks.CB_constructor(vtk_opt, solver_config)
  cb_constr === nothing && return nothing

  vtknum = Ref(1)

  mpicomm = solver_config.mpicomm
  dg = solver_config.dg
  bl = dg.balance_law
  Q = solver_config.Q
  FT = eltype(Q)
  state_auxilary = solver_config.dg.state_auxiliary

  cb_vtk = GenericCallbacks.AtInitAndFini() do
      simtime = ODESolvers.gettime(solver_config.solver)
      Qe = init_ode_state(dg, simtime)
      state_auxilary.ρ_exact .= Qe.ρ
      state_auxilary.ρu_exact .= Qe.ρu
      state_auxilary.ρe_exact .= Qe.ρe

      # TODO: make an object
      vprefix = @sprintf(
          "%s_mpirank%04d_num%04d",
          solver_config.name,
          MPI.Comm_rank(mpicomm),
          vtknum[],
      )
      outprefix = joinpath(output_dir, vprefix)

      statenames = flattenednames(vars_state(bl, Prognostic(), FT))
      auxnames = flattenednames(vars_state(bl, Auxiliary(), FT))

      writevtk(
          outprefix,
          Q,
          dg,
          statenames,
          dg.state_auxiliary,
          auxnames;
          number_sample_points = number_sample_points,
      )

      # Generate the pvtu file for these vtk files
      if MPI.Comm_rank(mpicomm) == 0
          # name of the pvtu file
          pprefix = @sprintf("%s_num%04d", solver_config.name, vtknum[])
          pvtuprefix = joinpath(output_dir, pprefix)

          # name of each of the ranks vtk files
          prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
              @sprintf(
                  "%s_mpirank%04d_num%04d",
                  solver_config.name,
                  i - 1,
                  vtknum[],
              )
          end
          writepvtu(
              pvtuprefix,
              prefixes,
              (statenames..., auxnames...),
              eltype(Q),
          )
      end

      vtknum[] += 1
      nothing
  end
  return cb_constr(cb_vtk)
end

using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import CLIMAParameters
CLIMAParameters.Planet.MSLP(::EarthParameterSet) = 1e5

Base.@kwdef struct GravityWaveSetup{FT}
  T_ref::FT = 250
  #ΔT::FT = 0.01
  ΔT::FT = 0.0001
  H::FT = 10e3
  u_0::FT = 20
  f::FT = 0
  L::FT
  d::FT
  x_c::FT
  timeend::FT
end
gw_small_setup(FT) = GravityWaveSetup{FT}(L=300e3, d=5e3, x_c=100e3, timeend=30*60)
gw_large_setup(FT) = GravityWaveSetup{FT}(L=24000e3, d=400e3, x_c=8000e3, timeend=3000*60)

function (setup::GravityWaveSetup{FT})(problem, bl, state, aux, (x, y, z), t) where {FT}
    _R_d::FT = R_d(bl.param_set)
    _cp_d::FT = cp_d(bl.param_set)
    _cv_d::FT = cv_d(bl.param_set)
    p_s::FT = MSLP(bl.param_set)
    g::FT = grav(bl.param_set)

    L = setup.L
    d = setup.d
    x_c = setup.x_c
    u_0 = setup.u_0
    H = setup.H
    T_ref = setup.T_ref
    ΔT = setup.ΔT
    f = setup.f
  
    δ = g / (_R_d * T_ref)
    c_s = sqrt(_cp_d / _cv_d * _R_d * T_ref)
    ρ_s = p_s / (T_ref * _R_d)

    if t == 0
      δT_b = ΔT * exp(-(x - x_c) ^ 2 / d ^ 2) * sin(π * z / H)
      δT = exp(δ * z / 2) * δT_b
      δρ_b = -ρ_s * δT_b / T_ref
      δρ = exp(-δ * z / 2) * δρ_b
      δu, δv, δw = 0, 0, 0
    else
      xp = x - u_0 * t

      δρ_b, δu_b, δv_b, δw_b, δp_b = zeros(SVector{5, Complex{FT}})
      for m in (-1, 1)
        for n in -100:100
          k_x = 2π * n / L
          k_z = π * m / H

          p_1 = c_s ^ 2 * (k_x ^ 2 + k_z ^ 2 + δ ^ 2 / 4) + f ^ 2
          q_1 = g * k_x ^ 2 * (c_s ^ 2 * δ - g) + c_s ^ 2 * f ^ 2 * (k_z ^ 2 + δ ^ 2 / 4)
          
          α = sqrt(p_1 / 2 - sqrt(p_1 ^ 2 / 4 - q_1))
          β = sqrt(p_1 / 2 + sqrt(p_1 ^ 2 / 4 - q_1))

          fac1 = 1 / (β ^ 2 - α ^ 2) 
          L_m1 = (-cos(α * t) / α ^ 2 + cos(β * t) / β ^ 2) * fac1 + 1 / (α ^ 2 * β ^ 2)
          L_0 = (sin(α * t) / α - sin(β * t) / β) * fac1
          L_1 = (cos(α * t) - cos(β * t)) * fac1
          L_2 = (-α * sin(α * t) + β * sin(β * t)) * fac1
          L_3 = (-α ^ 2 * cos(α * t) + β ^ 2 * cos(β * t)) * fac1
          
          if α == 0
            L_m1 = (β ^ 2 * t ^ 2 - 1 + cos(β * t)) / β ^ 4
            L_0 = (β * t - sin(β * t)) / β ^ 3
          end
      
          δρ̃_b0 = -ρ_s / T_ref * ΔT / sqrt(π) * d / L *
                  exp(-d ^ 2 * k_x ^ 2 / 4) * exp(-im * k_x * x_c) * k_z * H / 2im

          δρ̃_b = (L_3 + (p_1 + g * (im * k_z - δ / 2)) * L_1 +
                (c_s ^ 2 * (k_z ^ 2 + δ ^ 2 / 4) + g * (im * k_z - δ / 2)) * f ^ 2 * L_m1) * δρ̃_b0

          δp̃_b = -(g - c_s ^ 2 * (im * k_z + δ / 2)) * (L_1 + f ^ 2 * L_m1) * g * δρ̃_b0

          δũ_b = im * k_x * (g - c_s ^ 2 * (im * k_z + δ / 2)) * L_0 * g * δρ̃_b0 / ρ_s

          δṽ_b = -f * im * k_x * (g - c_s ^ 2 * (im * k_z + δ / 2)) * L_m1 * g * δρ̃_b0 / ρ_s 

          δw̃_b = -(L_2 + (f ^ 2 + c_s ^ 2 * k_x ^ 2) * L_0) * g * δρ̃_b0 / ρ_s 

          expfac = exp(im * (k_x * xp + k_z * z)) 
          
          δρ_b += δρ̃_b * expfac
          δp_b += δp̃_b * expfac

          δu_b += δũ_b * expfac
          δv_b += δṽ_b * expfac
          δw_b += δw̃_b * expfac
        end
      end

      δρ = exp(-δ * z / 2) * real(δρ_b)
      δp = exp(-δ * z / 2) * real(δp_b)

      δu = exp(δ * z / 2) * real(δu_b)
      δv = exp(δ * z / 2) * real(δv_b)
      δw = exp(δ * z / 2) * real(δw_b)

      δT_b = T_ref * (δp_b / p_s - δρ_b / ρ_s)
      δT = exp(δ * z / 2) * real(δT_b)
    end
   
    #ρ = ρ_s * exp(-δ * z) + δρ
    #T = T_ref + δT
    
    ρ = aux.ref_state.ρ + δρ
    T = aux.ref_state.T + δT

    u = SVector{3, FT}(u_0 + δu, δv, δw) 
    e_kin = u' * u / 2
    e_pot = gravitational_potential(bl.orientation, aux)
    e_int = internal_energy(bl.param_set, T, PhasePartition(FT(0)))
    ρe_tot = ρ * (e_int + e_kin + e_pot)

    state.ρ = ρ
    state.ρu = ρ * u
    state.ρe = ρe_tot
end

function config_gravitywave(FT, N, resolution, setup)
    ## Define the time integrator:
    ## We chose an explicit single-rate LSRK144 for this problem
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    ## Setup the source terms for this problem:
    source = (Gravity(),)

    temp_profile_ref = IsothermalProfile(param_set, setup.T_ref)
    ref_state = HydrostaticState(temp_profile_ref)

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        init_state_prognostic = setup,
        ref_state = ref_state,
        turbulence = ConstantKinematicViscosity(FT(0)),
        moisture = DryModel(),
        source = source,
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "BaldaufBrdarGravityWave", # Problem title [String]
        N,                       # Polynomial order [Int]
        resolution,              # (Δx, Δy, Δz) effective resolution [m]
        setup.L,                 # Domain maximum size [m]
        N * resolution[2],                     # Domain maximum size [m]
        setup.H,                  # Domain maximum size [m]
        param_set,               # Parameter set.
        setup,             # Function specifying initial condition
        solver_type = ode_solver,# Time-integrator type
        model = model,           # Model type
        numerical_flux_first_order=RoeNumericalFlux
    )

    return config
end

# Define a `main` method (entry point)
function main()
    FT = Float64
    setup = gw_small_setup(FT)

    ## Define the polynomial order and effective grid spacings:
    N = 3

    r = 1
    numelem_x = r * 64
    numelem_z = r * 8

    Δx = FT(setup.L / numelem_x / N)
    Δy = FT(Δx)
    Δz = FT(setup.H / numelem_z / N)
    resolution = (Δx, Δy, Δz)

    t0 = FT(0)
    timeend = FT(setup.timeend)

    ## Define the max Courant for the time time integrator (ode_solver).
    ## The default value is 1.7 for LSRK144:
    CFL = FT(0.5)

    ## Assign configurations so they can be passed to the `invoke!` function
    driver_config = config_gravitywave(FT, N, resolution, setup)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFL,
    )

    ## Set up the spectral filter to remove the solutions spurious modes
    ## Define the order of the exponential filter: use 32 or 64 for this problem.
    ## The larger the value, the less dissipation you get:
    filterorder = 64
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            AtmosFilterPerturbations(driver_config.bl),
            solver_config.dg.grid,
            filter,
            state_auxiliary = solver_config.dg.state_auxiliary,
        )
        nothing
    end
    ## End exponential filter

    ## Invoke solver (calls `solve!` function for time-integrator),
    ## pass the driver, solver and diagnostic config information.
    result = ClimateMachine.invoke!(
        solver_config;
        user_callbacks = (cbfilter,),
        check_euclidean_distance = true,
    )

end

# Call `main`
main()
