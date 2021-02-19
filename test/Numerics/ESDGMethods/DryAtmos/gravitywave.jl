using MPI
using ClimateMachine
using Logging
using ClimateMachine.DGMethods: ESDGModel, init_ode_state
using ClimateMachine.Mesh.Topologies: StackedBrickTopology
using ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid, min_node_distance
using ClimateMachine.Thermodynamics
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.VariableTemplates: flattenednames
import ClimateMachine.ODESolvers: LSRK144NiegemannDiehlBusch, solve!, gettime
using StaticArrays: @SVector
using LazyArrays

using ClimateMachine.TemperatureProfiles: IsothermalProfile

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("DryAtmos.jl")
import CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
CLIMAParameters.Planet.MSLP(::EarthParameterSet) = 1e5

Base.@kwdef struct GravityWave{FT} <: AbstractDryAtmosProblem
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
gw_small_setup(FT) = GravityWave{FT}(L=300e3, d=5e3, x_c=100e3, timeend=30*60)
gw_large_setup(FT) = GravityWave{FT}(L=24000e3, d=400e3, x_c=8000e3, timeend=3000*60)

function vars_state(::DryAtmosModel, ::GravityWave, ::Auxiliary, FT)
  @vars begin
    ρ_exact::FT
    ρu_exact::SVector{3, FT}
    ρe_exact::FT
  end
end

function init_state_prognostic!(bl::DryAtmosModel, 
                                problem::GravityWave,
                                state, aux, localgeo, t)
    x, z, _ = localgeo.coord
    FT = eltype(state)
    _R_d::FT = R_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _cv_d::FT = cv_d(param_set)
    p_s::FT = MSLP(param_set)
    g::FT = grav(param_set)

    L = problem.L
    d = problem.d
    x_c = problem.x_c
    u_0 = problem.u_0
    H = problem.H
    T_ref = problem.T_ref
    ΔT = problem.ΔT
    f = problem.f
  
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
   
    ρ = ρ_s * exp(-δ * z) + δρ
    T = T_ref + δT
    
    #ρ = aux.ref_state.ρ + δρ
    #T = aux.ref_state.T + δT

    u = SVector{3, FT}(u_0 + δu, δw, 0)
    e_kin = u' * u / 2
    e_pot = aux.Φ
    e_int = _cv_d * T
    ρe_tot = ρ * (e_int + e_kin + e_pot)

    state.ρ = ρ
    state.ρu = ρ * u
    state.ρe = ρe_tot
end

function main()
    ClimateMachine.init(parse_clargs=true)
    ArrayType = ClimateMachine.array_type()

    FT = Float64
    problem = gw_small_setup(FT)

    mpicomm = MPI.COMM_WORLD
    xmax = FT(problem.L)
    zmax = FT(problem.H)
    
    numlevels = 4
    l2_errors = zeros(FT, numlevels)
    linf_errors = zeros(FT, numlevels)

    for surfaceflux in (MatrixFlux,)
      for N in (1, 2, 3, 4, 5)
        ndof_x = 60
        ndof_y = 15

        Ne_x_base = round(Int, ndof_x / N)
        Ne_y_base = round(Int, ndof_y / N)

        Ne_x = Ne_x_base * 2 .^ ((1:numlevels) .- 1)
        Ne_y = Ne_y_base * 2 .^ ((1:numlevels) .- 1)

        for l in 1:numlevels
          timeend = problem.timeend
          FT = Float64
          l2_err, linf_err = run(
              mpicomm,
              N,
              (Ne_x[l], Ne_y[l]),
              xmax,
              zmax,
              timeend,
              problem,
              ArrayType,
              FT,
              surfaceflux()
          )
          @show l, l2_err, linf_err
          l2_errors[l] = l2_err
          linf_errors[l] = linf_err
        end
        l2_rates = log2.(l2_errors[1:numlevels-1] ./ l2_errors[2:numlevels])
        linf_rates = log2.(linf_errors[1:numlevels-1] ./ linf_errors[2:numlevels])

        path = "gravitywave_convergence_$(surfaceflux)_$N.txt"
        open(path, "w") do f
          msg = ""
          for l in 1:numlevels
            avg_Δx = problem.L / Ne_x[l] / N
            avg_Δy = problem.H / Ne_y[l] / N

            msg *= @sprintf(
              "%d %.4e %.4e %.16e %.4e %.16e %.4e\n",
              l,
              avg_Δx,
              avg_Δy,
              l2_errors[l],
              l > 1 ? l2_rates[l-1] : 0,
              linf_errors[l],
              l > 1 ? linf_rates[l-1] : 0,
            )
          end
          write(f, msg)
        end
      end
    end
end

function run(
    mpicomm,
    polynomialorder,
    Ne,
    xmax,
    zmax,
    timeend,
    problem,
    ArrayType,
    FT,
    surfaceflux
)

    dim = 2
    brickrange = (
        range(FT(0), stop = xmax, length = Ne[1] + 1),
        range(FT(0), stop = zmax, length = Ne[2] + 1),
    )
    boundary = ((0, 0), (1, 2))
    periodicity = (true, false)
    topology = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = periodicity,
        boundary = boundary,
    )
    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    T_profile = IsothermalProfile(param_set, FT(problem.T_ref))
    ref_state = DryReferenceState(T_profile)

    model = DryAtmosModel{dim}(FlatOrientation(),
                               problem;
                               ref_state=ref_state)

    esdg = ESDGModel(
        model,
        grid;
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = surfaceflux,
    )

    # determine the time step
    dx = min_node_distance(grid)
    cfl = FT(0.5)
    dt = cfl * dx / 330
    Q = init_ode_state(esdg, FT(0))
    odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem         = (%d, %d)
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      """ "$ArrayType" "$FT" polynomialorder Ne[1] Ne[2] dt eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXSimulationSteps(1000) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime = %.16e
                              runtime = %s
                              norm(Q) = %.16e
                              """ gettime(odesolver) runtime energy
        end
    end
    callbacks = (cbinfo,)

    output_vtk = false
    if output_vtk

        # create vtk dir
        Nelem = Ne[1]
        vtkdir =
            "esdg_small_gravitywave" *
            "_poly$(polynomialorder)_dims$(dim)_$(ArrayType)_$(FT)_nelem$(Nelem)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, esdg, Q, Q, model)

        # setup the output callback
        outputtime = timeend
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            Qexact = init_ode_state(esdg, FT(gettime(odesolver)))
            vtkstep += 1
            do_output(mpicomm, vtkdir, vtkstep, esdg, Q, Qexact, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, odesolver; callbacks = callbacks, timeend = timeend)
    Qexact = init_ode_state(esdg, FT(timeend))
    l2_err = norm(Q - Qexact)
    linf_err = maximum(abs.(Q - Qexact))

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)            = %.16e
    norm(Q) / norm(Q₀) = %.16e
    norm(Q) - norm(Q₀) = %.16e
    norm(Q - Qexact)   = %.16e
    """ engf engf / eng0 engf - eng0 l2_err
    l2_err, linf_err
end

function do_output(mpicomm, vtkdir, vtkstep, esdg, Q, Qexact, model, testname = "RTB")
    esdg.state_auxiliary.problem[:, 1, :] .= Qexact[:, 1, :]
    esdg.state_auxiliary.problem[:, 2:4, :] .= Qexact[:, 2:4, :]
    esdg.state_auxiliary.problem[:, 5, :] .= Qexact[:, 5, :]
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))

    writevtk(filename, Q, esdg, statenames, esdg.state_auxiliary, auxnames)#; number_sample_points = 10)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...), eltype(Q))

        @info "Done writing VTK: $pvtuprefix"
    end
end

main()
