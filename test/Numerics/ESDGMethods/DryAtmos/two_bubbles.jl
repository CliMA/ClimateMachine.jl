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
using ClimateMachine.ODESolvers
using StaticArrays: @SVector
using LazyArrays

using DoubleFloats
using GaussQuadrature
GaussQuadrature.maxiterations[Double64] = 40

using ClimateMachine.TemperatureProfiles: DryAdiabaticProfile

include("DryAtmos.jl")
include("../diagnostics.jl")

struct TwoBubbles <: AbstractDryAtmosProblem end

function init_state_prognostic!(bl::DryAtmosModel, 
                                ::TwoBubbles,
                                state, aux, localgeo, t)
    (x, z, _) = localgeo.coord
    FT = eltype(state)

    R_gas::FT = R_d(param_set)
    c_p::FT = cp_d(param_set)
    c_v::FT = cv_d(param_set)
    p0::FT = MSLP(param_set)
    _grav::FT = grav(param_set)
    γ::FT = c_p / c_v
    
    ## Reference temperature
    θ_ref::FT = 303.15
    Δθ::FT = 0

    # First bubble
    A₁ = FT(0.5)
    a₁ = 150
    s₁ = 50
    x₁ = 500
    z₁ = 300

    r₁ = sqrt((x - x₁) ^ 2 + (z - z₁) ^ 2)
    if r₁ <= a₁
      Δθ += A₁
    else
      Δθ += A₁ * exp(-(r₁ - a₁) ^ 2 / s₁ ^ 2)
    end
    
    # Second bubble
    A₂ = FT(-0.15)
    a₂ = 0
    s₂ = 50
    x₂ = 560
    z₂ = 640
    
    r₂ = sqrt((x - x₂) ^ 2 + (z - z₂) ^ 2)
    if r₂ <= a₂
      Δθ += A₂
    else
      Δθ += A₂ * exp(-(r₂ - a₂) ^ 2 / s₂ ^ 2)
    end

    ## Compute perturbed thermodynamic state:
    θ = θ_ref + Δθ                                      # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z             # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas)      # density
    T = θ * π_exner
    e_int = internal_energy(param_set, T)
    ts = PhaseDry(param_set, e_int, ρ)
    ρu = SVector(FT(0), FT(0), FT(0))                   # momentum
    ## State (prognostic) variable assignment
    e_kin = FT(0)                                       # kinetic energy
    e_pot = aux.Φ                                       # potential energy
    _cv_d = cv_d(param_set)
    e_int = _cv_d * T
    ρe = ρ * (e_kin + e_pot + e_int)

    ## Assign State Variables
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe
end

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    
    #FT = Double64
    FT = Float64

    mpicomm = MPI.COMM_WORLD

    xmax = FT(1000)
    zmax = FT(1000)
    timeend = 10 * 60
    outputtime = 6

    for Ndof in (50, 100, 200, 400)
      for N in (3, 4, 5, 6)
        Ne = round(Int, Ndof / (N+1))
        run(
            mpicomm,
            Ndof,
            N,
            Ne,
            xmax,
            zmax,
            timeend,
            outputtime,
            ArrayType,
            FT,
        )
      end
    end
end

function run(
    mpicomm,
    Ndof,
    N,
    Ne,
    xmax,
    zmax,
    timeend,
    outputtime,
    ArrayType,
    FT,
)

    dim = 2
    brickrange = (
        range(FT(0), stop = xmax, length = Ne + 1),
        range(FT(0), stop = zmax, length = Ne + 1),
    )
    boundary = ((1, 1), (1, 1))
    periodicity = (false, false)
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
        polynomialorder = N,
    )

    T_surface = FT(303.15)
    T_min_ref = FT(0)
    T_profile = DryAdiabaticProfile{FT}(param_set, T_surface, T_min_ref)
    ref_state = DryReferenceState(T_profile)

    problem = TwoBubbles()
    model = DryAtmosModel{dim}(FlatOrientation(),
                               problem;
                               ref_state=ref_state)

    esdg = ESDGModel(
        model,
        grid;
        volume_numerical_flux_first_order = EntropyConservative(),
        #surface_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = MatrixFlux(),
    )

    # determine the time step
    dx = min_node_distance(grid)
    cfl = FT(1.5)
    dt = cfl * dx / 330
    dt = outputtime / ceil(outputtime / dt)
    nsteps = ceil(Int, timeend / dt)

    Q = init_ode_state(esdg, FT(0))

    η = similar(Q, vars = @vars(η::FT), nstate=1)

    ∫η0 = entropy_integral(esdg, η, Q)

    η_int = function(dg, Q1)
      entropy_integral(dg, η, Q1)
    end
    η_prod = function(dg, Q1, Q2)
      entropy_product(dg, η, Q1, Q2)
    end

    odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)
    #odesolver = RLSRK144NiegemannDiehlBusch(esdg, η_int, η_prod, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem         = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      ∫η              = %.16e
                      """ "$ArrayType" "$FT" N Ne dt eng0 ∫η0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXSimulationSteps(100) do (s = false)
        if s
            starttime[] = now()
        else
            ∫η = entropy_integral(esdg, η, Q)
            dη = (∫η - ∫η0) / abs(∫η0)
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime            = %.16e
                              runtime            = %s
                              norm(Q)            = %.16e
                              ∫η                 = %.16e
                              (∫η - ∫η0) / |∫η0| = %.16e 
                              """ gettime(odesolver) runtime energy ∫η dη
        end
    end
    callbacks = (cbinfo,)

    output_vtk = true
    if output_vtk
        # create vtk dir
        vtkdir = vtkdir = "esdg_two_bubbles/Ndof$(Ndof)/N$(N)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, N)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, N)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, odesolver; callbacks = callbacks, numberofsteps = nsteps)

    # final statistics
    engf = norm(Q)
    ∫ηf = entropy_integral(esdg, η, Q)
    dηf = (∫ηf - ∫η0) / abs(∫η0)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    ∫η                      = %.16e
    (∫η - ∫η0) / |∫η0|      = %.16e 
    """ engf engf / eng0 engf - eng0 ∫ηf dηf
    engf
end

function do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, N, testname = "bubbles")
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

    writevtk(filename, Q, esdg, statenames, esdg.state_auxiliary, auxnames,
             number_sample_points = 2 * (N + 1))

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
