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

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("ShallowWater.jl")
include("../diagnostics.jl")

struct BickleyJet <: AbstractShallowWaterProblem end

function init_state_prognostic!(bl::ShallowWaterModel, 
                                ::BickleyJet,
                                state, aux, localgeo, t)
    FT = eltype(state)
    (x, y, z) = localgeo.coord

    ϵ = 0.1 # perturbation magnitude
    l = 0.5 # Gaussian width
    k = 0.5 # Sinusoidal wavenumber

    # The Bickley jet
    U = cosh(y)^(-2)

    # Slightly off-center vortical perturbations
    Ψ = exp(-(y + l / 10)^2 / (2 * (l^2))) * cos(k * x) * cos(k * y)

    # Vortical velocity fields (ũ, ṽ) = (-∂ʸ, +∂ˣ) ψ̃
    u = Ψ * (k * tan(k * y) + y / (l^2))
    v = -Ψ * k * tan(k * x)

    ρ = 1

    state.ρ = ρ
    state.ρu = ρ * @SVector [U + ϵ * u, ϵ * v]
    state.ρθ = ρ * sin(k * y)
end

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD
    
    #FT = Double64
    FT = Float64
    timeend = FT(200)
    surfaceflux = EntropyConservativeWithPenalty
    
    Lx = FT(4π)
    Ly = FT(4π)

    for Ndof in (32, 128, 512)
      for N in (1, 2, 3, 4)
        Ne = round(Int, Ndof / (N+1))
        run(
            mpicomm,
            Ndof,
            N,
            Ne,
            Lx,
            Ly,
            timeend,
            ArrayType,
            FT,
            surfaceflux
        )
      end
    end
end

function run(
    mpicomm,
    Ndof,
    N,
    Ne,
    Lx,
    Ly,
    timeend,
    ArrayType,
    FT,
    surfaceflux
)

    dim = 2
    brickrange = (
        range(FT(-Lx / 2), stop = Ly / 2, length = Ne + 1),
        range(FT(-Lx / 2), stop = Ly / 2, length = Ne + 1),
    )
    boundary = ((0, 0), (1, 1))
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
        polynomialorder = N,
    )

    problem = BickleyJet()
    g = FT(10)
    model = ShallowWaterModel(problem, g)

    esdg = ESDGModel(
        model,
        grid;
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = surfaceflux(),
    )

    # determine the time step
    dx = min_node_distance(grid)
    cfl = FT(1.0)
    dt = cfl * dx / sqrt(10)
    outputtime = 2
    dt = outputtime / ceil(outputtime / dt)
    nsteps = ceil(Int, timeend / dt)

    Q = init_ode_state(esdg, FT(0))

    η = similar(Q, vars = @vars(η::FT), nstate=1)
    ∫η0 = entropy_integral(esdg, η, Q)
    #η_int = function(dg, Q1)
    #  entropy_integral(dg, η, Q1)
    #end
    #η_prod = function(dg, Q1, Q2)
    #  entropy_product(dg, η, Q1, Q2)
    #end

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
        vtkdir = "esdg_bickley_jet/$surfaceflux/Ndof$(Ndof)/N$(N)"
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

function do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, N, testname = "bickley")
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

    writevtk(filename, Q, esdg, statenames, esdg.state_auxiliary, auxnames;
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
