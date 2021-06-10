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
using JLD2

using ClimateMachine.TemperatureProfiles: IsothermalProfile

include("gravitywave.jl")
include("../../diagnostics.jl")

function main()
    ClimateMachine.init(parse_clargs=true)
    ArrayType = ClimateMachine.array_type()

    FT = Float64
    problem = gw_small_setup(FT)

    mpicomm = MPI.COMM_WORLD
    xmax = FT(problem.L)
    zmax = FT(problem.H)
    
    #numlevels = 4
    numlevels = 2

    outdir = joinpath("esdg_output", "gravitywave")

    for surfaceflux in (EntropyConservative,)
      outdir = joinpath(outdir, "$surfaceflux")
      mkpath(outdir)

      convergence_data = Dict()
      for N in (2, 3)
        ndof_x = 60
        ndof_y = 15

        Kx_base = round(Int, ndof_x / N)
        Ky_base = round(Int, ndof_y / N)

        Kx = Kx_base * 2 .^ ((1:numlevels) .- 1)
        Ky = Ky_base * 2 .^ ((1:numlevels) .- 1)

        l2_errors = zeros(FT, numlevels)
        linf_errors = zeros(FT, numlevels)
        l2_errors_state = zeros(FT, numlevels, 5)
        linf_errors_state = zeros(FT, numlevels, 5)
        for l in 1:numlevels
          timeend = problem.timeend
          FT = Float64
          l2_err, l2_err_state, linf_err, linf_err_state = run(
              mpicomm,
              N,
              (Kx[l], Ky[l]),
              xmax,
              zmax,
              timeend,
              problem,
              ArrayType,
              FT,
              surfaceflux(),
              outdir
          )
          @show l, l2_err, linf_err
          l2_errors[l] = l2_err
          linf_errors[l] = linf_err
          l2_errors_state[l, :] .= l2_err_state
          linf_errors_state[l, :] .= linf_err_state
        end
        l2_rates = log2.(l2_errors[1:numlevels-1] ./ l2_errors[2:numlevels])
        linf_rates = log2.(linf_errors[1:numlevels-1] ./ linf_errors[2:numlevels])
        
        l2_rates_state = log2.(l2_errors_state[1:numlevels-1, :] ./ l2_errors_state[2:numlevels, :])
        linf_rates_state = log2.(linf_errors_state[1:numlevels-1, :] ./ linf_errors_state[2:numlevels, :])
    
        avg_dx = problem.L ./ Kx ./ N
        avg_dy = problem.H ./ Ky ./ N

        convergence_data[N] = (;
              avg_dx,
              avg_dy,
              l2_errors,
              l2_rates,
              linf_errors,
              linf_rates,
              l2_errors_state,
              l2_rates_state,
              linf_errors_state,
              linf_rates_state,
        )

        @show N
        @show l2_rates
        @show l2_rates_state[:, 1]
        @show l2_rates_state[:, 2]
        @show l2_rates_state[:, 3]
        @show l2_rates_state[:, 5]
        @show linf_rates
        @show linf_rates_state[:, 1]
        @show linf_rates_state[:, 2]
        @show linf_rates_state[:, 3]
        @show linf_rates_state[:, 5]
      end
      @save(joinpath(outdir, "gw_convergence_$surfaceflux.jld2"),
            convergence_data,
           )
    end
end

function run(
    mpicomm,
    N,
    K,
    xmax,
    zmax,
    timeend,
    problem,
    ArrayType,
    FT,
    surfaceflux,
    outdir
)
    dim = 2
    brickrange = (
        range(FT(0), stop = xmax, length = K[1] + 1),
        range(FT(0), stop = zmax, length = K[2] + 1),
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
        polynomialorder = N,
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
    cfl = FT(0.1)
    dt = cfl * dx / 330
    Q = init_ode_state(esdg, FT(0))
    #odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)
    odesolver = LSRK54CarpenterKennedy(esdg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem         = (%d, %d)
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      """ "$ArrayType" "$FT" N K[1] K[2] dt eng0

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
        Nelem = K[1]
        vtkdir =
            "esdg_small_gravitywave" *
            "_poly$(N)_dims$(dim)_$(ArrayType)_$(FT)_nelem$(Nelem)"
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

    l2_err = norm(Q - Qexact) / norm(Qexact)
    l2_err_state = norm(Q - Qexact, dims = (1, 3)) ./ norm(Qexact, dims = (1, 3))
    l2_err_state = l2_err_state[:]
    
    Q = Array(Q.data)
    Qexact = Array(Qexact.data)
    nstate = size(Q, 2)
    linf_err = maximum(abs.(Q - Qexact)) / maximum(abs.(Qexact))
    linf_err_state = [
      @views maximum(abs.(Q[:, s, :] - Qexact[:, s, :])) / maximum(abs.(Qexact[:, s, :]))
      for s in 1:nstate
     ]

    stepsdir = joinpath(outdir, "$N", "$(K[1])x$(K[2])", "steps")
    mkpath(stepsdir)

    step = getsteps(odesolver)
    time = gettime(odesolver)
    let
      state_prognostic = Array(Q)
      state_exact = Array(Qexact)
      state_auxiliary = Array(esdg.state_auxiliary)
      vgeo = Array(grid.vgeo)
      @save(joinpath(stepsdir, "gw_step$(lpad(step, 8, '0')).jld2"),
            model,
            problem,
            step,
            time,
            N,
            K,
            surfaceflux,
            state_prognostic,
            state_exact,
            state_auxiliary,
            vgeo)
    end

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)            = %.16e
    norm(Q) / norm(Q₀) = %.16e
    norm(Q) - norm(Q₀) = %.16e
    norm(Q - Qexact)   = %.16e
    """ engf engf / eng0 engf - eng0 l2_err
    l2_err, l2_err_state, linf_err, linf_err_state
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
