using MPI
using ClimateMachine
using Logging
using ClimateMachine.DGMethods: ESDGModel, init_ode_state
using ClimateMachine.Mesh.Topologies: StackedBrickTopology
using ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid, min_node_distance
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.ODESolvers
using LazyArrays
using JLD2

using DoubleFloats
using GaussQuadrature
GaussQuadrature.maxiterations[Double64] = 40
using ClimateMachine.TemperatureProfiles: DryAdiabaticProfile

include("pbl_def.jl")
include("../diagnostics.jl")

function main()
    ClimateMachine.init()
    mpicomm = MPI.COMM_WORLD
    ArrayType = ClimateMachine.array_type()
    FT = Float64

    N = 5
    KH = 6
    KV = 6
    surfaceflux = MatrixFlux()

    timeend = 15000
    #timeend = 500
    outputtime = 200

    result = run(
        mpicomm,
        N,
        KH,
        KV,
        timeend,
        outputtime,
        ArrayType,
        FT,
        surfaceflux
    )
end

function run(
    mpicomm,
    N,
    KH,
    KV,
    timeend,
    outputtime,
    ArrayType,
    FT,
    surfaceflux,
)
    setup = PBL{FT}()

    horz_range = range(FT(0),
                       stop = setup.domain_length,
                       length = KH+1)
    vert_range = range(FT(0),
                       stop = setup.domain_height,
                       length = KV+1)
    brickrange = (horz_range, horz_range, vert_range)
    topology = StackedBrickTopology(mpicomm,
                                    brickrange,
                                    periodicity=(true, true, false))

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    dim = 3

    absorber = Absorber(FT(1020), FT(1000), FT(setup.domain_height))
    model = DryAtmosModel{dim}(
        FlatOrientation(),
        setup,
        sources = (HeatFlux(), absorber),
        drag_source=Drag(),
        ref_state=DryReferenceState(PBLProfile(setup))
    )

    esdg = ESDGModel(
        model,
        grid;
        volume_numerical_flux_first_order = EntropyConservative(),
        #surface_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = surfaceflux
    )

    # determine the time step
    dx = min_node_distance(grid)
    acoustic_speed = soundspeed_air(param_set, FT(setup.θ0))
    cfl = FT(1.5)
    dt = cfl * dx / acoustic_speed

    Q = init_ode_state(esdg, FT(0); init_on_cpu=true)

    η = similar(Q, vars = @vars(η::FT), nstate=1)

    ∫η0 = entropy_integral(esdg, η, Q)

    #η_int = function(dg, Q1)
    #  entropy_integral(dg, η, Q1)
    #end
    #η_prod = function(dg, Q1, Q2)
    #  entropy_product(dg, η, Q1, Q2)
    #end
    #odesolver = RLSRK144NiegemannDiehlBusch(esdg, η_int, η_prod, Q; dt = dt, t0 = 0)

    odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)
    

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem_horz    = %d
                      numelem_vert    = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      ∫η              = %.16e
                      """ "$ArrayType" "$FT" N KH KV dt eng0 ∫η0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXSimulationSteps(1000) do (s = false)
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
    cbcheck = EveryXSimulationSteps(1000) do
        @views begin
          ρ = Array(Q.data[:, 1, :])
          ρu = Array(Q.data[:, 2, :])
          ρv = Array(Q.data[:, 3, :])
          ρw = Array(Q.data[:, 4, :])
        end

        u = ρu ./ ρ
        v = ρv ./ ρ
        w = ρw ./ ρ

        ω = Array(Q.weights)
        ekin = (u .^ 2 + v .^ 2 + w .^ 2) ./ 2

        ∫ekin = sum(ω .* ekin)

        @info "u = $(extrema(u))"
        @info "v = $(extrema(v))"
        @info "w = $(extrema(w))"
        @info "∫ekin = $(∫ekin)"
        nothing
    end

    callbacks = (cbinfo, cbcheck)

    output_vtk = false
    if output_vtk
        # create vtk dir
        Nelem = Ne[1]
        vtkdir =
            "test_RTB" *
            "_poly$(N)_dims$(dim)_$(ArrayType)_$(FT)_nelem$(Nelem)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)

        # setup the output callback
        outputtime = 50
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)
        end
        callbacks = (callbacks..., cbvtk)
    end


    nsteps = ceil(Int, timeend / dt)

    #prof_steps = 0
    #mkpath("esdg_pbl_profs")
    #diagnostic_vars = pbl_diagnostic_vars(FT)
    #state_diagnostic = similar(Q;
    #                           vars = pbl_diagnostic_vars(FT),
    #                           nstate=varsize(diagnostic_vars))
    #cbdiagnostics = EveryXSimulationSteps(1) do
    #  step = getsteps(odesolver)
    #  time = gettime(odesolver)
    #  if mod(step, 10000) == 0 || (time > (timeend - 500) && mod(step, 100) == 0)
    #    prof_steps += 1
    #    nodal_diagnostics!(pbl_diagnostics!, diagnostic_vars, 
    #                       esdg, state_diagnostic, Q)
    #    variance_pairs = ((:θ, :θ), (:w, :θ), (:w, :w))
    #    z, profs, variances = profiles(diagnostic_vars, variance_pairs, esdg, state_diagnostic)

    #    s = @sprintf "z θ w θxθ wxθ, wxw\n"
    #    for k in 1:length(profs.θ)
    #      s *= @sprintf("%.16e %.16e %.16e %.16e %.16e %.16e\n",
    #                    z[k],
    #                    profs.θ[k],
    #                    profs.w[k],
    #                    variances.θxθ[k],
    #                    variances.wxθ[k],
    #                    variances.wxw[k])
    #    end
    #    open("esdg_pbl_profs/pbl_profiles_$step.txt", "w") do f
    #      write(f, s)
    #    end
    #  end
    #end
    #callbacks = (callbacks..., cbdiagnostics)

    fluxshort = surfaceflux.low_mach ? "LM" : "NLM"
    fluxshort *= "$(surfaceflux.Mcut)"
    fluxshort *= surfaceflux.kinetic_energy_preserving ? "_KEP" : "_NKEP"
   
    datapath = "esdg_pbl_data/N$N/$(KH)x$(KV)/$fluxshort/"
    mkpath(datapath)
    cbsave = EveryXSimulationSteps(1) do
      step = getsteps(odesolver)
      time = gettime(odesolver)
      if (time > (timeend - 500) && mod(step, 100) == 0)
        let
          state_prognostic = Array(Q)
          state_auxiliary = Array(esdg.state_auxiliary)
          vgeo = Array(grid.vgeo)

          @save("$datapath/pbl_$step.jld2",
                model,
                step,
                time,
                N, KH, KV,
                surfaceflux,
                state_prognostic,
                state_auxiliary,
                vgeo)
        end
      end
    end
    callbacks = (callbacks..., cbsave)

    solve!(Q, odesolver; callbacks = callbacks, timeend = timeend)

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

function do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, testname = "RTB")
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
