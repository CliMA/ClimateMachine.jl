using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies:
    StackedCubedSphereTopology, cubedshellwarp, grid1d
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Elements: lglpoints, interpolationmatrix
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods: ESDGModel, init_ode_state, courant
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics: soundspeed_air
using ClimateMachine.TemperatureProfiles
using ClimateMachine.VariableTemplates: flattenednames

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cv_d, Omega, planet_radius, MSLP

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test
using JLD2
using CUDA

include("baroclinicwave.jl")

const output_vtk = false
const output_jld = true

function main()
    ClimateMachine.init(parse_clargs=true)
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 3

    #K = (16, 8)
    K = (8, 4)

    timeend = 15 * 24 * 3600
    outputtime = 24 * 3600
    tsoutputtime = 5 * 60

    FT = Float64
    result = run(
        mpicomm,
        polynomialorder,
        K,
        timeend,
        outputtime,
        tsoutputtime,
        ArrayType,
        FT,
    )
end

function run(
    mpicomm,
    N,
    K,
    timeend,
    outputtime,
    tsoutputtime,
    ArrayType,
    FT,
)

    outdir = joinpath("esdg_output",
                      "baroclinicwave",
                      "$N",
                      "$(K[1])x$(K[2])")

    Nq = N + 1
    _planet_radius::FT = planet_radius(param_set)
    domain_height = FT(30e3)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + domain_height),
        nelem = K[2],
    )
    topology = StackedCubedSphereTopology(mpicomm, K[1], vert_range)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
        meshwarp = cubedshellwarp,
    )

    T_profile = DecayingTemperatureProfile{FT}(param_set, FT(290), FT(220), FT(8e3))


    problem = BaroclinicWave()
    model = DryAtmosModel{FT}(SphericalOrientation(),
                              problem,
                              ref_state = DryReferenceState(T_profile),
                              sources = (Coriolis(),)
    )

    esdg = ESDGModel(
        model,
        grid,
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = MatrixFlux(),
    )

    linearmodel = DryAtmosAcousticGravityLinearModel(model)
    lineardg = DGModel(
        linearmodel,
        grid,
        RusanovNumericalFlux(),
        #CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = VerticalDirection(),
        state_auxiliary = esdg.state_auxiliary,
     )

    # determine the time step
    element_size = (domain_height / K[2])
    acoustic_speed = soundspeed_air(param_set, FT(330))
    #dt_factor = 1
    #dt = dt_factor * element_size / acoustic_speed / polynomialorder^2
    dx = min_node_distance(grid)
    cfl = 3.0
    dt = cfl * dx / acoustic_speed

    Q = init_ode_state(esdg, FT(0))

    #odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)

    linearsolver = ManyColumnLU()
    odesolver = ARK2GiraldoKellyConstantinescu(
        esdg,
        lineardg,
        LinearBackwardEulerSolver(linearsolver; isadjustable = false),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = false,
    )

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem_horz    = %d
                      numelem_vert    = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      """ "$ArrayType" "$FT" N K[1] K[2] dt eng0

    savejld2 = function(step, time)
      state_prognostic = Array(Q)
      state_auxiliary = Array(esdg.state_auxiliary)
      vgeo = Array(grid.vgeo)
      @save(joinpath(stepsdir, "bw_step$(lpad(step, 8, '0')).jld2"),
            model,
            problem,
            step,
            time,
            N,
            K,
            state_prognostic,
            state_auxiliary,
            vgeo)
    end

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
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
    cbcfl = EveryXSimulationSteps(100) do
            simtime = gettime(odesolver) 
            c_v = courant(
                nondiffusive_courant,
                esdg,
                model,
                Q,
                dt,
                simtime,
                VerticalDirection(),
            )
            c_h = courant(
                nondiffusive_courant,
                esdg,
                model,
                Q,
                dt,
                simtime,
                HorizontalDirection(),
            )
            ca_v = courant(
                advective_courant,
                esdg,
                model,
                Q,
                dt,
                simtime,
                VerticalDirection(),
            )
            ca_h = courant(
                advective_courant,
                esdg,
                model,
                Q,
                dt,
                simtime,
                HorizontalDirection(),
            )

            @info @sprintf """CFL
                              simtime = %.16e
                              Acoustic (vertical) Courant number    = %.2g
                              Acoustic (horizontal) Courant number  = %.2g
                              Advection (vertical) Courant number   = %.2g
                              Advection (horizontal) Courant number = %.2g
                              """ simtime c_v c_h ca_v ca_h
    end

    times = FT[]
    pmin = FT[]
    vmax = FT[]

    _grav = FT(grav(param_set))
    k1 = Array(view(esdg.state_auxiliary.data, :, 2, :)) ./ _grav
    k2 = Array(view(esdg.state_auxiliary.data, :, 3, :)) ./ _grav
    k3 = Array(view(esdg.state_auxiliary.data, :, 4, :)) ./ _grav

    cb_vel_p = EveryXSimulationSteps(floor(tsoutputtime / dt)) do
            γ = FT(gamma(param_set))
            simtime = gettime(odesolver)
            push!(times, simtime)
            
            ρ = Array(view(Q.data, :, 1, :))
            ρu = Array(view(Q.data, :, 2, :))
            ρv = Array(view(Q.data, :, 3, :))
            ρw = Array(view(Q.data, :, 4, :))
            ρe = Array(view(Q.data, :, 5, :))
            Φ = Array(view(esdg.state_auxiliary.data, :, 1, :))

            u = ρu ./ ρ
            v = ρv ./ ρ
            w = ρw ./ ρ

            s = @. k1 * u + k2 * v + k3 * w
            @. u -= s * k1
            @. v -= s * k2
            @. w -= s * k3
            
            vel = @. sqrt(u ^ 2 + v ^ 2 + w ^ 2)
            push!(vmax, maximum(vel))

            p = @. (γ - 1) * (ρe - (ρu ^ 2 + ρv ^ 2 + ρw ^ 2) / (2 * ρ) - ρ * Φ)
            psurf = @view p[1:Nq^2, 1:K[2]:end]
            push!(pmin, minimum(psurf))
    end
    callbacks = (cbinfo, cbcfl, cb_vel_p)

    if output_vtk
        # vorticity stuff
        ω = similar(Q; vars = @vars(ω::SVector{3, FT}), nstate = 3)
        vort_model = VorticityModel()
        vort_dg = DGModel(
            vort_model,
            grid,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
        )
        vortQ = init_ode_state(vort_dg, FT(0))
        ∇Φ1 = view(esdg.state_auxiliary.data, :, 2, :)
        ∇Φ2 = view(esdg.state_auxiliary.data, :, 3, :)
        ∇Φ3 = view(esdg.state_auxiliary.data, :, 4, :)
        
        # create vtk dir
        vtkdir = joinpath(outdir, "vtk")
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, Nq)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1

            vort_dg.state_auxiliary.data .= @view Q.data[:, 1:4, :]
            vort_dg(ω, vortQ, nothing, FT(0))
            
            ω1 = view(ω.data, :, 1, :)
            ω2 = view(ω.data, :, 2, :)
            ω3 = view(ω.data, :, 3, :)
            esdg.state_auxiliary.data[:, end, :]  .= @. (∇Φ1 * ω1 + ∇Φ2 * ω2 + ∇Φ3 * ω3) / _grav

            do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, Nq)
        end
        callbacks = (callbacks..., cbvtk)
    end
  
    if output_jld
      stepsdir = joinpath(outdir, "steps")
      mkpath(stepsdir)

      step = 0
      time = FT(0)
      savejld2(step, time)
      # setup the output callback
      cbstep = EveryXSimulationSteps(floor(outputtime / dt)) do
        step = getsteps(odesolver)
        time = gettime(odesolver)
        savejld2(step, time)
      end
      callbacks = (callbacks..., cbstep)
    end

    solve!(
        Q,
        odesolver;
        timeend = timeend,
        adjustfinalstep = false,
        callbacks = callbacks,
    )
    @save(joinpath(outdir, "timeseries.jld2"), times, pmin, vmax, N, K)

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    """ engf engf / eng0 engf - eng0
    engf
end


function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    Nq,
    testname = "baroclinicwave",
)
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
    writevtk(filename, Q, dg, statenames, dg.state_auxiliary, auxnames;
             number_sample_points=2 * Nq)

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
