using ProgressMeter

abstract type AbstractCallback end

struct Default <: AbstractCallback end
struct Info <: AbstractCallback end
struct CFL <: AbstractCallback end
struct StateCheck{T} <: AbstractCallback
    number_of_checks::T
end

Base.@kwdef struct Progress{T, B} <: AbstractCallback
    update_seconds::T = 1  # Time in seconds between updates.
    show_extrema::B = true # Whether or not to show extrema of `ρu/ρ`.
end

Base.@kwdef struct JLD2State{T, V, B} <: AbstractCallback
    iteration::T
    filepath::V
    overwrite::B = true
end

Base.@kwdef struct VTKState{T, V, C, B} <: AbstractCallback
    iteration::T = 1
    filepath::V = "."
    counter::C = [0]
    overwrite::B = true
end

Base.@kwdef struct NetCDF{T, V, R, C, B} <: AbstractCallback
    iteration::T = 1
    filepath::V = "."
    resolution::R = (2.0, 2.0, 2000.0)
    counter::C = [0]
    overwrite::B = true
end

Base.@kwdef struct TMARCallback{ℱ} <: AbstractCallback 
    filterstates::ℱ = 6:6
end

Base.@kwdef struct ReferenceStateUpdate{ℱ} <: AbstractCallback 
    recompute::ℱ = 20
end

function create_callbacks(simulation::Simulation, odesolver)
    callbacks = simulation.callbacks

    if isempty(callbacks)
        return ()
    else
        cbvector = [
            create_callback(callback, simulation, odesolver)
            for callback in callbacks
        ]
        return tuple(cbvector...)
    end
end

function create_callback(::Default, simulation::Simulation, odesolver)
    cb_info = create_callback(Info(), simulation, odesolver)
    cb_state_check = create_callback(StateCheck(10), simulation, odesolver)

    return (cb_info, cb_state_check)
end

function create_callback(::Info, simulation::Simulation, odesolver)
    Q = simulation.state
    timeend = simulation.time.finish
    mpicomm = MPI.COMM_WORLD

    starttime = Ref(now())
    cbinfo = ClimateMachine.GenericCallbacks.EveryXWallTimeSeconds(
        60,
        mpicomm,
    ) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            @info @sprintf(
                """Update
                simtime = %8.4f / %8.4f
                runtime = %s
                norm(Q) = %.16e""",
                ClimateMachine.ODESolvers.gettime(odesolver),
                timeend,
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )

            if isnan(energy)
                error("NaNs")
            end
        end
    end

    return cbinfo
end

function create_callback(::CFL, simulation::Simulation, odesolver)
    Q = simulation.state
    # timeend = simulation.time.finish
    # mpicomm = MPI.COMM_WORLD
    # starttime = Ref(now())
    cbcfl = EveryXSimulationSteps(100) do
            simtime = gettime(odesolver)

            @views begin
                ρ = Array(Q.data[:, 1, :])
                ρu = Array(Q.data[:, 2, :])
                ρv = Array(Q.data[:, 3, :])
                ρw = Array(Q.data[:, 4, :])
            end

            u = ρu ./ ρ
            v = ρv ./ ρ
            w = ρw ./ ρ

            # TODO! transform onto sphere

            ue = extrema(u)
            ve = extrema(v)
            we = extrema(w)

            @info @sprintf """CFL
                    simtime = %.16e
                    u = (%.4e, %.4e)
                    v = (%.4e, %.4e)
                    w = (%.4e, %.4e)
                    """ simtime ue... ve... we...
        end

    return cbcfl
end

function create_callback(callback::StateCheck, simulation::Simulation, _...)
    sim_length = simulation.time.finish - simulation.time.start
    timestep = simulation.timestepper.timestep
    nChecks = callback.number_of_checks

    nt_freq = floor(Int, sim_length / timestep / nChecks)

    cbcs_dg = ClimateMachine.StateCheck.sccreate(
        [(simulation.state, "state")],
        nt_freq,
    )

    return cbcs_dg
end

time_string(seconds::Real) = string(
    @sprintf("%.3f", seconds),
    " s (",
    Dates.canonicalize(Dates.Second(round(Int, seconds))),
    ")",
)

function create_callback(p::Progress, simulation::Simulation, odesolver)
    timeend = simulation.time.finish
    Q = simulation.state

    # To allow for an accurate estimate of the remaining time, this needs to be
    # initialized at the end of the first timestep, by which point almost all
    # compilation should be done.
    progress = Ref{ProgressMeter.Progress}()

    if p.show_extrema
        # This code can easily be generalized beyond just `ρu/ρ`. Perhaps the
        # boolean `show_extrema` could be replaced with a tuple of symbols
        # representing extrema to compute? Or perhpas this tuple could be
        # automatically generated from the balance law?
        # TODO: Fix the extrema code to eliminate unnecesary allocations. The
        #       following snippet would be nice, but it doesn't work on the GPU
        #       because of `allowscalar(false)`:
        #       extrema(
        #           ((ijk, e),) -> Q.data[ijk, iv, e] / Q.data[ijk, iρ, e],
        #           Iterators.product(size(Q.data, 1), size(Q.data, 3)),
        #       )
        #       This can wait until the code has been more widely tested...
        Qvars = ClimateMachine.MPIStateArrays.vars(Q)
        iρ = ClimateMachine.VariableTemplates.varsindex(Qvars, :ρ)[1]
        iρus = ClimateMachine.VariableTemplates.varsindex(Qvars, :ρu)
        show_extrema = () -> map(enumerate(iρus)) do (number, iρu)
            (
                "Extrema of ρu[$number]/ρ",
                @sprintf(
                    "%.4e, %.4e",
                    extrema(Array(Q.data[:, iρu, :] ./ Q.data[:, iρ, :]))...,
                ),
            )
        end
    else
        show_extrema = () -> ()
    end

    showvalues = () -> [
        ("Wall-Clock Time", time_string(time() - progress[].tinit)),
        (
            "Simulation Time",
            time_string(ClimateMachine.ODESolvers.gettime(odesolver)),
        ),
        show_extrema()...,
        ("norm(Q)", norm(Q)),
    ]

    cbprogress = ClimateMachine.GenericCallbacks.EveryXWallTimeSeconds(
        p.update_seconds,
        MPI.COMM_WORLD,
    ) do
        ProgressMeter.update!(
            progress[],
            round(Int, ClimateMachine.ODESolvers.gettime(odesolver));
            showvalues = showvalues,
        )
    end

    # The first part of cbprogress_setup() should be run at the end of the
    # first timestep, and the second part should be run before the call to
    # @sprintf in ClimateMachine.invoke!(). This does not need to be a
    # callback that is run on every timestep, but it's not too inefficient.
    function cbprogress_setup()
        if !isdefined(progress, 1)
            progress[] = ProgressMeter.Progress(
                round(Int, timeend);
                desc = "",   # Don't print anything before the progress bar.
                barlen = 80, # Set the bar to 80 characters on all terminals.
            )
        end
        if ClimateMachine.ODESolvers.gettime(odesolver) >= timeend
            ProgressMeter.update!(progress[], round(Int, timeend))
        end
    end

    return (cbprogress_setup, cbprogress)
end

function create_callback(output::JLD2State, simulation::Simulation, odesolver)
    # Initialize output
    output.overwrite &&
        isfile(output.filepath) &&
        rm(output.filepath; force = output.overwrite)

    Q = simulation.state
    mpicomm = MPI.COMM_WORLD
    iteration = output.iteration

    steps = ClimateMachine.ODESolvers.getsteps(odesolver)
    time = ClimateMachine.ODESolvers.gettime(odesolver)

    file = jldopen(output.filepath, "a+")
    JLD2.Group(file, "state")
    JLD2.Group(file, "time")
    file["state"][string(steps)] = Array(Q)
    file["time"][string(steps)] = time
    close(file)


    jldcallback = ClimateMachine.GenericCallbacks.EveryXSimulationSteps(
        iteration,
    ) do (s = false)
        steps = ClimateMachine.ODESolvers.getsteps(odesolver)
        time = ClimateMachine.ODESolvers.gettime(odesolver)
        @info steps, time
        file = jldopen(output.filepath, "a+")
        file["state"][string(steps)] = Array(Q)
        file["time"][string(steps)] = time
        close(file)
        return nothing
    end

    return jldcallback
end

function create_callback(output::VTKState, simulation::Simulation, odesolver)
    # Initialize output
    output.overwrite &&
        isfile(output.filepath) &&
        rm(output.filepath; force = output.overwrite)
    mkpath(output.filepath)

    state = simulation.state
    if simulation.rhs isa Tuple
        if simulation.rhs[1] isa AbstractRate 
            model = simulation.rhs[1].model
        else
            model = simulation.rhs[1]
        end
    else
        model = simulation.rhs
    end
    # model = (simulation.rhs isa Tuple) ? simulation.rhs[1] : simulation.rhs 

    function do_output(counter, model, state)
        mpicomm = MPI.COMM_WORLD
        balance_law = model.balance_law
        aux_state = model.state_auxiliary

        outprefix = @sprintf(
            "%s/mpirank%04d_step%04d",
            output.filepath,
            MPI.Comm_rank(mpicomm),
            counter[1],
        )

        @info "doing VTK output" outprefix

        state_names =
            flattenednames(vars_state(balance_law, Prognostic(), eltype(state)))
        aux_names =
            flattenednames(vars_state(balance_law, Auxiliary(), eltype(state)))

        writevtk(outprefix, state, model, state_names, aux_state, aux_names)

        counter[1] += 1

        return nothing
    end

    do_output(output.counter, model, state)
    cbvtk =
        ClimateMachine.GenericCallbacks.EveryXSimulationSteps(output.iteration) do (
            init = false
        )
            do_output(output.counter, model, state)
            return nothing
        end

    return cbvtk
end


function create_callback(output::NetCDF, simulation::Simulation, odesolver)
    # Initialize output
    output.overwrite &&
        isfile(output.filepath) &&
        rm(output.filepath; force = output.overwrite)
    mkpath(output.filepath)

    resolution = output.resolution
    if simulation.rhs isa Tuple
        if simulation.rhs[1] isa AbstractRate 
            model = simulation.rhs[1].model
        else
            model = simulation.rhs[1]
        end
    else
        model = simulation.rhs
    end
    parameters = model.balance_law.physics.parameters
    state = simulation.state
    aux = model.state_auxiliary

    # interpolate to lat lon height
    boundaries = [
        Float64(-90.0) Float64(-180.0) parameters.a
        Float64(90.0) Float64(180.0) Float64(parameters.a + parameters.H)
    ]
    axes = (
            collect(range(
                boundaries[1, 1],
                boundaries[2, 1],
                step = resolution[1],
            )),
            collect(range(
                boundaries[1, 2],
                boundaries[2, 2],
                step = resolution[2],
            )),
            collect(range(
                boundaries[1, 3],
                boundaries[2, 3],
                step = resolution[3],
            )),
        )

    vert_range = grid1d(
        parameters.a,
        parameters.a + parameters.H,
        simulation.grid.resolution.grid_stretching,
        nelem = simulation.grid.resolution.elements.vertical,
    )

    interpol = InterpolationCubedSphere(
        model.grid,
        vert_range,
        simulation.grid.resolution.elements.horizontal,
        axes[1],
        axes[2],
        axes[3];
    )

    dgngrp = setup_atmos_default_diagnostics(
        simulation,
        output.iteration,
        "TEST_GCM_Experiment",
        output.filepath,
        interpol = interpol,
    )
    
    netcdf_write = EveryXSimulationSteps(output.iteration) do
        # TODO: collection function in DiagnosticsGroup
        dgngrp.collect
    #    @warn "Entered NETCDF callback. Do nothing currently. Test" maxlog = 1
    end
    
    return netcdf_write

end

function create_callback(filter::TMARCallback, simulation::Simulation, odesolver)
    Q = simulation.state
    grid = simulation.grid.numerical
    tmar_filter = EveryXSimulationSteps(1) do
        Filters.apply!(Q, filter.filterstates, grid, TMARFilter())
        end
    return tmar_filter
end

# helper function 

function update_ref_state!(
    model::DryAtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    eos = model.physics.eos
    parameters = model.physics.parameters
    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    aux.ref_state.ρ = ρ
    aux.ref_state.ρu = ρu # @SVector[0.0,0.0,0.0]
    aux.ref_state.ρe = ρe

    aux.ref_state.p = calc_pressure(eos, state, aux, parameters)
end

function create_callback(update_ref::ReferenceStateUpdate, simulation::Simulation, odesolver)
    Q = simulation.state
    grid = simulation.grid.numerical
    step =  update_ref.recompute
    dg = simulation.rhs[2].model
    balance_law = dg.balance_law

    relinearize = EveryXSimulationSteps(step) do       
        t = gettime(odesolver)
        
        update_auxiliary_state!(update_ref_state!, dg, balance_law, Q, t)

        α = odesolver.dt * odesolver.RKA_implicit[2, 2]
        # hack
        be_solver = odesolver.implicit_solvers[odesolver.RKA_implicit[2, 2]][1]
        update_backward_Euler_solver!(be_solver, Q, α)
        nothing
    end
    return relinearize
end
