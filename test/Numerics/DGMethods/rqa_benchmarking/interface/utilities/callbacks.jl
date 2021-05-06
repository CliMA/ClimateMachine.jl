abstract type AbstractCallback end

struct Default <: AbstractCallback end
struct Info <: AbstractCallback end
struct CFL <: AbstractCallback end
struct StateCheck{T} <: AbstractCallback
    number_of_checks::T
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
    model = (simulation.rhs isa Tuple) ? simulation.rhs[1] : simulation.rhs 

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
