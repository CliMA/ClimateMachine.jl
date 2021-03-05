
using Test
using JLD2
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates

using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.BalanceLaws:
    vars_state, Prognostic, Auxiliary, number_states
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.VTK

using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

struct Config{N, D, O, M, AT}
    name::N
    dg::D
    Nover::O
    mpicomm::M
    ArrayType::AT
end

function run_CNSE(
    config,
    resolution,
    timespan;
    TimeStepper = LSRK54CarpenterKennedy,
    refDat = (),
)
    dg = config.dg
    Q = init_ode_state(dg, FT(0); init_on_cpu = true)

    if config.Nover > 0
        cutoff = CutoffFilter(dg.grid, resolution.N + 1)
        num_state_prognostic = number_states(dg.balance_law, Prognostic())
        Filters.apply!(Q, 1:num_state_prognostic, dg.grid, cutoff)
    end

    function custom_tendency(tendency, x...; kw...)
        dg(tendency, x...; kw...)
        if config.Nover > 0
            cutoff = CutoffFilter(dg.grid, resolution.N + 1)
            num_state_prognostic = number_states(dg.balance_law, Prognostic())
            Filters.apply!(tendency, 1:num_state_prognostic, dg.grid, cutoff)
        end
    end

    println("time step is " * string(timespan.dt))
    println("time end is " * string(timespan.timeend))
    odesolver = TimeStepper(custom_tendency, Q, dt = timespan.dt, t0 = 0)

    vtkstep = 0
    cbvector = make_callbacks(
        vtkpath,
        vtkstep,
        timespan,
        config.mpicomm,
        odesolver,
        dg,
        dg.balance_law,
        Q,
        filename = config.name,
    )

    eng0 = norm(Q)
    @info @sprintf """Starting
    norm(Q₀) = %.16e
    ArrayType = %s""" eng0 config.ArrayType

    solve!(Q, odesolver; timeend = timespan.timeend, callbacks = cbvector)

    ## Check results against reference
    ClimateMachine.StateCheck.scprintref(cbvector[end])
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cbvector[end], refDat)
    end

    return nothing
end

function make_callbacks(
    vtkpath,
    vtkstep,
    timespan,
    mpicomm,
    odesolver,
    dg,
    model,
    Q;
    filename = " ",
)
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
                simtime = %8.2f / %8.2f
                runtime = %s
                norm(Q) = %.16e""",
                ODESolvers.gettime(odesolver),
                timespan.timeend,
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

    cbvector = (cbinfo,)

    if !isnothing(vtkpath)

        if isdir(vtkpath)
            rm(vtkpath, recursive = true)
        end
        mkpath(vtkpath)

        """
        file = jldopen(vtkpath * "/" * filename * ".jld2", "w")
        file["grid"] = dg.grid
        close(file)
        """

        function do_output(vtkstep, model, dg, Q)
            outprefix = @sprintf(
                "%s/mpirank%04d_step%04d",
                vtkpath,
                MPI.Comm_rank(mpicomm),
                vtkstep
            )
            @info "doing VTK output" outprefix
            statenames =
                flattenednames(vars_state(model, Prognostic(), eltype(Q)))
            auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
            writevtk(outprefix, Q, dg, statenames, dg.state_auxiliary, auxnames)

            """
            @info "doing JLD2 output" vtkstep
            file = jldopen(vtkpath * "/" * filename * ".jld2", "a+")
            file[string(vtkstep)] = Array(Q.realdata)
            close(file)
            """

            vtkstep += 1

            return vtkstep
        end

        vtkstep = do_output(vtkstep, model, dg, Q)
        cbvtk =
            ClimateMachine.GenericCallbacks.EveryXSimulationSteps(timespan.nout) do (
                init = false
            )
                vtkstep = do_output(vtkstep, model, dg, Q)
                return nothing
            end
        cbvector = (cbvector..., cbvtk)
    end

    cbcs_dg = ClimateMachine.StateCheck.sccreate(
        [(Q, "state")],
        timespan.nout;
        prec = 12,
    )

    cbvector = (cbvector..., cbcs_dg)

    return cbvector
end
