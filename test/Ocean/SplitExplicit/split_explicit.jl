#!/usr/bin/env julia --project
using ClimateMachine
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.Ocean
using ClimateMachine.Ocean.HydrostaticBoussinesq
using ClimateMachine.Ocean.ShallowWater
using ClimateMachine.Ocean.SplitExplicit: VerticalIntegralModel
using ClimateMachine.Ocean.OceanProblems

using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.BalanceLaws: vars_state, Prognostic, Auxiliary
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.VTK
using ClimateMachine.Checkpoint

using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

struct SplitConfig{N, M3, M2, G3, G2, M, AT}
    name::N
    model_3D::M3
    model_2D::M2
    grid_3D::G3
    grid_2D::G2
    mpicomm::M
    ArrayType::AT
end

function run_split_explicit(
    config::SplitConfig,
    timespan;
    dt_fast = 300,
    dt_slow = 300,
    refDat = (),
    restart = 0,
    analytic_solution = false,
)
    tout, timeend = timespan

    nout = ceil(Int64, tout / dt_slow)
    dt_slow = tout / nout

    vert_filter =
        CutoffFilter(config.grid_3D, polynomialorder(config.grid_3D) - 1)
    exp_filter = ExponentialFilter(config.grid_3D, 1, 8)

    integral_model = DGModel(
        VerticalIntegralModel(config.model_3D),
        config.grid_3D,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    modeldata = (
        vert_filter = vert_filter,
        exp_filter = exp_filter,
        integral_model = integral_model,
    )

    if restart > 0
        Q_3D, A_3D, t0 = read_checkpoint(
            config.name,
            "baroclinic",
            config.ArrayType,
            config.mpicomm,
            restart,
        )
        Q_2D, A_2D, _ = read_checkpoint(
            config.name,
            "barotropic",
            config.ArrayType,
            config.mpicomm,
            restart,
        )

        direction = EveryDirection()
        A_3D = restart_auxiliary_state(
            config.model_3D,
            config.grid_3D,
            A_3D,
            direction,
        )
        A_2D = restart_auxiliary_state(
            config.model_2D,
            config.grid_2D,
            A_2D,
            direction,
        )

        dg_3D = DGModel(
            config.model_3D,
            config.grid_3D,
            RusanovNumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient();
            state_auxiliary = A_3D,
            modeldata = modeldata,
        )
        dg_2D = DGModel(
            config.model_2D,
            config.grid_2D,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            state_auxiliary = A_2D,
        )

        Q_3D = restart_ode_state(dg_3D, Q_3D; init_on_cpu = true)
        Q_2D = restart_ode_state(dg_2D, Q_2D; init_on_cpu = true)

        lsrk_3D = LSRK54CarpenterKennedy(dg_3D, Q_3D, dt = dt_slow, t0 = t0)
        lsrk_2D = LSRK54CarpenterKennedy(dg_2D, Q_2D, dt = dt_fast, t0 = t0)

        timeendlocal = timeend + t0
    else
        dg_3D = DGModel(
            config.model_3D,
            config.grid_3D,
            RusanovNumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient();
            modeldata = modeldata,
        )

        dg_2D = DGModel(
            config.model_2D,
            config.grid_2D,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
        )

        Q_3D = init_ode_state(dg_3D, FT(0); init_on_cpu = true)
        Q_2D = init_ode_state(dg_2D, FT(0); init_on_cpu = true)

        lsrk_3D = LSRK54CarpenterKennedy(dg_3D, Q_3D, dt = dt_slow, t0 = 0)
        lsrk_2D = LSRK54CarpenterKennedy(dg_2D, Q_2D, dt = dt_fast, t0 = 0)

        timeendlocal = timeend
    end

    odesolver = SplitExplicitSolver(lsrk_3D, lsrk_2D;)

    vtkstep = [restart, restart, restart, restart]
    cbvector = make_callbacks(
        config.name,
        vtkstep,
        nout,
        config.mpicomm,
        odesolver,
        dg_3D,
        config.model_3D,
        Q_3D,
        dg_2D,
        config.model_2D,
        Q_2D,
        timeendlocal,
    )

    eng0 = norm(Q_3D)
    @info @sprintf """Starting
    norm(Q₀) = %.16e
    ArrayType = %s""" eng0 config.ArrayType

    # slow fast state tuple
    Qvec = (slow = Q_3D, fast = Q_2D)
    solve!(Qvec, odesolver; timeend = timeendlocal, callbacks = cbvector)

    if analytic_solution
        Qe_3D = init_ode_state(dg_3D, timeendlocal, init_on_cpu = true)
        Qe_2D = init_ode_state(dg_2D, timeendlocal, init_on_cpu = true)

        error_3D = euclidean_distance(Q_3D, Qe_3D) / norm(Qe_3D)
        error_2D = euclidean_distance(Q_2D, Qe_2D) / norm(Qe_2D)

        println("3D error = ", error_3D)
        println("2D error = ", error_2D)
        @test isapprox(error_3D, FT(0.0); atol = 0.005)
        @test isapprox(error_2D, FT(0.0); atol = 0.005)
    end

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
    nout,
    mpicomm,
    odesolver,
    dg_slow,
    model_slow,
    Q_slow,
    dg_fast,
    model_fast,
    Q_fast,
    timeend,
)
    if isdir(vtkpath)
        rm(vtkpath, recursive = true)
    end
    mkpath(vtkpath)
    mkpath(vtkpath * "/slow")
    mkpath(vtkpath * "/fast")

    A_slow = dg_slow.state_auxiliary
    A_fast = dg_fast.state_auxiliary

    function do_output(span, vtkstep, model, dg, Q, A)
        outprefix = @sprintf(
            "%s/%s/mpirank%04d_step%04d",
            vtkpath,
            span,
            MPI.Comm_rank(mpicomm),
            vtkstep
        )
        @info "doing VTK output" outprefix
        statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
        auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
        writevtk(outprefix, Q, dg, statenames, A, auxnames)
    end

    do_output("slow", vtkstep[1], model_slow, dg_slow, Q_slow, A_slow)
    cbvtk_slow = GenericCallbacks.EveryXSimulationSteps(nout) do (init = false)
        do_output("slow", vtkstep[1], model_slow, dg_slow, Q_slow, A_slow)
        vtkstep[1] += 1
        nothing
    end

    do_output("fast", vtkstep[2], model_fast, dg_fast, Q_fast, A_fast)
    cbvtk_fast = GenericCallbacks.EveryXSimulationSteps(nout) do (init = false)
        do_output("fast", vtkstep[2], model_fast, dg_fast, Q_fast, A_fast)
        vtkstep[2] += 1
        nothing
    end

    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q_slow)
            @info @sprintf(
                """Update
                simtime = %8.2f / %8.2f
                runtime = %s
                norm(Q) = %.16e""",
                ODESolvers.gettime(odesolver),
                timeend,
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end

    cbcs_dg = ClimateMachine.StateCheck.sccreate(
        [
            (Q_slow, "3D state"),
            (A_slow, "3D aux"),
            (Q_fast, "2D state"),
            (A_fast, "2D aux"),
        ],
        nout;
        prec = 12,
    )

    cb_checkpoint = GenericCallbacks.EveryXSimulationSteps(nout) do
        write_checkpoint(
            Q_slow,
            A_slow,
            odesolver,
            vtkpath,
            "baroclinic",
            mpicomm,
            vtkstep[3],
        )

        write_checkpoint(
            Q_fast,
            A_fast,
            odesolver,
            vtkpath,
            "barotropic",
            mpicomm,
            vtkstep[4],
        )

        rm_checkpoint(vtkpath, "baroclinic", mpicomm, vtkstep[3] - 1)

        rm_checkpoint(vtkpath, "barotropic", mpicomm, vtkstep[4] - 1)

        vtkstep[3] += 1
        vtkstep[4] += 1
        nothing
    end

    return (cbvtk_slow, cbvtk_fast, cbinfo, cb_checkpoint, cbcs_dg)
end
