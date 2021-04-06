using Dates
using FileIO
using MPI
using NCDatasets
using Printf
using Random
using StaticArrays
using Test

using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.Thermodynamics
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.Writers
using ClimateMachine.GenericCallbacks

using CLIMAParameters
using CLIMAParameters.Planet: grav, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

include("sin_init.jl")

function config_sin_test(FT, N, resolution, xmax, ymax, zmax)
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK54CarpenterKennedy,
    )
    config = ClimateMachine.AtmosLESConfiguration(
        "Diagnostics SIN test",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_sin_test!;
        solver_type = ode_solver,
    )

    return config
end

function config_diagnostics(driver_config)
    interval = "100steps"
    dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        interval,
        replace(driver_config.name, " " => "_"),
        writer = NetCDFWriter(),
    )
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()
    # Disable driver diagnostics as we're testing it here
    ClimateMachine.Settings.diagnostics = "never"

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δh = FT(50)
    Δv = FT(20)
    resolution = (Δh, Δh, Δv)

    xmax = FT(1500)
    ymax = FT(1500)
    zmax = FT(1500)

    t0 = FT(0)
    dt = FT(0.01)
    timeend = dt

    driver_config = config_sin_test(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
        init_on_cpu = true,
    )
    dgn_config = config_diagnostics(driver_config)

    mpicomm = solver_config.mpicomm
    dg = solver_config.dg
    Q = solver_config.Q
    solver = solver_config.solver

    outdir = mktempdir()
    currtime = ODESolvers.gettime(solver)
    starttime = replace(string(now()), ":" => ".")
    Diagnostics.init(mpicomm, param_set, dg, Q, starttime, outdir, false)
    GenericCallbacks.init!(
        dgn_config.groups[1],
        nothing,
        nothing,
        nothing,
        currtime,
    )

    ClimateMachine.invoke!(solver_config)

    # Check results
    mpirank = MPI.Comm_rank(mpicomm)
    if mpirank == 0
        dgngrp = dgn_config.groups[1]
        nm = @sprintf("%s_%s.nc", dgngrp.out_prefix, dgngrp.name)
        ds = NCDataset(joinpath(outdir, nm), "r")
        ds_u = ds["u"][:]
        ds_cov_w_u = ds["cov_w_u"][:]
        N = size(ds_u, 1)
        err = 0
        err1 = 0
        for i in 1:N
            u = ds_u[i]
            cov_w_u = ds_cov_w_u[i]
            err += (cov_w_u - 0.5)^2
            err1 += (u - 5)^2
        end
        close(ds)
        err = sqrt(err / N)
        @test err1 <= 1e-16
        @test err <= 2e-15
    end
end

main()
