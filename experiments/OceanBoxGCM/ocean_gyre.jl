#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.cli()

using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.HydrostaticBoussinesq
using ClimateMachine.GenericCallbacks: EveryXSimulationSteps
using Plots

using Test

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

include("ocean_init_state.jl")

function CLIMA_plot(solver_config, filename)
    Qnd=reshape(solver_config.Q.realdata,(5,5,5,4,20,20,20));
    Gnd=reshape(solver_config.dg.grid.vgeo,(5,5,5,16,20,20,20));
    tval=Qnd[1,1,:,4,:,1,1][:];
    zval=Gnd[1,1,:,15,:,1,1][:]
    tval_cpu=ones( size(tval) )
    copyto!(tval_cpu, tval)
    typeof(tval_cpu)
    zval_cpu=ones(size(zval))
    copyto!(zval_cpu, zval)
    savefig(scatter(tval_cpu,zval_cpu;label="",xlabel="u[1] (m/s)",ylabel="z (m)"),filename) 
    return nothing
end

function config_simple_box(FT, N, resolution, dimensions; BC = nothing)
    if BC == nothing
        problem = OceanGyre{FT}(dimensions...)
    else
        problem = OceanGyre{FT}(dimensions...; BC = BC)
    end

    _grav::FT = grav(param_set)
    cʰ = sqrt(_grav * problem.H) # m/s
    model = HydrostaticBoussinesqModel{FT}(param_set, problem, cʰ = 1, αᵀ = 0)

    config = ClimateMachine.OceanBoxGCMConfiguration(
        "ocean_gyre",
        N,
        resolution,
        model,
    )

    return config
end

function run_ocean_gyre(; imex::Bool = false, BC = nothing)
    FT = Float64

    # DG polynomial order
    N = Int(4)

    # Domain resolution and size
    Nˣ = Int(20)
    Nʸ = Int(20)
    Nᶻ = Int(20)
    resolution = (Nˣ, Nʸ, Nᶻ)

    Lˣ = 4e6    # m
    Lʸ = 4e6    # m
    H = 1000   # m
    dimensions = (Lˣ, Lʸ, H)

    timestart = FT(0)    # s
    timeout = FT(0.25 * 86400) # s
    #timeend = FT(86400) # s
    timeend = FT(1000000) # s
    value = 1

    dt = FT(100)    # s

    if imex
        solver_type =
            ClimateMachine.IMEXSolverType(ClimateMachine.OceanBoxGCMSpecificInfo)
    else
        solver_type = ClimateMachine.ExplicitSolverType(
            solver_method = LSRK144NiegemannDiehlBusch,
        )
    end
    println(solver_type)

    driver_config = config_simple_box(FT, N, resolution, dimensions; BC = BC)

    grid = driver_config.grid
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1)
    exp_filter = ExponentialFilter(grid, 1, 8)
    modeldata = (vert_filter = vert_filter, exp_filter = exp_filter)

    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        init_on_cpu = true,
        ode_dt = dt,
        Courant_number = 0.4,
        ode_solver_type = solver_type,
        modeldata = modeldata,
    )

    ClimateMachine.Settings.vtk = "never"
    # vtk_interval = ceil(Int64, timeout / solver_config.dt)
    # ClimateMachine.Settings.vtk = "$(vtk_interval)steps"

    ClimateMachine.Settings.diagnostics = "never"
    # diagnostics_interval = ceil(Int64, timeout / solver_config.dt)
    # ClimateMachine.Settings.diagnostics = "$(diagnostics_interval)steps"

    plot_callback = EveryXSimulationSteps(1000) do
        outfile = "out"*lpad(value,5,"0")*".png"
        println("output filename is: ", outfile)
        CLIMA_plot(solver_config, outfile)
        value += 1
        return nothing
    end

    callbacks = [plot_callback]

    #result = ClimateMachine.invoke!(solver_config;)
    result = ClimateMachine.invoke!(solver_config;user_callbacks=callbacks)

    @test true

end

@testset "$(@__FILE__)" begin
    boundary_conditions = [
        (
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Penetrable(KinematicStress()), TemperatureFlux()),
        ),
    ]
#=
        (
            OceanBC(Impenetrable(FreeSlip()), Insulating()),
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Penetrable(KinematicStress()), TemperatureFlux()),
        ),
        (
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Impenetrable(FreeSlip()), Insulating()),
            OceanBC(Penetrable(KinematicStress()), TemperatureFlux()),
        ),
        (
            OceanBC(Impenetrable(FreeSlip()), Insulating()),
            OceanBC(Impenetrable(FreeSlip()), Insulating()),
            OceanBC(Penetrable(KinematicStress()), TemperatureFlux()),
        ),
        (
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Penetrable(FreeSlip()), TemperatureFlux()),
        ),
        (
            OceanBC(Impenetrable(FreeSlip()), Insulating()),
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Penetrable(FreeSlip()), TemperatureFlux()),
        ),
        (
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Impenetrable(FreeSlip()), Insulating()),
            OceanBC(Penetrable(FreeSlip()), TemperatureFlux()),
        ),
        (
            OceanBC(Impenetrable(FreeSlip()), Insulating()),
            OceanBC(Impenetrable(FreeSlip()), Insulating()),
            OceanBC(Penetrable(FreeSlip()), TemperatureFlux()),
        ),
    ]
=#

    for BC in boundary_conditions
        run_ocean_gyre(imex = true, BC = BC)
        #run_ocean_gyre(imex = false, BC = BC)
    end
end
