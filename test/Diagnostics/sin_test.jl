using Dates
using FileIO
using MPI
using Printf
using Random
using StaticArrays
using Test

using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.MoistThermodynamics
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.Writers

using CLIMAParameters
using CLIMAParameters.Planet: grav, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_sin_test!(bl, state, aux, (x, y, z), t)
    FT = eltype(state)

    z = FT(z)
    _grav::FT = grav(bl.param_set)
    _MSLP::FT = MSLP(bl.param_set)

    # These constants are those used by Stevens et al. (2005)
    qref = FT(9.0e-3)
    q_pt_sfc = PhasePartition(qref)
    Rm_sfc = FT(gas_constant_air(param_set, q_pt_sfc))
    T_sfc = FT(292.5)
    P_sfc = _MSLP

    # Specify moisture profiles
    q_liq = FT(0)
    q_ice = FT(0)
    zb = FT(600)         # initial cloud bottom
    zi = FT(840)         # initial cloud top
    dz_cloud = zi - zb
    q_liq_peak = FT(0.00045)     # cloud mixing ratio at z_i

    if z > zb && z <= zi
        q_liq = (z - zb) * q_liq_peak / dz_cloud
    end

    if z <= zi
        θ_liq = FT(289.0)
        q_tot = qref
    else
        θ_liq = FT(297.5) + (z - zi)^(FT(1 / 3))
        q_tot = FT(1.5e-3)
    end

    w = FT(10 + 0.5 * sin(2 * π * ((x / 1500) + (y / 1500))))
    u = (5 + 2 * sin(2 * π * ((x / 1500) + (y / 1500))))
    v = FT(5 + 2 * sin(2 * π * ((x / 1500) + (y / 1500))))

    # Pressure
    H = Rm_sfc * T_sfc / _grav
    p = P_sfc * exp(-z / H)

    # Density, Temperature
    ts = LiquidIcePotTempSHumEquil_given_pressure(bl.param_set, θ_liq, p, q_tot)
    #ρ = air_density(ts)
    ρ = one(FT)

    e_kin = FT(1 / 2) * FT((u^2 + v^2 + w^2))
    e_pot = _grav * z
    E = ρ * total_energy(e_kin, e_pot, ts)

    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, ρ * w)
    state.ρe = E
    state.moisture.ρq_tot = ρ * q_tot
end

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
        interval,
        replace(driver_config.name, " " => "_"),
        writer = JLD2Writer(),
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
    Diagnostics.init(mpicomm, dg, Q, starttime, outdir)
    dgn_config.groups[1](currtime, init = true)
    dgn_config.groups[1](currtime)

    ClimateMachine.invoke!(solver_config)

    # Check results
    mpirank = MPI.Comm_rank(mpicomm)
    if mpirank == 0
        dgngrp = dgn_config.groups[1]
        nm = @sprintf(
            "%s_%s_%s_num%04d.jld2",
            replace(dgngrp.out_prefix, " " => "_"),
            dgngrp.name,
            starttime,
            1,
        )
        ds = load(joinpath(outdir, nm))
        N = size(ds["u"], 1)
        err = 0
        err1 = 0
        for i in 1:N
            u = ds["u"][i]
            cov_w_u = ds["cov_w_u"][i]
            err += (cov_w_u - 0.5)^2
            err1 += (u - 5)^2
        end
        err = sqrt(err / N)
        @test err1 <= 1e-16
        @test err <= 2e-15
    end
end

main()
