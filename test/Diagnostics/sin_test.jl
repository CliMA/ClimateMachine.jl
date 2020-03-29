using Dates
using FileIO
using MPI
using Random
using StaticArrays
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.Diagnostics
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.ODESolvers
using CLIMA.PlanetParameters: grav, MSLP
using CLIMA.VariableTemplates

function init_sin_test!(bl, state, aux, (x, y, z), t)
    FT = eltype(state)

    z = FT(z)

    # These constants are those used by Stevens et al. (2005)
    qref = FT(9.0e-3)
    q_pt_sfc = PhasePartition(qref)
    Rm_sfc = FT(gas_constant_air(q_pt_sfc))
    T_sfc = FT(292.5)
    P_sfc = FT(MSLP)

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

    u1, u2 = FT(6), FT(7)
    v1, v2 = FT(-4.25), FT(-5.5)
    w = FT(10 + 0.5 * sin(2 * π * ((x / 1500) + (y / 1500))))
    u = (5 + 2 * sin(2 * π * ((x / 1500) + (y / 1500))))
    v = FT(5 + 2 * sin(2 * π * ((x / 1500) + (y / 1500))))

    # Pressure
    H = Rm_sfc * T_sfc / grav
    p = P_sfc * exp(-z / H)

    # Density, Temperature
    ts = LiquidIcePotTempSHumEquil_given_pressure(θ_liq, p, q_tot, bl.param_set)
    ρ = air_density(ts)

    e_kin = FT(1 / 2) * FT((u^2 + v^2 + w^2))
    e_pot = grav * z
    E = ρ * total_energy(e_kin, e_pot, ts)

    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, ρ * w)
    state.ρe = E
    state.moisture.ρq_tot = ρ * q_tot
end

function config_sin_test(FT, N, resolution, xmax, ymax, zmax)
    ode_solver =
        CLIMA.ExplicitSolverType(solver_method = LSRK54CarpenterKennedy)
    config = CLIMA.AtmosLESConfiguration(
        "Diagnostics SIN test",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        init_sin_test!,
        solver_type = ode_solver,
    )

    return config
end

function config_diagnostics(driver_config)
    interval = 100
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return CLIMA.DiagnosticsConfiguration([dgngrp])
end

function main()
    CLIMA.init()

    # Disable driver diagnostics as we're testing it here
    CLIMA.Settings.enable_diagnostics = false

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
    solver_config = CLIMA.SolverConfiguration(
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

    CLIMA.invoke!(solver_config)

    # Check results
    mpirank = MPI.Comm_rank(mpicomm)
    if mpirank == 0
        d = load(joinpath(
            outdir,
            "$(dgn_config.groups[1].out_prefix)-$(dgn_config.groups[1].name)-$(starttime).jld2",
        ))
        Nqk = size(d["0.0"], 1)
        Nev = size(d["0.0"], 2)
        S = zeros(Nqk * Nev)
        S1 = zeros(Nqk * Nev)
        err = 0
        err1 = 0
        for ev in 1:Nev
            for k in 1:Nqk
                dv = Diagnostics.diagnostic_vars(d["0.0"][k, ev])
                S[k + (ev - 1) * Nqk] = dv.vert_eddy_u_flux
                S1[k + (ev - 1) * Nqk] = dv.u
                err += (S[k + (ev - 1) * Nqk] - 0.5)^2
                err1 += (S1[k + (ev - 1) * Nqk] - 5)^2
            end
        end
        err = sqrt(err / (Nqk * Nev))
        @test err <= 2e-15
        @test err1 <= 1e-16
    end
end
main()
