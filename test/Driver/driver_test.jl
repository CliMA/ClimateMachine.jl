using StaticArrays
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.PlanetParameters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

function init_test!(state, aux, (x,y,z), t)
    FT = eltype(state)

    z = FT(z)

    # These constants are those used by Stevens et al. (2005)
    qref       = FT(9.0e-3)
    q_pt_sfc   = PhasePartition(qref)
    Rm_sfc     = FT(gas_constant_air(q_pt_sfc))
    T_sfc      = FT(290.4)
    P_sfc      = FT(MSLP)

    # Specify moisture profiles
    q_liq      = FT(0)
    q_ice      = FT(0)

    θ_liq  = FT(289.0)
    q_tot  = qref

    ugeo = FT(7)
    vgeo = FT(-5.5)
    u, v, w = ugeo, vgeo, FT(0)

    # Pressure
    H     = Rm_sfc * T_sfc / grav
    p     = P_sfc * exp(-z / H)

    # Density, Temperature
    ts    = LiquidIcePotTempSHumEquil_given_pressure(θ_liq, p, q_tot)
    ρ     = air_density(ts)

    e_kin = FT(1/2) * FT((u^2 + v^2 + w^2))
    e_pot = grav * z
    E     = ρ * total_energy(e_kin, e_pot, ts)

    state.ρ               = ρ
    state.ρu              = SVector(ρ*u, ρ*v, ρ*w)
    state.ρe              = E
    state.moisture.ρq_tot = ρ * q_tot

    return nothing
end

function main()
    CLIMA.init()

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δh = FT(35)
    Δv = FT(5)
    resolution = (Δh, Δh, Δv)

    xmax = 100
    ymax = 100
    zmax = 250

    t0 = FT(0)
    timeend = FT(10)

    driver_config = CLIMA.LES_Configuration("Driver test", N, resolution,
                                            xmax, ymax, zmax, init_test!)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config)

    result = CLIMA.invoke!(solver_config)
end

main()

