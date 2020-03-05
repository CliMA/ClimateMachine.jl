using StaticArrays
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.PlanetParameters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

function init_test!(bl, state, aux, (x,y,z), t)
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
    Δh = FT(40)
    Δv = FT(40)
    resolution = (Δh, Δh, Δv)

    xmax = 320
    ymax = 320
    zmax = 400

    t0 = FT(0)
    timeend = FT(10)

    CFL = FT(0.4)

    driver_config = CLIMA.Atmos_LES_Configuration("Driver test", N, resolution,
                                                  xmax, ymax, zmax, init_test!)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config,
                                       Courant_number = CFL)


    # Test the courant wrapper
    CFL_nondiff = CLIMA.DGmethods.courant(CLIMA.Courant.nondiffusive_courant,
                                          solver_config)
    # Since the dt is computed before the initial condition, these might be
    # difference by a fairly large factor
    @test isapprox(CFL_nondiff, CFL, rtol=0.03)

    cb_test = 0
    result = CLIMA.invoke!(solver_config)
    # cb_test should be zero since user_info_callback not specified
    @test cb_test == 0

    result = CLIMA.invoke!(solver_config, user_info_callback=(init)->cb_test+=1)
    # cb_test should be greater than one if the user_info_callback got called
    @test cb_test > 0
end

main()

