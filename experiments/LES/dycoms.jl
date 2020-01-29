using Distributions
using Random
using StaticArrays
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

"""
  Initial Condition for DYCOMS_RF01 LES
@article{doi:10.1175/MWR2930.1,
author = {Stevens, Bjorn and Moeng, Chin-Hoh and Ackerman,
          Andrew S. and Bretherton, Christopher S. and Chlond,
          Andreas and de Roode, Stephan and Edwards, James and Golaz,
          Jean-Christophe and Jiang, Hongli and Khairoutdinov,
          Marat and Kirkpatrick, Michael P. and Lewellen, David C. and Lock, Adrian and
          Maeller, Frank and Stevens, David E. and Whelan, Eoin and Zhu, Ping},
title = {Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus},
journal = {Monthly Weather Review},
volume = {133},
number = {6},
pages = {1443-1462},
year = {2005},
doi = {10.1175/MWR2930.1},
URL = {https://doi.org/10.1175/MWR2930.1},
eprint = {https://doi.org/10.1175/MWR2930.1}
}
"""
function init_dycoms!(state, aux, (x,y,z), t)
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
    zb         = FT(600)         # initial cloud bottom
    zi         = FT(840)         # initial cloud top

    if z <= zi
        θ_liq  = FT(289.0)
        q_tot  = qref
    else
        θ_liq  = FT(297.0) + (z - zi)^(FT(1/3))
        q_tot  = FT(1.5e-3)
    end

    ugeo = FT(7)
    vgeo = FT(-5.5)
    u, v, w = ugeo, vgeo, FT(0)

    # Perturb initial state to break symmetry and trigger turbulent convection
    r1 = FT(rand(Uniform(-0.002, 0.002)))
    r2 = FT(rand(Uniform(-0.00001, 0.00001)))
    r3 = FT(rand(Uniform(-0.001, 0.001)))
    r4 = FT(rand(Uniform(-0.001, 0.001)))
    if z <= 400.0
        θ_liq += r1 * θ_liq
        q_tot += r2 * q_tot
        u     += r3 * u
        v     += r4 * v
    end

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

function config_dycoms(FT, N, resolution, xmax, ymax, zmax)
    # Reference state
    T_min   = FT(289)
    T_s     = FT(290.4)
    Γ_lapse = FT(grav/cp_d)
    T       = LinearTemperatureProfile(T_min, T_s, Γ_lapse)
    rel_hum = FT(0)
    ref_state = HydrostaticState(T, rel_hum)

    # Radiation model
    κ             = FT(85)
    α_z           = FT(1)
    z_i           = FT(840)
    ρ_i           = FT(1.13)
    D_subsidence  = FT(0) # 0 for stable testing, 3.75e-6 in practice
    F_0           = FT(70)
    F_1           = FT(22)
    radiation = DYCOMSRadiation{FT}(κ, α_z, z_i, ρ_i, D_subsidence, F_0, F_1)

    # Sources
    f_coriolis    = FT(1.03e-4)
    u_geostrophic = FT(7.0)
    v_geostrophic = FT(-5.5)
    w_ref         = FT(0)
    u_relaxation  = SVector(u_geostrophic, v_geostrophic, w_ref)
    # Sponge
    c_sponge = 1
    # Rayleigh damping
    zsponge = FT(1500.0)
    rayleigh_sponge = RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2)
    # Geostrophic forcing
    geostrophic_forcing = GeostrophicForcing{FT}(f_coriolis, u_geostrophic, v_geostrophic)

    # Boundary conditions
    # SGS Filter constants
    C_smag = FT(0.21) # 0.21 for stable testing, 0.18 in practice
    C_drag = FT(0.0011)
    LHF    = FT(115)
    SHF    = FT(15)
    bc = DYCOMS_BC{FT}(C_drag, LHF, SHF)

    config = CLIMA.LES_Configuration("DYCOMS", N, resolution, xmax, ymax, zmax,
                                     init_dycoms!,
                                     solver_type=CLIMA.ExplicitSolverType(LSRK144NiegemannDiehlBusch),
                                     ref_state=ref_state,
                                     C_smag=C_smag,
                                     moisture=EquilMoist(5),
                                     radiation=radiation,
                                     subsidence=ConstantSubsidence{FT}(D_subsidence),
                                     sources=(Gravity(),
                                              rayleigh_sponge,
                                              geostrophic_forcing),
                                     bc=bc)

    return config
end

function main()
    CLIMA.init()

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δh = FT(40)
    Δv = FT(20)
    resolution = (Δh, Δh, Δv)

    xmax = 1000
    ymax = 1000
    zmax = 2500

    t0 = FT(0)
    timeend = FT(100)

    driver_config = config_dycoms(FT, N, resolution, xmax, ymax, zmax)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config, forcecpu=true)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(2) do (init=false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    result = CLIMA.invoke!(solver_config;
                          user_callbacks=(cbtmarfilter,),
                          check_euclidean_distance=true)

    @testset begin
        @test result ≈ FT(0.9999734954176608)
    end
end

main()
