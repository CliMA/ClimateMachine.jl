using Test

using ClimateMachine.SurfaceFluxes
using ClimateMachine.SurfaceFluxes.Nishizawa2018
using ClimateMachine.SurfaceFluxes.Byun1990
using ClimateMachine.MoistThermodynamics
using RootSolvers
using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# FIXME: Use realistic values / test for correctness
# These tests have been run to ensure they do not fail,
# but they need further testing for correctness.

FT = Float32
rtol = 10 * eps(FT)

@testset "SurfaceFluxes" begin
    shf, lhf, T_b, q_pt, α_0 =
        FT(60), FT(50), FT(350), PhasePartition{FT}(0.01, 0.002, 0.0001), FT(1)
    buoyancy_flux =
        SurfaceFluxes.compute_buoyancy_flux(param_set, shf, lhf, T_b, q_pt, α_0)
    @test buoyancy_flux ≈ 0.0017808608107074118
    @test buoyancy_flux isa FT
end

@testset "SurfaceFluxes.Byun1990" begin
    u, flux = FT(0.1), FT(0.2)
    MO_len = Byun1990.monin_obukhov_len(param_set, u, flux)
    @test MO_len ≈ -0.0125

    u_ave, buoyancy_flux, z_0, z_1 = FT(0.1), FT(0.2), FT(2), FT(5)
    γ_m, β_m = FT(15), FT(4.8)
    tol_abs, iter_max = FT(1e-3), 10
    u_star = Byun1990.compute_friction_velocity(
        param_set,
        u_ave,
        buoyancy_flux,
        z_0,
        z_1,
        β_m,
        γ_m,
        tol_abs,
        iter_max,
    )
    @test u_star ≈ 0.201347256193615 atol = tol_abs
    @test u_star isa FT


    Ri, z_b, z_0, Pr_0 = FT(10), FT(2), FT(5.0), FT(0.74)
    γ_m, γ_h, β_m, β_h = FT(15), FT(9), FT(4.8), FT(7.8)
    cm, ch, L_mo = Byun1990.compute_exchange_coefficients(
        param_set,
        Ri,
        z_b,
        z_0,
        γ_m,
        γ_h,
        β_m,
        β_h,
        Pr_0,
    )
    @test cm ≈ 19.700348427787368
    @test ch ≈ 3.3362564728997803
    @test L_mo ≈ -14.308268023583906
    @test cm isa FT
    @test ch isa FT
    @test L_mo isa FT

    Ri, z_b, z_0, Pr_0 = FT(-10), FT(10.0), FT(1.0), FT(0.74)
    γ_m, γ_h, β_m, β_h = FT(15.0), FT(9.0), FT(4.8), FT(7.8)
    cm, ch, L_mo = Byun1990.compute_exchange_coefficients(
        param_set,
        Ri,
        z_b,
        z_0,
        γ_m,
        γ_h,
        β_m,
        β_h,
        Pr_0,
    )
    @test cm ≈ 0.33300280321092746 rtol = rtol
    @test ch ≈ 1.131830939627489 rtol = rtol
    @test L_mo ≈ -0.3726237964444814 rtol = rtol
    @test cm isa FT
    @test ch isa FT
    @test L_mo isa FT

end

@testset "SurfaceFluxes.Nishizawa2018" begin
    FT = Float32
    rtol = 10 * eps(FT)
    u, θ, flux = FT(2), FT(350), FT(20)
    MO_len = Nishizawa2018.monin_obukhov_len(param_set, u, θ, flux)
    @test MO_len ≈ -35.67787971457696

    u_ave, θ, flux, Δz, z_0, a =
        FT(110.0), FT(350.0), FT(20.0), FT(100.0), FT(0.01), FT(5.0)
    Ψ_m_tol, tol_abs, iter_max = FT(1e-3), FT(1e-3), 10
    u_star = Nishizawa2018.compute_friction_velocity(
        param_set,
        u_ave,
        θ,
        flux,
        Δz,
        z_0,
        a,
        Ψ_m_tol,
        tol_abs,
        iter_max,
    )
    @test u_star ≈ 5.526644550864822 atol = tol_abs
    @test u_star isa FT

    FT = Float64
    rtol = 10 * eps(FT)
    z, F_m, F_h, a, u_star, θ, flux, Pr =
        FT(1), FT(2), FT(3), FT(5), FT(110), FT(350), FT(20), FT(0.74)

    K_m, K_h, L_mo = Nishizawa2018.compute_exchange_coefficients(
        param_set,
        z,
        F_m,
        F_h,
        a,
        u_star,
        θ,
        flux,
        Pr,
    )
    @test K_m ≈ -11512.071612359368 rtol = rtol
    @test K_h ≈ -6111.6196776263805 rtol = rtol
    @test L_mo ≈ -5.935907237512742e6 rtol = rtol

    # Test type-stability:
    FT = Float32
    rtol = 10 * eps(FT)
    z, F_m, F_h, a, u_star, θ, flux, Pr =
        FT(1), FT(2), FT(3), FT(5), FT(110), FT(350), FT(20), FT(0.74)

    K_m, K_h, L_mo = Nishizawa2018.compute_exchange_coefficients(
        param_set,
        z,
        F_m,
        F_h,
        a,
        u_star,
        θ,
        flux,
        Pr,
    )
    @test K_m isa FT
    @test K_h isa FT
    @test L_mo isa FT

end
