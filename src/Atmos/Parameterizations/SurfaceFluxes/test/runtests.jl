using Test

using CLIMA.SurfaceFluxes
using CLIMA.SurfaceFluxes.Nishizawa2018
using CLIMA.SurfaceFluxes.Byun1990
using CLIMA.MoistThermodynamics
using CLIMA.RootSolvers

# FIXME: Use realistic values / test for correctness
# These tests have been run to ensure they do not fail,
# but they need further testing for correctness.

@testset "SurfaceFluxes" begin

  shf, lhf, T_b, qt_b, ql_b, qi_b, alpha0_0 = 60.0, 50.0, 350.0, 0.01, 0.002, 0.0001, 1.0
  buoyancy_flux = SurfaceFluxes.compute_buoyancy_flux(shf,
                                                 lhf,
                                                 T_b,
                                                 qt_b,
                                                 ql_b,
                                                 qi_b,
                                                 alpha0_0
                                                 )
  @test buoyancy_flux ≈ 0.0017808608107074118
end

@testset "SurfaceFluxes.Byun1990" begin
  u, flux = 0.1, 0.2
  MO_len = Byun1990.compute_MO_len(u, flux)
  @test MO_len ≈ -0.0125

  u_ave, buoyancy_flux, z_0, z_1 = 0.1, 0.2, 2.0, 5.0
  γ_m, β_m = 15.0, 4.8
  tol_abs, iter_max = 1e-3, 10
  u_star = Byun1990.compute_friction_velocity(u_ave,
                                              buoyancy_flux,
                                              z_0,
                                              z_1,
                                              β_m,
                                              γ_m,
                                              tol_abs,
                                              iter_max
                                              )
  @test u_star ≈ 0.201347256193615


  Ri, z_b, z_0, Pr_0 = 10, 2.0, 5.0, 0.74
  γ_m, γ_h, β_m, β_h = 15.0, 9.0, 4.8, 7.8
  cm, ch, L_mo = Byun1990.compute_exchange_coefficients(Ri, z_b, z_0, γ_m, γ_h, β_m, β_h, Pr_0)
  @test cm ≈ 19.700348427787368
  @test ch ≈ 3.3362564728997803
  @test L_mo ≈ -14.308268023583906

  Ri, z_b, z_0, Pr_0 = -10, 10.0, 1.0, 0.74
  γ_m, γ_h, β_m, β_h = 15.0, 9.0, 4.8, 7.8
  cm, ch, L_mo = Byun1990.compute_exchange_coefficients(Ri, z_b, z_0, γ_m, γ_h, β_m, β_h, Pr_0)
  @test cm ≈ 0.33300280321092746
  @test ch ≈ 1.131830939627489
  @test L_mo ≈ -0.3726237964444814

end

@testset "SurfaceFluxes.Nishizawa2018" begin
  u, θ, flux = 2, 350, 20
  MO_len = Nishizawa2018.compute_MO_len(u, θ, flux)
  @test MO_len ≈ -35.67787971457696

  u_ave, θ, flux, Δz, z_0, a = 110, 350, 20, 100, 0.01, 5
  Ψ_m_tol, tol_abs, iter_max = 1e-3, 1e-3, 10
  u_star = Nishizawa2018.compute_friction_velocity(u_ave, θ, flux, Δz, z_0, a, Ψ_m_tol, tol_abs, iter_max)
  @test u_star ≈ 5.526644550864822

  z, F_m, F_h, a, u_star, θ, flux, Pr = 1.0, 2.0, 3.0, 5, 110, 350, 20, 0.74

  K_m, K_h, L_mo = Nishizawa2018.compute_exchange_coefficients(z,F_m,F_h,a,u_star,θ,flux,Pr)
  @test K_m ≈ -11512.071612359368
  @test K_h ≈ -6111.6196776263805
  @test L_mo ≈ -5.935907237512742e6

end
