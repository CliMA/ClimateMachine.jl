using Test

using CLIMA.SurfaceFluxes
using CLIMA.MoistThermodynamics
using CLIMA.RootSolvers

# FIXME: Use realistic values / test for correctness
# These tests have been run to ensure they do not fail,
# but they need further testing for correctness.

@testset "SurfaceFluxes" begin

  F_m, F_h = 2.0, 3.0
  Pr = 0.74
  u_ave = 5.0
  θ_ave = 300.0
  θ_bar = 330.0
  θ_s   = 300.0
  Δz = 100.0
  z = 5.0
  z_0_m = 10.0
  z_0_h = 10.0
  a = 4.7
  pottemp_flux = 0.2

  sfc = surface_conditions(u_ave,
                           θ_ave,
                           θ_bar,
                           θ_s,
                           Δz,
                           z_0_m,
                           z_0_h,
                           z,
                           F_m,
                           F_h,
                           a,
                           Pr,
                           pottemp_flux
                           )
  @show sfc
  # @test sfc.momentum_flux ≈
  # @test sfc.buoyancy_flux ≈
  # @test sfc.Monin_Obukhov_length ≈
  # @test sfc.friction_velocity ≈
  # @test sfc.temperature_scale ≈
  # @test sfc.exchange_coeff_momentum ≈
  # @test sfc.exchange_coeff_heat ≈

end

# @testset "SurfaceFluxes" begin
#   u, flux = 0.1, 0.2
#   MO_len = Byun1990.compute_MO_len(u, flux)
#   @test MO_len ≈ -0.0125

#   u_ave, buoyancy_flux, z_0, z_1 = 0.1, 0.2, 2.0, 5.0
#   γ_m, β_m = 15.0, 4.8
#   tol_abs, iter_max = 1e-3, 10
#   u_star = Byun1990.compute_friction_velocity(u_ave,
#                                               buoyancy_flux,
#                                               z_0,
#                                               z_1,
#                                               β_m,
#                                               γ_m,
#                                               tol_abs,
#                                               iter_max
#                                               )
#   @test u_star ≈ 0.201347256193615 atol=tol_abs


#   Ri, z_b, z_0, Pr_0 = 10, 2.0, 5.0, 0.74
#   γ_m, γ_h, β_m, β_h = 15.0, 9.0, 4.8, 7.8
#   cm, ch, L_mo = Byun1990.compute_exchange_coefficients(Ri, z_b, z_0, γ_m, γ_h, β_m, β_h, Pr_0)
#   @test cm ≈ 19.700348427787368
#   @test ch ≈ 3.3362564728997803
#   @test L_mo ≈ -14.308268023583906

#   Ri, z_b, z_0, Pr_0 = -10, 10.0, 1.0, 0.74
#   γ_m, γ_h, β_m, β_h = 15.0, 9.0, 4.8, 7.8
#   cm, ch, L_mo = Byun1990.compute_exchange_coefficients(Ri, z_b, z_0, γ_m, γ_h, β_m, β_h, Pr_0)
#   @test cm ≈ 0.33300280321092746
#   @test ch ≈ 1.131830939627489
#   @test L_mo ≈ -0.3726237964444814

# end

# @testset "SurfaceFluxes.Nishizawa2018" begin
#   u, θ, flux = 2, 350, 20
#   MO_len = Nishizawa2018.compute_MO_len(u, θ, flux)
#   @test MO_len ≈ -35.67787971457696

#   u_ave, θ, flux, Δz, z_0, a = 110, 350, 20, 100, 0.01, 5
#   Ψ_m_tol, tol_abs, iter_max = 1e-3, 1e-3, 10
#   u_star = Nishizawa2018.compute_friction_velocity(u_ave, θ, flux, Δz, z_0, a, Ψ_m_tol, tol_abs, iter_max)
#   @test u_star ≈ 5.526644550864822 atol=tol_abs

#   z, F_m, F_h, a, u_star, θ, flux, Pr = 1.0, 2.0, 3.0, 5, 110, 350, 20, 0.74

#   K_m, K_h, L_mo = Nishizawa2018.compute_exchange_coefficients(z,F_m,F_h,a,u_star,θ,flux,Pr)
#   @test K_m ≈ -11512.071612359368
#   @test K_h ≈ -6111.6196776263805
#   @test L_mo ≈ -5.935907237512742e6

# end
