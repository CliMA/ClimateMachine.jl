using SurfaceFluxes.Nishizawa2018
using SurfaceFluxes.Byun1990
using Test
using Utilities.MoistThermodynamics
using Utilities.RootSolvers

@testset "Byun1990 SurfaceFluxes" begin
  # Not sure how to test this, just making sure it runs for now:

  u = rand(1)[1]
  flux = rand(1)[1]
  MO_len = Byun1990.compute_MO_len(u, flux)
  @test MO_len ≈ MO_len

  windspeed = rand(1)[1]
  buoyancy_flux = rand(1)[1]
  z_0 = rand(1)[1]
  z_1 = rand(1)[1]
  γ_m = 15.0
  β_m = 4.8

  tol_abs = 1e-3
  iter_max = 10
  u_star = Byun1990.compute_friction_velocity(windspeed,
                                              buoyancy_flux,
                                              z_0,
                                              z_1,
                                              β_m,
                                              γ_m,
                                              tol_abs,
                                              iter_max
                                              )
  @test u_star ≈ u_star

  shf = rand(1)[1]
  lhf = rand(1)[1]
  T_b = rand(1)[1]
  qt_b = rand(1)[1]
  ql_b = rand(1)[1]
  qi_b = rand(1)[1]
  alpha0_0 = rand(1)[1]
  Pr_0 = rand(1)[1]

  buoyancy_flux = Byun1990.compute_buoyancy_flux(shf,
                                                 lhf,
                                                 T_b,
                                                 qt_b,
                                                 ql_b,
                                                 qi_b,
                                                 alpha0_0
                                                 )
  @test buoyancy_flux ≈ buoyancy_flux

  Ri = rand(1)[1]
  z_b = rand(1)[1]
  z_0 = rand(1)[1]
  γ_m = 15.0
  γ_h = 9.0
  β_m = 4.8
  β_h = 7.8
  cm, ch, Λ_mo = Byun1990.compute_exchange_coefficients(Ri,
                                                        z_b,
                                                        z_0,
                                                        γ_m,
                                                        γ_h,
                                                        β_m,
                                                        β_h,
                                                        Pr_0
                                                        )
  @test cm ≈ cm
  @test ch ≈ ch
  @test Λ_mo ≈ Λ_mo

end

@testset "Nishizawa2018 SurfaceFluxes" begin
  # Not sure how to test this, just making sure it runs for now:

  u = rand(1)[1]
  flux = rand(1)[1]
  θ = rand(1)[1]
  MO_len = Nishizawa2018.compute_MO_len(u, θ, flux)
  @test MO_len ≈ MO_len

  shf = rand(1)[1]
  lhf = rand(1)[1]
  T_b = rand(1)[1]
  qt_b = rand(1)[1]
  ql_b = rand(1)[1]
  qi_b = rand(1)[1]
  alpha0_0 = rand(1)[1]
  Pr_0 = rand(1)[1]

  windspeed = rand(1)[1]
  buoyancy_flux = rand(1)[1]
  z_0 = rand(1)[1]
  z_1 = rand(1)[1]
  γ_m = 15.0
  β_m = 4.8

  tol_abs = 1e-3
  iter_max = 10
  # u_star = Nishizawa2018.compute_friction_velocity(windspeed,
  #                                             buoyancy_flux,
  #                                             z_0,
  #                                             z_1,
  #                                             β_m,
  #                                             γ_m,
  #                                             tol_abs,
  #                                             iter_max
  #                                             )
  # @test u_star ≈ u_star


  # buoyancy_flux = Nishizawa2018.compute_buoyancy_flux(shf,
  #                                                lhf,
  #                                                T_b,
  #                                                qt_b,
  #                                                ql_b,
  #                                                qi_b,
  #                                                alpha0_0
  #                                                )
  # @test buoyancy_flux ≈ buoyancy_flux

  # Ri = rand(1)[1]
  # z_b = rand(1)[1]
  # z_0 = rand(1)[1]
  # γ_m = 15.0
  # γ_h = 9.0
  # β_m = 4.8
  # β_h = 7.8
  # cm, ch, Λ_mo = Nishizawa2018.compute_exchange_coefficients(Ri,
  #                                                            z_b,
  #                                                            z_0,
  #                                                            γ_m,
  #                                                            γ_h,
  #                                                            β_m,
  #                                                            β_h,
  #                                                            Pr_0
  #                                                            )
  # @test cm ≈ cm
  # @test ch ≈ ch
  # @test Λ_mo ≈ Λ_mo

end

@static if Base.find_package("CuArrays") !== nothing
  using CUDAdrv
  using CUDAnative
  using CuArrays
  @testset "CUDA SurfaceFluxes" begin

    # include("src/SurfaceFluxes.jl")

    windspeed = cu(rand(5,5))
    buoyancy_flux = cu(rand(5,5))
    z_0 = cu(rand(5,5))
    z_1 = cu(rand(5,5))
    γ_m = 15.0
    β_m = 4.8

    tol_abs = 1e-3
    iter_max = 10
    u_star = SurfaceFluxes.Byun1990.compute_friction_velocity.(windspeed,
                                                               buoyancy_flux,
                                                               z_0,
                                                               z_1,
                                                               Ref(β_m),
                                                               Ref(γ_m),
                                                               tol_abs,
                                                               iter_max
                                                               )
  end
end
