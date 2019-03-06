using SurfaceFluxes.Nishizawa2018
using SurfaceFluxes.Byun1990
using Test
using Utilities.MoistThermodynamics
using Utilities.RootSolvers

@testset "Byun1990 SurfaceFluxes" begin
  # Not sure how to test this, just making sure it runs for now:

  u, flux = rand(2,1)
  MO_len = Byun1990.compute_MO_len(u, flux)
  @test MO_len ≈ MO_len

  u_ave, buoyancy_flux, z_0, z_1 = rand(4,1)
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
  @test u_star ≈ u_star

  shf, lhf, T_b, qt_b, ql_b, qi_b, alpha0_0 = rand(7,1)

  buoyancy_flux = Byun1990.compute_buoyancy_flux(shf,
                                                 lhf,
                                                 T_b,
                                                 qt_b,
                                                 ql_b,
                                                 qi_b,
                                                 alpha0_0
                                                 )
  @test buoyancy_flux ≈ buoyancy_flux

  Ri, z_b, z_0, Pr_0 = rand(4,1)
  γ_m, γ_h, β_m, β_h = 15.0, 9.0, 4.8, 7.8
  cm, ch, L_mo = Byun1990.compute_exchange_coefficients(Ri,
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
  @test L_mo ≈ L_mo

end

@testset "Nishizawa2018 SurfaceFluxes" begin
  # Not sure how to test this, just making sure it runs for now:
  u, θ, flux = rand(3,1)
  MO_len = Nishizawa2018.compute_MO_len(u, θ, flux)
  @test MO_len ≈ MO_len

  u_ave, θ, flux, Δz, z_0, a = rand(6,1)
  u_ave = 100+u_ave*10
  z_0 = z_0/1000
  Ψ_m_tol, tol_abs, iter_max = 1e-3, 1e-3, 10
  u_star = Nishizawa2018.compute_friction_velocity(u_ave, θ, flux, Δz, z_0, a, Ψ_m_tol, tol_abs, iter_max)
  @test u_star ≈ u_star

  z, F_m, F_h, a, u_star, θ, flux, Pr = rand(8,1)

  K_m, K_h, L_mo = Nishizawa2018.compute_exchange_coefficients(z,F_m,F_h,a,u_star,θ,flux,Pr)
  @test K_m ≈ K_m
  @test K_h ≈ K_h
  @test L_mo ≈ L_mo

end

@static if Base.find_package("CuArrays") !== nothing
  using CUDAdrv
  using CUDAnative
  using CuArrays
  @testset "CUDA SurfaceFluxes" begin

    u_ave = cu(rand(5,5))
    buoyancy_flux = cu(rand(5,5))
    z_0 = cu(rand(5,5))
    z_1 = cu(rand(5,5))
    γ_m = 15.0
    β_m = 4.8

    tol_abs = 1e-3
    iter_max = 10
    # u_star = Byun1990.compute_friction_velocity.(u_ave,
    #                                              buoyancy_flux,
    #                                              z_0,
    #                                              z_1,
    #                                              Ref(β_m),
    #                                              Ref(γ_m),
    #                                              Ref(tol_abs),
    #                                              Ref(iter_max)
    #                                              )
  end
end

@static if Base.find_package("Plots") !== nothing
  linspace(a, b, n) = collect(a .+ (b-a).*range(0.0, stop=1.0, length=n))
  using Plots
  Ri = linspace(-1.2, 0.4, 100)
  z_b, z_0, Pr_0 = rand(3,1)
  γ_m, γ_h, β_m, β_h = 15.0, 9.0, 4.8, 7.8
  R = Byun1990.compute_exchange_coefficients.(Ri,
                                              Ref(z_b),
                                              Ref(z_0),
                                              Ref(γ_m),
                                              Ref(γ_h),
                                              Ref(β_m),
                                              Ref(β_h),
                                              Ref(Pr_0)
                                              )
  cm = [cm for (cm, ch, L_mo) in R]
  ch = [ch for (cm, ch, L_mo) in R]
  L_mo = [L_mo for (cm, ch, L_mo) in R]

  # To verify with Fig 4 in Ref. Byun1990
  plot(Ri, cm, label="cm")
  plot!(Ri, ch, label="ch")
  plot!(Ri, L_mo, label="L_mo")
  plot!(title = "exchange coefficients vs Ri", xlabel = "Ri", ylabel = "exchange coefficients")
  png("exchange_vs_Ri")
end

@static if Base.find_package("Plots") !== nothing
  linspace(a, b, n) = collect(a .+ (b-a).*range(0.0, stop=1.0, length=n))
  using Plots
  L, a, θ, z, flux = rand(5,1)
  z = z/100
  u = linspace(-0.1, 0.1, 100)
  L = Nishizawa2018.compute_MO_len.(u, θ, flux)
  ζ = (z/L)'
  ϕ_m = zeros(length(u))
  for k in eachindex(ϕ_m)
    ϕ_m[k] = Nishizawa2018.compute_ϕ_m(ζ[k], L[k], a)
  end
  # To verify with Fig 1 in Ref. Businger
  plot(ζ, ϕ_m, label="phi_m")
  plot!(title = "phi_m vs zeta", xlabel = "zeta", ylabel = "phi_m")
  png("phi_vs_zeta")
end

@static if Base.find_package("Plots") !== nothing
  linspace(a, b, n) = collect(a .+ (b-a).*range(0.0, stop=1.0, length=n))
  using Plots
  u_ave, a, θ, flux, z_0 = rand(5,1)
  z_0 = z_0/1000
  u_ave = 200+u_ave*10
  Δz = linspace(10.0, 100.0, 100)
  Ψ_m_tol, tol_abs, iter_max = 1e-3, 1e-3, 10
  u_star = Nishizawa2018.compute_friction_velocity.(Ref(u_ave),
                                                    Ref(θ),
                                                    Ref(flux),
                                                    Δz,
                                                    Ref(z_0),
                                                    Ref(a),
                                                    Ref(Ψ_m_tol),
                                                    Ref(tol_abs),
                                                    Ref(iter_max))
  plot(u_star, Δz, label="Friction velocity")
  plot!(title = "Friction velocity vs dz", xlabel = "Friction velocity", ylabel = "dz")
  png("ustar_vs_dz")
end

