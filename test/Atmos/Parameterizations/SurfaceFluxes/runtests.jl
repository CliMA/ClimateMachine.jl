using Test

using CLIMA.SurfaceFluxes
using CLIMA.MoistThermodynamics
using CLIMA.RootSolvers

# FIXME: Use realistic values / test for correctness
# These tests have been run to ensure they do not fail,
# but they need further testing for correctness.

@testset "SurfaceFluxes" begin

  F_m, F_h = 2.0, 3.0
  Δz = 2.0
  z = 0.5
  z_0_m = 1.0
  z_0_h = 1.0
  pottemp_flux = 0.2

  Pr = 0.74
  u_ave = 5.0
  θ_ave = 300.0
  θ_bar = 350.0
  θ_s   = 300.0
  a = 4.7

  args = u_ave, θ_ave, θ_bar, θ_s, Δz, z_0_m, z_0_h, z, F_m, F_h, a, Pr, pottemp_flux
  sfc = surface_conditions(args...)
  @show sfc
  sfc = surface_conditions(args[1:end-1]...)
  @show sfc
end

