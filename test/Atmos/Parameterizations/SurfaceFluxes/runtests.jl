using Test

using CLIMA.SurfaceFluxes
using CLIMA.MoistThermodynamics
using CLIMA.RootSolvers

# FIXME: Use realistic values / test for correctness
# These tests have been run to ensure they do not fail,
# but they need further testing for correctness.

@testset "SurfaceFluxes" begin

  x_initial = [100, 15.0, 350.0]
  F_exchange = [2.0, 3.0]
  z_0 = [1.0, 1.0]
  dimensionless_number = [1.0, 0.74]
  x_ave = [5.0, 350.0]
  x_s   = [0.0, 300.0]

  Δz = 2.0
  z = 0.5
  θ_bar = 300.0
  a = 4.7
  pottemp_flux_given = -200.0
  args = x_initial, x_ave, x_s, z_0, F_exchange, dimensionless_number, θ_bar, Δz, z, a, pottemp_flux_given
  sfc = surface_conditions(args[1:end-1]...)
  @show sfc
  sfc = surface_conditions(args...)
  @show sfc
end

