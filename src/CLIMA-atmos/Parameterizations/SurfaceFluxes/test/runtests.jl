using SurfaceFluxes
using Test
using Utilities.MoistThermodynamics
using Utilities.RootSolvers

@testset "SurfaceFluxes" begin
  # Not sure how to test this, just making sure it runs for now:
  windspeed = 1.0
  buoyancy_flux = 1.0
  z_0 = 0.1
  z_1 = 1.0
  z_b = 1.0
  iter_max = 10
  tol_abs = 1e-3
  u_star = SurfaceFluxes.compute_friction_velocity(windspeed,
                                                   buoyancy_flux,
                                                   z_0,
                                                   z_1,
                                                   tol_abs,
                                                   iter_max
                                                   )

  Ri = 1.0
  cm, ch, Λ_mo = SurfaceFluxes.exchange_coefficients_byun(Ri, z_b, z_0)

  shf = rand(1)[1]
  lhf = rand(1)[1]
  T_b = rand(1)[1]
  qt_b = rand(1)[1]
  ql_b = rand(1)[1]
  qi_b = rand(1)[1]
  alpha0_0 = rand(1)[1]

  buoyancy_flux = SurfaceFluxes.compute_buoyancy_flux(
  shf,
  lhf,
  T_b,
  qt_b,
  ql_b,
  qi_b,
  alpha0_0
  )
end


const HAVE_CUDA = try
  using CUDAdrv
  using CUDAnative
  using CuArrays
  true
catch
  false
end
@testset "CUDA SurfaceFluxes" begin
  if HAVE_CUDA
  # Not sure how to test this...
  end
end
