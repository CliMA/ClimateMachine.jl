using MPI
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using Printf
using LinearAlgebra

@inline function constant_auxiliary_init!(ϕ_c, x, y, z, dim)
  @inbounds begin
    if dim == 2
      ϕ_c[1] = x^2 + y^3 - x*y
      ϕ_c[5] = 2*x - y
      ϕ_c[6] = 3*y^2 - x
      ϕ_c[7] = 0
    else
      ϕ_c[1] = x^2 + y^3 + z^2*y^2 - x*y*z
      ϕ_c[5] = 2*x - y*z
      ϕ_c[6] = 3*y^2 + 2*z^2*y - x*z
      ϕ_c[7] = 2*z*y^2 - x*y
    end
  end
end

using Test
function run(dim, Ne, N, DFloat)
  ArrayType = Array

  MPI.Initialized() || MPI.Init()
  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())

  mpicomm = MPI.COMM_WORLD

  brickrange = ntuple(j->range(DFloat(-1); length=Ne[j]+1, stop=1), dim)
  topl = BrickTopology(mpicomm, brickrange, periodicity=ntuple(j->true, dim))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = 0,
                           flux! = (x...) -> (),
                           numericalflux! = (x...) -> (),
                           length_constant_auxiliary = 7,
                           constant_auxiliary_init! = (x...) ->
                           constant_auxiliary_init!(x..., dim))

  DGBalanceLawDiscretizations.grad_constant_auxiliary!(spacedisc, 1, (2,3,4))

  @test spacedisc.auxc.Q[:, 2, :] ≈ spacedisc.auxc.Q[:, 5, :]
  @test spacedisc.auxc.Q[:, 3, :] ≈ spacedisc.auxc.Q[:, 6, :]
  @test spacedisc.auxc.Q[:, 4, :] ≈ spacedisc.auxc.Q[:, 7, :]
end

let
  numelem = (5, 5, 1)
  lvls = 1

  polynomialorder = 4

  for DFloat in (Float64,) #Float32)
    for dim = 2:3
      err = zeros(DFloat, lvls)
      for l = 1:lvls
        run(dim, ntuple(j->2^(l-1) * numelem[j], dim), polynomialorder, DFloat)
      end
    end
  end
end

isinteractive() || MPI.Finalize()

nothing
