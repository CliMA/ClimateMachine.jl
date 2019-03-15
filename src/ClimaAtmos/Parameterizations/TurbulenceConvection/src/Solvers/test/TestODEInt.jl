using Test

using Solvers: ODEIntegration

using Grids
using Fields

rhs(Y, Z, args) = Z^2.0/10.0 + 3.0

function test_int(data_type::UnionAll, # <:{Fields.Center, Fields.Node}
                  grid::Grid,
                  rhs::Function,
                  Y_0,
                  args::Tuple)
  fint = data_type(grid, 0.0)
  f_dual = Fields.init_dual(fint, grid)
  ODEIntegration.integrate_ode!(fint, grid, rhs, Y_0, args, f_dual)
  data_loc = get_type(fint)
  fcorrect = data_loc(grid, 0.0)
  z = grid.z[data_loc]
  fcorrect.val[:] = (z[:].^3.0)/30.0 + 3.0*z[:]
  fcorrect.val[1] = 0.0
  fcorrect.val[end] = 0.0
  return sum([x - y for (x,y) in zip(fint.val, fcorrect.val)])/sum(fcorrect.val)
end

@testset "TestODEInt" begin
  K_range = 1:4
  a = 0.0
  b = 10.0
  Y_0 = 0.0
  args = Tuple([1])
  dz_range = [5.0/(10.0^k) for k in K_range]

  @testset begin
    for (dz, k) in zip(dz_range, K_range)
      N_z = ceil(Int64, (b-a)/dz)
      grid = Grid(a, b, N_z)
      err = test_int(Fields.Node, grid, rhs, Y_0, args)
      @test abs(err) <= dz
    end
  end

  @testset begin
    for (dz, k) in zip(dz_range, K_range)
      N_z = ceil(Int64, (b-a)/dz)
      grid = Grid(a, b, N_z)
      err = test_int(Fields.Center, grid, rhs, Y_0, args)
      @test abs(err) <= dz
    end
  end
end


