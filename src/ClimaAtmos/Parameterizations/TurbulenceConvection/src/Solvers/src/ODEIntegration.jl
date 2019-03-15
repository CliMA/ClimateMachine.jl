module ODEIntegration

using Grids
using Fields
using Fields.ArithmeticFuncs
using Fields.GridOperators

"""
integrate_ode! integrates

  dy/dz = func(y, z, args)
  y = y_0 + int_{z=0}^{z} func(y, z) dz

using Trapezoidal method.
Interpolations are used because
the integral lives on the dual
grid of the primary data (Y).
This interp moves the integral
back to the primary grid.

"""

function integrate_ode!(Y_sum::primary,
                        grid::Grid,
                        func::Function,
                        Y_0,
                        args::Tuple,
                        Y_dual::dual
                        ) where {primary <: Fields.Center, dual <: Fields.Node}
  assign!(Y_dual, Y_0)
  assign!(Y_sum, 0.0)
  z = grid.z[get_type(Y_dual)]
  for k in over_elems_real(Y_sum, grid)
    fz1 = func(Y_dual.val[ k ], z[ k ], args...) # Note differences in indexes
    fz2 = func(Y_dual.val[k+1], z[k+1], args...) # Note differences in indexes
    Y_sum[k] = Y_sum[k-1] + grid.dz * 0.5*(fz1+fz2)
  end
  add!(Y_sum, Y_0)
  assign_ghost!(Y_sum, 0.0)
end

function integrate_ode!(Y_sum::primary,
                        grid::Grid,
                        func::Function,
                        Y_0,
                        args::Tuple,
                        Y_dual::dual
                        ) where {primary <: Fields.Node, dual <: Fields.Center}
  assign!(Y_dual, Y_0)
  assign!(Y_sum, 0.0)
  z = grid.z[get_type(Y_dual)]
  for k in over_elems_real(Y_sum, grid)
    fz1 = func(Y_dual.val[k-1], z[k-1], args...) # Note differences in indexes
    fz2 = func(Y_dual.val[ k ], z[ k ], args...) # Note differences in indexes
    Y_sum[k] = Y_sum[k-1] + grid.dz * 0.5*(fz1+fz2)
  end
  add!(Y_sum, Y_0)
  assign_ghost!(Y_sum, 0.0)
end

end