module TriDiagSolverFuncs

"""
solve_tridiag! solves for x in the equation
          Ax = b
when A is a tridiagonal matrix:
           _                                           _ -1
          |  dd[1] du[1]                                 |
          |  dl[1] dd[2]  du[2]                          |
          |        dl[2]  dd[3]  du[3]                   |
 x    =   |           *     *     *                      |   b
          |                 *     *     *                |
          |                    dl[n-2] dd[n-1]  du[n-1]  |
          |_                           dl[n-1]  dd[n]   _|

          |______________________________________________|
                                 A
"""
function solve_tridiag!(
  x::Vector{R},     #  x[1:n]       - the result
  b::Vector{R},     #  b[1:n]       - right hand side
  dl::Vector{R},    #  dl[1:n-1]    - sub-diagonal
  dd::Vector{R},    #  dd[1:n]      - main diagonal
  du::Vector{R},    #  du[1:n-1]    - super-diagonal
  n::Int,           #  n            - system size
  xtemp::Vector{R}, #  xtemp[1:n]   - temporary
  gamma::Vector{R}, #  gamma[1:n-1] - temporary
  beta::Vector{R}   #  beta[1:n]    - temporary
  ) where R
  # Define coefficients:
  beta[1] = dd[1]
  gamma[1] = du[1]/beta[1]
  for i in 2:n-1
    beta[i] = dd[i]-dl[i-1]*gamma[i-1]
    gamma[i] = du[i]/beta[i]
  end
  beta[n] = dd[n]-dl[n-1]*gamma[n-1]

  # Forward substitution:
  xtemp[1] = b[1]/beta[1]
  for i = 2:n
    m = b[i] - dl[i-1]*xtemp[i-1]
    xtemp[i] = m/beta[i]
  end

  # Backward substitution:
  x[n] = xtemp[n]
  for i in n-1:-1:1
    x[i] = xtemp[i]-gamma[i]*x[i+1]
  end
end


"""
solve_tridiag_stored! solves for x in the equation
          Ax = b
when A is a tridiagonal matrix. It can be seen that
coefficients in solve_tridiag! can be pre-computed,
by applying LU factorization to A (shown below).
The coefficients, beta and gamma, can be computed
in init_beta_gamma!.
 _                                             _
|  dd[1]  du[1]                                 |
|  dl[1]  dd[2]  du[2]                          |
|         dl[2]  dd[3]  du[3]                   |
|           *     *     *                       |
|                 *     *     *                 |
|                    dl[n-2]  dd[n-1]  du[n-1]  |
|_                            dl[n-1]  dd[n]   _|

=
 _                                               _   _                                          _ -1
|  beta[1]                                        | |  1  gamma[1]                               |
|  alpha[1]  beta[2]                              | |        1  gamma[2]                         |
|        alpha[2]  beta[3]                        | |              1  gamma[3]                   |
|           *     *     *                         | |                 *     *                    |
|                 *     *                         | |                       *     *              |
|                    alpha[n-2]  beta[n-1]        | |                             1   gamma[n-1] |
|_                           alpha[n-1]  beta[n] _| |_                                1         _|

"""
function solve_tridiag_stored!(
  x::Vector{R},      # x[1:n]       - result
  b::Vector{R},      # b[1:n]       - right-hand side
  dl::Vector{R},     # dl[1:n-1]    - sub-diagonal
  beta::Vector{R},   # beta[1:n]    - coefficient, computed from init_beta_gamma!
  gamma::Vector{R},  # gamma[1:n-1] - coefficient, computed from init_beta_gamma!
  n::Int,            # n            - system size
  xtemp::Vector{R}   # xtemp[1:n]   - temporary
  ) where R
  # Forward substitution:
  xtemp[1] = b[1]/beta[1]
  for i = 2:n
    m = b[i] - dl[i-1]*xtemp[i-1]
    xtemp[i] = m/beta[i]
  end

  # Backward substitution:
  x[n] = xtemp[n]
  for i = n-1:-1:1
    x[i] = xtemp[i]-gamma[i]*x[i+1]
  end
end

"""
init_beta_gamma! returns the coefficients, that can be pre-computed
using LU factorization, for the tridiagonal system. These coefficients
can be passed as arguments to solve_tridiag_stored!.
"""
function init_beta_gamma!(
  beta::Vector{R},  # updated        coefficients
  gamma::Vector{R}, # updated        coefficients
  dl::Vector{R},    #                  sub-diagonal
  dd::Vector{R},    #                 main-diagonal
  du::Vector{R},    #                super-diagonal
  n::Int            #                system size
  ) where R
  beta[1] = dd[1]
  gamma[1] = du[1]/beta[1]
  for i = 2:n-1
    beta[i] = dd[i]-dl[i-1]*gamma[i-1]
    gamma[i] = du[i]/beta[i]
  end
  beta[n] = dd[n]-dl[n-1]*gamma[n-1]
end

end