#### TriDiagSolverFuncs

# A module with functions to solve tridiagonal
# systems of equations using the Thomas algorithm.

export solve_tridiag!
export solve_tridiag_stored!
export init_β_γ!

"""
    solve_tridiag!(x, B, a, b, c, n, xtemp, γ, β)

Solves for `x` in the equation
          `Ax = B`
where `A` is a tridiagonal matrix:
```
           _                                           _ -1
          |  b[1] c[1]                                   |
          |  a[1] b[2]  c[2]                             |
          |       a[2]  b[3]  c[3]                       |
 x    =   |           *     *     *                      |   B
          |                 *     *     *                |
          |                    a[n-2] b[n-1]  c[n-1]     |
          |_                           a[n-1]  b[n]     _|

          |______________________________________________|
                                 A
```
and given arguments:
--------------------------------------------
| x[1:n]       | the result                |
| B[1:n]       | right hand side           |
| a[1:n-1]     | sub-diagonal              |
| b[1:n]       | main diagonal             |
| c[1:n-1]     | super-diagonal            |
| n            | system size               |
| xtemp[1:n]   | temporary                 |
| γ[1:n-1]     | temporary                 |
| β[1:n]       | temporary                 |
--------------------------------------------
"""
function solve_tridiag!(x, B, a, b, c, n, xtemp, γ, β)
  # Define coefficients:
  β[1] = b[1]
  γ[1] = c[1]/β[1]
  for i in 2:n-1
    β[i] = b[i]-a[i-1]*γ[i-1]
    γ[i] = c[i]/β[i]
  end
  β[n] = b[n]-a[n-1]*γ[n-1]

  # Forward substitution:
  xtemp[1] = B[1]/β[1]
  for i = 2:n
    m = B[i] - a[i-1]*xtemp[i-1]
    xtemp[i] = m/β[i]
  end

  # Backward substitution:
  x[n] = xtemp[n]
  for i in n-1:-1:1
    x[i] = xtemp[i]-γ[i]*x[i+1]
  end
end


"""
    solve_tridiag_stored!(x, B, a, β, γ, n, xtemp)

Solves for `x` in the equation
          `Ax = B`
where `A` is a tridiagonal matrix.

Coefficients in solve_tridiag! can be pre-computed,
by applying LU factorization to A (shown below).
The coefficients, β and γ, can be computed in init_β_γ!.
```
 _                                           _
|  b[1]  c[1]                                 |
|  a[1]  b[2]  c[2]                           |
|         a[2]  b[3]  c[3]                    |
|           *     *     *                     |
|                 *     *     *               |
|                    a[n-2]  b[n-1]  c[n-1]   |
|_                            a[n-1]  b[n]   _|

=
 _                                        _   _                                      _ -1
|  β[1]                                    | |  1  γ[1]                               |
|  α[1]  β[2]                              | |        1  γ[2]                         |
|        α[2]  β[3]                        | |              1  γ[3]                   |
|           *     *     *                  | |                 *     *                |
|                 *     *                  | |                       *     *          |
|                    α[n-2]  β[n-1]        | |                             1   γ[n-1] |
|_                           α[n-1]  β[n] _| |_                                1     _|
```

and given arguments:
--------------------------------------------
| x[1:n]       | the result                |
| B[1:n]       | right hand side           |
| a[1:n-1]     | sub-diagonal              |
| β[1:n]       | temporary                 |
| γ[1:n-1]     | temporary                 |
| n            | system size               |
| xtemp[1:n]   | temporary                 |
--------------------------------------------
"""
function solve_tridiag_stored!(x, B, a, β, γ, n, xtemp)
  # Forward substitution:
  xtemp[1] = B[1]/β[1]
  for i = 2:n
    m = B[i] - a[i-1]*xtemp[i-1]
    xtemp[i] = m/β[i]
  end

  # Backward substitution:
  x[n] = xtemp[n]
  for i = n-1:-1:1
    x[i] = xtemp[i]-γ[i]*x[i+1]
  end
end

"""
    init_β_γ!(β, γ, a, b, c, n)

Returns the pre-computed coefficients, from applying
LU factorization, for the tridiagonal system. These
coefficients can be passed as arguments to solve_tridiag_stored!.
--------------------------------------------
| β[1:n]       | temporary                 |
| γ[1:n-1]     | temporary                 |
| a[1:n-1]     | sub-diagonal              |
| b[1:n]       | main diagonal             |
| c[1:n-1]     | super-diagonal            |
| n            | system size               |
--------------------------------------------
"""
function init_β_γ!(β, γ, a, b, c, n)
  β[1] = b[1]
  γ[1] = c[1]/β[1]
  for i = 2:n-1
    β[i] = b[i]-a[i-1]*γ[i-1]
    γ[i] = c[i]/β[i]
  end
  β[n] = b[n]-a[n-1]*γ[n-1]
end
