# Positivity Preservation

It is important to conserve mass and perserve positive sign of active tracers,
  such as suspended or precipitating water.
The tracer concentrations may turn negative because of the
  numerical errors caused either by the advection scheme,
  or by time integration of source terms.
Implemented method ensures positivity when solving the
  advection-(TODO: diffusion) problem and falls under the
  Maximum Principle Preserving (MPP) algorithm category.
Ensuring positivity while integrating source terms is left as future work.

Implementation follows the approach presented by
[Light\_and\_Durran\_2016](https://journals.ametsoc.org/mwr/article/144/12/4771/70817/Preserving-Nonnegativity-in-Discontinuous-Galerkin)
and
[Xiong\_et\_al\_2015](https://epubs.siam.org/doi/10.1137/140965326).
Positivity preservation is done in two steps:
  - In the first step, a flux correction is applied to those DG elements,
    whose average value would otherwise turn negative
    at the end of the time step.
  - Secondly, an adjustment is applied to nodal points inside the element
    that truncates the negative nodal points to zero
    and adjusts the values of the remaining points
    to preserve the element average.


## Flux correction

The flux correction is implemented following
  [Xiong\_et\_al\_2015](https://epubs.siam.org/doi/10.1137/140965326).
Because of the efficiency concerns, the correction is applied only
  at the last stage of RK algorithm.
As a result the positivity is preserved only at the final stage
  and not at all the intermediate RK stages.

The correction is obtained with the help of a finite volume scheme
  (i.e. low order sheme) running in parallel with the DG scheme
  (i.e. high order scheme).
Let ``H^{rk}_{j+1/2}`` and ``h_{j+1/2}`` denote the high order and low order
  schme fluxes through element face ``j+1/2``.
The corrected ``\hat{H}^{rk}_{j+1/2}`` is computed
  with the help of limiter ``\theta_{j+1/2}``:
```math
\hat{H}^{rk}_{j+1/2} = \theta_{j+1/2} H^{rk}_{j+1/2} + (1 - \theta_{j+1/2}) h_{j+1/2}
```
If ``H^{rk}_{j+1/2}`` does not cause negative numbers,
  ``\theta_{j+1/2} is set to 1.
Otherwise


## Rescaling

The rescaling is implemented following
  [Light\_and\_Durran\_2016](https://journals.ametsoc.org/mwr/article/144/12/4771/70817/Preserving-Nonnegativity-in-Discontinuous-Galerkin)
  chapter 3.

For scalar ``\phi`` the negative nodal points ``k`` in element ``i``
  are truncated to zero:
```math
\phi_{i,k}^{+} = \left\{
    \begin{array}{ll}
        \phi_{i,k} & \mathrm{if} \;\; \phi_{i,k} \geq 0 \\
        0 & \mathrm{if} \;\; \phi_{i,k} < 0
    \end{array}
\right.
```
The rescaling factor is computed as:
```math
r_i = \frac{\overline{\phi_i}}{\overline{\phi_i^+}}
```
where:
 - ``\overline{\phi_i}`` and ``\overline{\phi_i^+}`` denote the average
   element value before and after the truncation.
Finally the rescaled nodal points ``\phi_{i,k}^{*}`` are defined as:
```math
\phi_{i,k}^{*} = \left\{
    \begin{array}{ll}
        r_i \phi_{i,k} & \mathrm{if} \;\; \phi_{i,k} \geq 0 \\
        0 & \mathrm{if} \;\; \phi_{i,k} < 0
    \end{array}
\right.
```

## Limitations

  - The MPP was only tested with LSRK explicit timesteppers.

  - Positivity is ensured only at the last RK stage.

  - This approach could potentailly be extended to offer a local limiter.
    (This is not described in the literature. Not sure how effective).

## Results (?)
