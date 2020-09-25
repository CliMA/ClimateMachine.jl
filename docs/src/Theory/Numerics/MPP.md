# Positivity Preservation

It is important to conserve mass and perserve positive sign of active tracers,
  such as suspended or precipitating water.
The tracer concentrations may turn negative because of the
  numerical errors caused either by the advection scheme,
  or by time integration of source terms.
Implemented method ensures positivity when solving the
  advection-diffusion problem and falls under the
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
    that truncates the negative nodel points to zero
    and adjusts the values of the remaining points
    to preserve the element average.


## Flux correction

The flux correction is implemented following
  [Xiong\_et\_al\_2015](https://epubs.siam.org/doi/10.1137/140965326).
Because of the efficiency concerns, the correction is applied only
  at the last stage of RK algorithm.
As a result the positivity is preserved only at the final stage
  and not at all the intermediate RK stages.




## Rescaling


## Limitations

  - The MPP was only tested with LSRK explicit timesteppers.
  - Positivity is ensured only at the last RK stage.
  - This approach could potentailly be extended to offer a local limiter.
    (This is not described in the literature. Not sure how effective).

