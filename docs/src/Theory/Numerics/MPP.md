# Positivity Preservation

It is important to preserve positive sign of active tracers,
  such as the mass fraction of the suspended or precipitating water.
The tracer mass may unphysically turn negative because of the
  numerical errors caused either by the advection scheme,
  or by time integration of source terms.
The algorithm described here ensures positivity when solving the
  advection-(TODO: diffusion) problem and falls under the
  Maximum Principle Preserving (MPP) category.
Ensuring positivity while integrating source terms is left as future work.

The implementation follows
[Light\_and\_Durran\_2016](https://journals.ametsoc.org/mwr/article/144/12/4771/70817/Preserving-Nonnegativity-in-Discontinuous-Galerkin)
and
[Xiong\_et\_al\_2015](https://epubs.siam.org/doi/10.1137/140965326).
Positivity preservation is done in two steps:
  - Flux correction is applied to those DG elements,
    whose average value would otherwise turn negative
    at the end of the time step.
  - An adjustment is applied to nodal points inside the element
    that truncates the negative nodal points to zero
    and adjusts the values of the remaining points
    to preserve the element average.

The MPP algorithm can be applied to a user-defined sub-set of state variables.


## Flux correction

The flux correction is implemented following
  [Xiong\_et\_al\_2015](https://epubs.siam.org/doi/10.1137/140965326).
Because of the efficiency concerns, the correction is applied only
  at the last stage of the Runge-Kutta (RK) algorithm.
As a result the positivity is preserved only at the final timestepping stage
  and not during the intermediate RK stages.
The correction is obtained with the help of a finite volume (FV) scheme
  (i.e. low order scheme) running in parallel with the DG scheme
  (i.e. high order scheme).
A low order FV scheme preserves the minima/maxima
  and provides a bound on how big a DG flux through an element face can be
  to preserve positivity.
Below we show the algorithm for the 1-dimensional case.

Let ``H_{j+\frac{1}{2}}`` and ``h_{j+\frac{1}{2}}`` denote
  the high order and low order scheme fluxes
  through element face ``j+\frac{1}{2}`` of the DG element ``j``.
Additionally, let ``F_{j+\frac{1}{2}}`` denote the difference
  between the high and low order scheme fluxes
  through element face ``j+\frac{1}{2}``:
```math
  F_{j+\frac{1}{2}} = H_{j+\frac{1}{2}} - h_{j+\frac{1}{2}}
```
The corrected ``\hat{H}_{j+\frac{1}{2}}`` flux through
  the face ``j+\frac{1}{2}`` is computed with the limiter
  ``\theta_{j+\frac{1}{2}}``:
```math
\hat{H}_{j+\frac{1}{2}} = \theta_{j+\frac{1}{2}} H_{j+\frac{1}{2}} + (1 - \theta_{j+\frac{1}{2}}) h_{j+\frac{1}{2}}
```
If ``H_{j+\frac{1}{2}}`` does not cause negative numbers,
  ``\theta_{j+\frac{1}{2}}`` is set to 1
  and the final DG solution is not changed.
Otherwise:
```math
\theta_{j+\frac{1}{2}} = min(\Lambda_{j+\frac{1}{2}}, \Lambda_{j+1-\frac{1}{2}})
```
where ``\Lambda_{j+\frac{1}{2}}`` is the helper limiter concerned with
  the flux out of element ``j``
  and ``\Lambda_{j+1-\frac{1}{2}}`` is the helper limiter concerned with
  the flux out of element ``j+1``.
The helper limiter is defined as
```math
\Lambda_{j+\frac{1}{2}} = \frac{\phi^{t+1}_{j, FV} - \phi_{min}}{\frac{\Delta t}{\Delta x} \Delta F_{j, out}}
```
where:
 - ``\phi_{min} = 0`` is the minimum allowed value
 - ``\phi^{n+1}_{j, FV} = \phi^{n}_{j, FV} - \frac{\Delta t}{\Delta x} (h_{j+\frac{1}{2}} - h_{j-\frac{1}{2}})``
     is the advection diffusion solution of the FV scheme
 - ``\Delta F_{j, out} = max(0, F_{j+\frac{1}{2}}) + max(0, F_{j-\frac{1}{2}})``
     is the sum of the flux differences between the DG and FV schemes
     for element ``j``, but only those contributions where DG overestimates
     the flux compared to FV
 - ``\Delta t`` is the model time step
 - ``\Delta x_j`` is the FV grid resolution.
The nominator represents the maximum scalar concentration
  that can be moved out of the DG element in a given time step,
  based on the FV scheme.
The denominator sums the "overshoots" from the DG scheme
  compared to the FV scheme that would result in negative average element value.
Notice that the limiter on the ``j+\frac{1}{2}`` face potentially depends
  on all the fluxes going in/out of the element ``j``.
The MPP implementation requires additional memory allocated to store the
  FV solutions for all elements and the limiters on all the element faces.
See [Xiong\_et\_al\_2015](https://epubs.siam.org/doi/10.1137/140965326)
  for the full derivation
  (eq. 2.12 and above define the limiters in a more detailed way).


## Rescaling

The rescaling is implemented following
  [Light\_and\_Durran\_2016](https://journals.ametsoc.org/mwr/article/144/12/4771/70817/Preserving-Nonnegativity-in-Discontinuous-Galerkin)
  Chapter 3.
For scalar ``\phi`` in DG element ``j`` with nodal points ``k``
  the truncation is:
```math
\phi_{j,k}^{+} = \left\{
    \begin{array}{ll}
        \phi_{j,k} & \mathrm{if} \;\; \phi_{j,k} \geq 0 \\
        0 & \mathrm{if} \;\; \phi_{j,k} < 0
    \end{array}
\right.
```
The rescaling factor is computed as:
```math
r_j = \frac{\overline{\phi_j}}{\overline{\phi_j^+}}
```
where:
 - ``\overline{\phi_{j}}`` and ``\overline{\phi_j^+}`` denote the average
   element value before and after the truncation.
Finally, the rescaled nodal point values
``\phi_{j,k}^{*}`` are defined as:
```math
\phi_{j,k}^{*} = \left\{
    \begin{array}{ll}
        r_j \phi_{j,k} & \mathrm{if} \;\; \phi_{j,k} \geq 0 \\
        0 & \mathrm{if} \;\; \phi_{j,k} < 0
    \end{array}
\right.
```
The flux correction guarantees that the average element value is non-negative.
Because of this, the above truncation and mass adjustment, will not introduce
  any additional mass into the system.


## Limitations

 - Positivity is ensured only at the last RK stage.

 - The MPP algorithm is implemented with LSRK explicit timestepper.
   Adding it to a different timestepper, would require coding FV scheme
   to work in tandem with it.

 - This approach could potentially be extended to offer a local limiter,
   along the lines of limiters for finite volume schemes in
   [Smolarkiewicz\_1989](https://journals.ametsoc.org/mwr/article/117/11/2626/64201/Comment-on-A-Positive-Definite-Advection-Scheme).
   Such application for DG is not described in the literature.
   Not sure how effective it would be.
   For our application one would have to additionally deal with
   divergence and CFL > 1.

## Results

TODO - Once the interface with CliMA is implemented I'll add here a
  test/usage example.
It will either be the 1D-box filter showcase
  or the cone test from Light and Durran 2016.
