# How to define boundary conditions

```@meta
CurrentModule = ClimateMachine.BalanceLaws
```

## Discontinuous Galerkin (DG) method

Boundary conditions (BCs) are defined _weakly_ in ClimateMachine's
DG implementation. We do this by defining BCs _through_ the _fluxes_,
rather than on the prognostic variables themselves. Therefore, BCs
are enforced by prescribing fluxes (numerical fluxes, in DG terminology)
that drive the solution at the boundary to satisfy our desired BCs.

One major advantage of this implementation approach is that BCs are
handled in phase space, so we only need one implementation for all
time steppers.

In the context of the PDE that `ClimateMachine.jl` solves:

`R(Y, t) = ∂_t Yᵢ + (∇•(F₁(Y) + F₂(Y,G)))ᵢ - (S(Y,G))ᵢ = 0, G = ∇Y`,

the following functions must be defined for a
[`BalanceLaw`](@ref) to fully define BCs:

```@meta
CurrentModule = ClimateMachine.DGMethods.NumericalFluxes
```

| **Method**                                       | Internal/User-facing  |  If present | Calls to user-facing functions |
|:-----|:-----|:-----|:-----|
| [`numerical_boundary_flux_first_order!`](@ref)   | User-facing  | F₁                  | `boundary_state!(::NumericalFluxFirstOrder)`, `numerical_flux_first_order!(::NumericalFluxFirstOrder)`    |
| [`numerical_boundary_flux_second_order!`](@ref)  | User-facing  | F₂                  | `normal_boundary_flux_second_order!`[, `boundary_flux_second_order!` ,`boundary_state!(::NumericalFluxSecondOrder)`, `flux_second_order!`]  |
| [`numerical_boundary_flux_gradient!`](@ref)      | User-facing  | Non-empty G         | `boundary_state!(::NumericalFluxGradient)`, `compute_gradient_argument!` |
| [`numerical_boundary_flux_divergence!`](@ref)    | Internal     | ∇•∇ terms           | `boundary_state!(::DivNumericalPenalty)`, `numerical_flux_divergence!`  |
| [`numerical_boundary_flux_higher_order!`](@ref)  | Internal     | Higher order fluxes | `boundary_state!(::GradNumericalFlux)`, `numerical_flux_higher_order!`  |

Using DG notation, these boundary numerical fluxes are used to
force our solution to satisfy the BCs.

``
∫_D R(Y, t) l(x) dx = ∫_∂D n̂ • (f^{-} - f^{*}) l(x) dx
``

Where ``f^{-}`` is the surface value on the ``-`` state (the interior
element), and ``f^{*}`` is the prescribed numerical flux that satisfies
our BCs.

This requires us to translate our boundary conditions.
It is often desired to use the same numerical flux stencil on the interior
 as the boundary.

