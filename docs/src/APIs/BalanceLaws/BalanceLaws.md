# Balance Laws

```@meta
CurrentModule = ClimateMachine.BalanceLaws
```

## The balance law

```@docs
BalanceLaw
```

## Variable specification methods

```@docs
vars_state_conservative
vars_state_auxiliary
vars_state_gradient
vars_integrals
vars_reverse_integrals
vars_state_gradient_flux
```

## Initial condition methods

```@docs
init_state_conservative!
init_state_auxiliary!
```

## Source term kernels

```@docs
flux_first_order!
flux_second_order!
source!
```

## Integral kernels

```@docs
indefinite_stack_integral!
reverse_indefinite_stack_integral!
integral_load_auxiliary_state!
integral_set_auxiliary_state!
reverse_integral_load_auxiliary_state!
reverse_integral_set_auxiliary_state!
```

## Gradient/Laplacian kernels

```@docs
compute_gradient_flux!
compute_gradient_argument!
transform_post_gradient_laplacian!
```

## Auxiliary kernels

```@docs
wavespeed
boundary_state!
update_auxiliary_state!
update_auxiliary_state_gradient!
nodal_update_auxiliary_state!
```
