# DG Balance Law Method

```@meta
CurrentModule = ClimateMachine.DGMethods
```

## The balance law

```@docs
BalanceLaw
```

## Continuous Balance Law Formulation

to be filled

## Discontinuous Galerkin Method Formulation

to be filled

## Examples

!!! attribution
    The style of examples we use here is heavily inspired by
    [`JuAFEM.jl`](https://github.com/KristofferC/JuAFEM.jl)

to be filled

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
```
