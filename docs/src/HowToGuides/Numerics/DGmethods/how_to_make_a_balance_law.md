# How to make a Balance law

Defining the set of solved PDEs in ClimateMachine revolve around defining a [`BalanceLaw`](@ref). A balance law solves equations of the form:

```
∂Y
-- = - ∇ • {F_{first_order}(Y) + F_{second_order}(Y, Σ) ) + S_{non_conservative}(Y, Σ)
∂t
```

Here, `Y`, `Σ`, `F_{first_order}`, `F_{second_order}`, and `S_{non_conservative}` can be thought of column vectors[^1] expressing:

 - `Y` the conservative state variables, or unknowns of the PDEs to be solved
 - `Σ` the gradients of functions of the conservative state variables
 - `F_{first-order}` contains all first-order fluxes (e.g. **not** functions of gradients of any variables)
 - `F_{second-order}` contains all second-order and higher-order fluxes (e.g. functions of gradients of any variables)
 - `S_{non_conservative}` non-conservative sources[^2]

In order to alleviate users from being concerned with the burden of spatial discretization, users must provide their own implementations of the following methods, which are computed locally at each nodal point:

## Variable name specification methods
| **Method** | Necessary | Purpose |
|:-----|:-|:----|
| [`vars_state_conservative`](@ref)         | **YES** |  specify the names of the variables in the conservative state vector, typically mass, momentum, and various tracers. |
| [`vars_state_auxiliary`](@ref)           | **YES** |  specify the names of any variables required for the balance law that aren't related to derivatives of the state variables (e.g. spatial coordinates or various integrals) or those needed to solve expensive auxiliary equations (e.g., temperature via a non-linear equation solve)     |
| [`vars_state_gradient`](@ref)         | **YES** |  specify the names of the gradients of functions of the conservative state variables. used to represent values before **and** after differentiation |
| [`vars_state_gradient_flux`](@ref)         | **YES** |  specify the names of the gradient fluxes necessary to impose Neumann boundary conditions. typically the product of a diffusivity tensor with a gradient state variable, potentially equivalent to the second-order flux for a conservative state variable |
| [`vars_integrals`](@ref)         | **NO** |  specify the names of any one-dimensional vertical integrals from **bottom to top** of the domain required for the balance law. used to represent both the integrand **and** the resulting indefinite integral |
| [`vars_reverse_integrals`](@ref)         | **NO** |  specify the names of any one-dimensional vertical integral from **top to bottom** of the domain required for the balance law. each variable here must also exist in `vars_integrals` since the reverse integral kernels use subtraction to reverse the integral instead of performing a new integral. use to represent the value before **and** after reversing direction |

## Methods to compute gradients and integrals
| **Method** |  Purpose |
|:-----|:-----|
| [`compute_gradient_argument!`](@ref) | specify how to compute the arguments to the gradients. can be functions of conservative state  and auxiliary variables. |
| [`compute_gradient_flux!`](@ref) | specify how to compute gradient fluxes. can be a functions of the gradient state, the conservative state, and auxiliary variables.|
| [`integral_load_auxiliary_state!`](@ref) | specify how to compute integrands. can be functions of the conservative state and auxiliary variables. |
| [`integral_set_auxiliary_state!`](@ref) | specify which auxiliary variables are used to store the output of the integrals. |
| [`reverse_integral_load_auxiliary_state!`](@ref) | specify auxiliary variables need their integrals reversed. |
| [`reverse_integral_set_auxiliary_state!`](@ref) | specify which auxiliary variables are used to store the output of the reversed integrals. |
| [`update_auxiliary_state!`](@ref) | perform any updates to the auxiliary variables needed at the beginning of each time-step. Can be used to solve non-linear equations, calculate integrals, and apply filters. |
| [`update_auxiliary_state_gradient!`](@ref) | same as above, but after computing gradients and gradient fluxes in case these variables are needed during the update. |


## Methods to compute fluxes and sources
| **Method** | Purpose |
|:-----|:-----|
| [`flux_first_order!`](@ref) |  specify `F_{first_order}` for each conservative state variable. can be functions of the conservative state and auxiliary variables. |
| [`flux_second_order!`](@ref)    |  specify `F_{second_order}` for each conservative state variable. can be functions of the conservative state, gradient flux state, and auxiliary variables. |
| [`source!`](@ref)            |  specify `S_{non_conservative}`  for each conservative state variable. can be functions of the conservative state, gradient flux state, and auxiliary variables. |

## Methods to compute numerical fluxes
| **Method** | Purpose |
|:-----|:-----|
| [`wavespeed`](@ref) | specify how to compute the local wavespeed if using the `RusanovNumericalFlux`. |
| [`boundary_state!`](@ref) | define exterior nodal values of the conservative state and gradient flux state used to compute the numerical boundary fluxes. |

## Methods to set initial conditions
| **Method** | Purpose |
|:-----|:-----|
| [`init_state_conservative!`](@ref) | provide initial values for the conservative state as a function of time and space. |
| [`init_state_auxiliary!`](@ref) | provide initial values for the auxiliary variables as a function of the geometry. |


## General Remarks

While `Y` can be thought of a column vector (each _row_ of which corresponds to each state variable and its prognostic equation), the _second_ function argument inside these methods behave as dictionaries, for example:

```
struct MyModel <: BalanceLaw end

function vars_state_conservative(m::MyModel, FT)
    @vars begin
        ρ::FT
        T::FT
    end
end

function source!(m::MyModel, source::Vars, args...)
    source.ρ = 1 # adds a source of 1 to RHS of ρ equation
    source.T = 1 # adds a source of 1 to RHS of T equation
end
```

All equations are marched simultaneously in time.


# Reference links

[^1]: [Column Vectors](https://en.wikipedia.org/wiki/Row_and_column_vectors)
[^2]: Note that using non-conservative sources should be a final resort, as this can leak conservation of the unknowns and lead to numerical instabilities. It is recommended to use either `F_{diffusive}` or `F_{non_diffusive}`, as these fluxes are communicated across elements[^3]
[^3]: MPI communication occurs only across elements, not within each element, where there may be many [Gauss-Lobatto][^4] points
[^4]: [Gauss-Lobatto](https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Lobatto_rules)
