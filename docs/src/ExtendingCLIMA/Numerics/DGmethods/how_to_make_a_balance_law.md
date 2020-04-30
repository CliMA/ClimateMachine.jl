# How to make a Balance law

Defining the set of solved PDEs in CLIMA revolve around defining a [`BalanceLaw`](@ref). A balance law solves equations of the form:

```
∂Y
-- = - ∇ • {F_{first_order}(Y) + F_{second_order}(Y, Σ) ) + S_{non_conservative}(Y, Σ)
∂t
```

Here, `Y`, `Σ`, `F_{first_order}`, `F_{second_order}`, , and `S_{non_conservative}` can be thought of column vectors[^1] expressing:

 - `Y` the state variables, or unknowns of the PDEs to be solved
 - `Σ` gradients of variables
 - `F_{diffusive}` is non-diffusive fluxes (are **not** functions of gradients of any variables)
 - `F_{non_diffusive}` is diffusive fluxes (are functions of gradients of any variables)
 - `S_{non_conservative}` non-conservative sources[^2]

In order to alleviate users from being concerned with the burden of spatial discretization, users must provide their own implementations of the following methods:

## Variable name specification methods
| **Method** | Purpose |
|:-----|:-----|
| [`vars_state_conservative`](@ref)         |  specify the names of the unknowns |
| [`vars_state_auxiliary`](@ref)           |  specify variable names needed to solve expensive auxiliary equations (e.g., temperature via a non-linear equation solve)     |

## Methods to compute RHS source terms / BCs
| **Method** | Purpose |
|:-----|:-----|
| [`flux_first_order!`](@ref) |  specify `F_{first_order}`       |
| [`flux_second_order!`](@ref)    |  specify `F_{second_order}`           |
| [`source!`](@ref)            |  specify `S_{non_conservative}`    |


While `Y` can be thought of a column vector (each _row_ of which corresponding to each equation), the _second_ function argument inside these methods behave as dictionaries, for example:

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
