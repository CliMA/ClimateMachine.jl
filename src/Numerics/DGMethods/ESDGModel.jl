using .NumericalFluxes: EntropyConservative

"""
    ESDGModel

Contain type and functor that is used to evaluated the tendency for a entropy
stable / conservative DGSEM discretization. Major fundamental difference between
this and the more vanilla DGSEM is that the first order flux derivatives in the
balance laws are evaluated using "flux-differencing". Namely, the following identities are used:
```math
    ∂x f(q(x)) = 2∂x F(q(x), q(y))|_{x = y},
    A(q(x)) ∂x q(x) = 2∂x D(q(x), q(y))|_{x = y},
```
where the numerical conservative flux `F` and numerical fluctuation flux `D`
satisfy the following consistency and symmetry properties
```math
    F(q, p) = F(p, q),
    F(q, q) = f(q),
    D(q, p) = B(q, p)(q - p),
    2B(q, q) = A(p).
```
For the scheme to be entropy stable (and not just consistent) other properties
of the numerical flux are also required. In particular, consider a balance laws
of the form
```math
    ∂t q + ∑_{j=1:d} (∂xj fj(q) + Aj(q) ∂xj q) = g(q, x, t),
```
where `q` is the state vector, `fj` is the conservative flux, and `Aj`
nonconservative variable coefficient matrix, and `g` is the production field.
Let there exists a scalar companion balance law of the form
```math
    ∂t η(q) + ∑_{j=1:d} ∂xj ζj(q) = Π(q, x, t),
    Π(q, x, t) = β(q)^T g(q, x, t),
    β(q) = ∂q η(q).
```
Then for the scheme to be entropy stable it is requires that the numerical flux
`H(q, p) = F(q, p) + D(q, p)` satisfy the following Tadmor-shuffle:
```math
    β(q)^T Hj(q, p) - β(p)^T Hj(p, q) <= ψj(q) - ψ(p),
    ψj(q) = β(q)^T fj(q) - ζj(q);
```
when the equality is satisfied the scheme is called entropy conservative. For
balance laws without a nonconservative term, `ψj` is the entropy potential.
"""
struct ESDGModel{BL, SA, VNFFO, SNFFO} <: AbstractDGModel
    "definition of the physics being considered, primary dispatch type"
    balance_law::BL
    "all the grid related information (connectivity, metric terms, etc.)"
    grid::DiscontinuousSpectralElementGrid
    "auxiliary state are quantities needed to evaluate the physics that are not
    explicitly time stepped by the ode solvers"
    state_auxiliary::SA
    "first order, two-point flux to be used for volume derivatives"
    volume_numerical_flux_first_order::VNFFO
    "first order, two-point flux to be used for surface integrals"
    surface_numerical_flux_first_order::SNFFO
end

"""
    ESDGModel(
        balance_law,
        grid;
        state_auxiliary = create_auxiliary_state(balance_law, grid),
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = EntropyConservative(),
    )

Construct a `ESDGModel` type from a given `grid` and `balance_law` using the
`volume_numerical_flux_first_order` and `surface_numerical_flux_first_order`
two-point fluxes. If the two-point fluxes satisfy the appropriate Tadmor shuffle
then semi-discrete scheme will be entropy stable (or conservative).
"""
function ESDGModel(
    balance_law,
    grid;
    state_auxiliary = create_auxiliary_state(balance_law, grid),
    volume_numerical_flux_first_order = EntropyConservative(),
    surface_numerical_flux_first_order = EntropyConservative(),
)
    ESDGModel(
        balance_law,
        grid,
        state_auxiliary,
        volume_numerical_flux_first_order,
        surface_numerical_flux_first_order,
    )
end
