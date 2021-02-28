using .NumericalFluxes: EntropyConservative

include("ESDGModel_kernels.jl")

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
struct ESDGModel{BL, SA, VNFFO, SNFFO} <:
       AbstractDGModel{BL, DiscontinuousSpectralElementGrid, SA}
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
        state_auxiliary = create_state(balance_law, grid, Auxiliary()),
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
    # FIXME: this probably should be done differently
    state_auxiliary =
    (aux=create_state(balance_law, grid, Auxiliary());
     init_state(aux, balance_law, grid, EveryDirection(), Auxiliary())),
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

"""
    (esdg::ESDGModel)(
        tendency::MPIStateArray,
        state_prognostic::MPIStateArray,
        param::Nothing,
        t,
        α = true,
        β = false,
    )

Compute the entropy stable tendency from the model `esdg`.

    tendency .= α .* dQdt(state_prognostic, param, t) .+ β .* tendency
"""
function (esdg::ESDGModel)(
    tendency::MPIStateArray,
    state_prognostic::MPIStateArray,
    ::Nothing,
    t,
    α = true,
    β = false,
)
    device = array_device(state_prognostic)

    balance_law = esdg.balance_law
    @assert number_states(balance_law, GradientFlux(), Int) == 0

    grid = esdg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    # XXX: Needs updating for multiple polynomial orders
    N = polynomialorders(grid)
    # Currently only support single polynomial order
    @assert all(N[1] .== N)
    N = N[1]

    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk

    nrealelem = length(topology.realelems)

    state_auxiliary = esdg.state_auxiliary

    # XXX: When we do stacked meshes and IMEX this will change
    communicate = true

    exchange_state_prognostic = NoneEvent()

    comp_stream = Event(device)

    ########################
    # tendency Computation #
    ########################
    if communicate
        exchange_state_prognostic = MPIStateArrays.begin_ghost_exchange!(
            state_prognostic;
            dependencies = comp_stream,
        )
    end

    # volume tendency
    volume_tendency_kernel = :naive_split
    if volume_tendency_kernel == :orig
      comp_stream = esdg_volume_tendency!(device, (Nq, Nq))(
          balance_law,
          Val(dim),
          Val(N),
          esdg.volume_numerical_flux_first_order,
          tendency.data,
          state_prognostic.data,
          state_auxiliary.data,
          grid.vgeo,
          grid.D[1],
          α,
          β,
          ndrange = (nrealelem * Nq, Nq),
          dependencies = (comp_stream,),
      )
    elseif volume_tendency_kernel == :naive
      comp_stream = esdg_volume_tendency_naive!(device, (Nq, Nq, Nqk))(
          balance_law,
          Val(dim),
          Val(N),
          esdg.volume_numerical_flux_first_order,
          tendency.data,
          state_prognostic.data,
          state_auxiliary.data,
          grid.vgeo,
          grid.D[1],
          α,
          β,
          ndrange = (nrealelem * Nq, Nq, Nqk),
          dependencies = (comp_stream,),
      )
    elseif volume_tendency_kernel == :naive_split
      comp_stream = esdg_volume_tendency_naive_ξ1ξ2!(device, (Nq, Nq, Nqk))(
          balance_law,
          Val(dim),
          Val(N),
          esdg.volume_numerical_flux_first_order,
          tendency.data,
          state_prognostic.data,
          state_auxiliary.data,
          grid.vgeo,
          grid.D[1],
          α,
          β,
          ndrange = (nrealelem * Nq, Nq, Nqk),
          dependencies = (comp_stream,),
      )
      comp_stream = esdg_volume_tendency_naive_ξ3!(device, (Nq, Nq, Nqk))(
          balance_law,
          Val(dim),
          Val(N),
          esdg.volume_numerical_flux_first_order,
          tendency.data,
          state_prognostic.data,
          state_auxiliary.data,
          grid.vgeo,
          grid.D[1],
          α,
          β,
          ndrange = (nrealelem * Nq, Nq, Nqk),
          dependencies = (comp_stream,),
      )
    end

    # non-mirror surface tendency
    comp_stream = interface_tendency!(device, (Nfp,))(
        balance_law,
        Val(dim),
        Val(N),
        EveryDirection(),
        esdg.surface_numerical_flux_first_order,
        nothing,
        tendency.data,
        state_prognostic.data,
        nothing,
        nothing,
        state_auxiliary.data,
        grid.vgeo,
        grid.sgeo,
        t,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.interiorelems,
        α;
        ndrange = Nfp * length(grid.interiorelems),
        dependencies = (comp_stream,),
    )

    if communicate
        exchange_state_prognostic = MPIStateArrays.end_ghost_exchange!(
            state_prognostic;
            dependencies = exchange_state_prognostic,
        )
    end

    # mirror surface tendency
    comp_stream = interface_tendency!(device, (Nfp,))(
        balance_law,
        Val(dim),
        Val(N),
        EveryDirection(),
        esdg.surface_numerical_flux_first_order,
        nothing,
        tendency.data,
        state_prognostic.data,
        nothing,
        nothing,
        state_auxiliary.data,
        grid.vgeo,
        grid.sgeo,
        t,
        grid.vmap⁻,
        grid.vmap⁺,
        grid.elemtobndy,
        grid.exteriorelems,
        α;
        ndrange = Nfp * length(grid.exteriorelems),
        dependencies = (comp_stream, exchange_state_prognostic),
    )

    comp_stream = launch_drag_source!(esdg, tendency, state_prognostic, t,
                                      dependencies=comp_stream)

    # The synchronization here through a device event prevents CuArray based and
    # other default stream kernels from launching before the work scheduled in
    # this function is finished.
    wait(device, comp_stream)
end

function launch_drag_source!(dg, tendency, state_prognostic, t; dependencies)
    FT = eltype(state_prognostic)
    info = basic_launch_info(dg)

    Nq1 = info.Nq
    Nqj = info.dim == 2 ? 1 : info.Nq
    comp_stream = dependencies

    topology = dg.grid.topology
    
    elems = topology.elems
    nelem = length(elems)
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)
    
    comp_stream = kernel_drag_source!(info.device, (Nq1, Nqj))(
        dg.balance_law,
        Val(info.dim),
        Val(info.N),
        Val(nvertelem),
        tendency.data,
        state_prognostic.data,
        dg.state_auxiliary.data,
        horzelems;
        ndrange = (length(horzelems) * Nq1, Nqj),
        dependencies = (comp_stream,),
    )
    return comp_stream
end
