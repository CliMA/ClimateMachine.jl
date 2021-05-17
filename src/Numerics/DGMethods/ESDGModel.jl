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
struct ESDGModel{BL, SA, VNFFO, SNFFO} <: SpaceDiscretization
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
    state_auxiliary = (
        aux = create_state(balance_law, grid, Auxiliary());
        init_state(aux, balance_law, grid, EveryDirection(), Auxiliary())
    ),
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

    balance_law = esdg.balance_law
    @assert number_states(balance_law, GradientFlux(), Int) == 0

    grid = esdg.grid
    topology = grid.topology

    # Currently only support two polynomial orders
    info = basic_launch_info(esdg)

    device = info.device # array_device(state_prognostic)
    workgroup = (info.Nq[1], info.Nq[2], info.Nqk)
    ndrange = (info.Nq[1] * info.nrealelem, info.Nq[2], info.Nqk)
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
    comp_stream = esdg_volume_tendency!(device, workgroup)(
          balance_law,
          Val(1),
          Val(info),
          esdg.volume_numerical_flux_first_order,
          tendency.data,
          state_prognostic.data,
          state_auxiliary.data,
          grid.vgeo,
          grid.D[1],
          α,
          β,
          true, # add_source
          ndrange = ndrange,
          dependencies = (comp_stream,),
      )
      comp_stream = esdg_volume_tendency!(device, workgroup)(
          balance_law,
          Val(2),
          Val(info),
          esdg.volume_numerical_flux_first_order,
          tendency.data,
          state_prognostic.data,
          state_auxiliary.data,
          grid.vgeo,
          grid.D[2],
          α,
          true,
          ndrange = ndrange,
          dependencies = (comp_stream,),
      )
      if info.dim == 3
        comp_stream = esdg_volume_tendency!(device, workgroup)(
            balance_law,
            Val(3),
            Val(info),
            esdg.volume_numerical_flux_first_order,
            tendency.data,
            state_prognostic.data,
            state_auxiliary.data,
            grid.vgeo,
            grid.D[3],
            α,
            true,
            ndrange = ndrange,
            dependencies = (comp_stream,),
        )
    end
    
    # interfaces: Horizontal => Nfp_v and Vertical => Nfp_h
    # mirror surface tendency: interior
    Nfp = info.Nfp_v
    ndrange = Nfp * info.ninteriorelem
    comp_stream = dgsem_interface_tendency!(device, (Nfp,))(
        balance_law,
        Val(info),
        HorizontalDirection(),
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
        ndrange = ndrange,
        dependencies = (comp_stream,),
    )

    Nfp = info.Nfp_h
    ndrange = Nfp * info.ninteriorelem
    comp_stream = dgsem_interface_tendency!(device, (Nfp,))(
        balance_law,
        Val(info),
        VerticalDirection(),
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
        ndrange = ndrange,
        dependencies = (comp_stream,),
    )

    if communicate
        exchange_state_prognostic = MPIStateArrays.end_ghost_exchange!(
            state_prognostic;
            dependencies = exchange_state_prognostic,
        )
    end

    # mirror surface tendency: exterior
    Nfp = info.Nfp_v
    ndrange = Nfp * info.nexteriorelem
    comp_stream = dgsem_interface_tendency!(device, (Nfp,))(
        balance_law,
        Val(info),
        HorizontalDirection(),
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
        ndrange = ndrange,
        dependencies = (comp_stream, exchange_state_prognostic),
    )

    Nfp = info.Nfp_h
    ndrange = Nfp * info.nexteriorelem
    comp_stream = dgsem_interface_tendency!(device, (Nfp,))(
        balance_law,
        Val(info),
        VerticalDirection(),
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
        ndrange = ndrange,
        dependencies = (comp_stream, exchange_state_prognostic),
    )

    # The synchronization here through a device event prevents CuArray based and
    # other default stream kernels from launching before the work scheduled in
    # this function is finished.
    wait(device, comp_stream)
end


## Vertical ESDG model VESDG

"""
    VESDGModel

Contain type and functor that is used to evaluated the tendency for a entropy
stable / conservative vertical only DGSEM discretization. Major fundamental difference between
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
struct VESDGModel{BL, SA, VNFFO, SNFFO, DIR} <: SpaceDiscretization
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
    direction::DIR
end

"""
    VESDGModel(
        balance_law,
        grid;
        state_auxiliary = create_state(balance_law, grid, Auxiliary()),
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = EntropyConservative(),
    )

Construct a `VESDGModel` type from a given `grid` and `balance_law` using the
`volume_numerical_flux_first_order` and `surface_numerical_flux_first_order`
two-point fluxes. If the two-point fluxes satisfy the appropriate Tadmor shuffle
then semi-discrete scheme will be entropy stable (or conservative).
"""
function VESDGModel(
    balance_law,
    grid;
    # FIXME: this probably should be done differently
    state_auxiliary = (
        aux = create_state(balance_law, grid, Auxiliary());
        init_state(aux, balance_law, grid, EveryDirection(), Auxiliary())
    ),
    volume_numerical_flux_first_order = EntropyConservative(),
    surface_numerical_flux_first_order = EntropyConservative(),
    direction = VerticalDirection(),
)

    VESDGModel(
        balance_law,
        grid,
        state_auxiliary,
        volume_numerical_flux_first_order,
        surface_numerical_flux_first_order,
        VerticalDirection(),
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
function (esdg::VESDGModel)(
    tendency::MPIStateArray,
    state_prognostic::MPIStateArray,
    ::Nothing,
    t,
    α = true,
    β = false,
)


    balance_law = esdg.balance_law
    @assert number_states(balance_law, GradientFlux(), Int) == 0

    grid = esdg.grid
    topology = grid.topology

    # Currently only support two polynomial orders
    info = basic_launch_info(esdg)

    device = info.device # array_device(state_prognostic)
    workgroup = (info.Nq[1], info.Nq[2], info.Nqk)
    ndrange = (info.Nq[1] * info.nrealelem, info.Nq[2], info.Nqk)
    nrealelem = length(topology.realelems)

    state_auxiliary = esdg.state_auxiliary

    # XXX: When we do stacked meshes and IMEX this will change
    communicate = false

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

    # volume tendency, ξ³
    comp_stream = esdg_volume_tendency!(device, workgroup)(
        balance_law,
        Val(3),
        Val(info),
        esdg.volume_numerical_flux_first_order,
        tendency.data,
        state_prognostic.data,
        state_auxiliary.data,
        grid.vgeo,
        grid.D[3],
        α,
        β,
        ndrange = ndrange,
        dependencies = (comp_stream,),
    )
     
    # interfaces: Horizontal => Nfp_v and Vertical => Nfp_h
    # mirror surface tendency: interior
    
    Nfp = info.Nfp_h
    ndrange = Nfp * info.ninteriorelem
    comp_stream = dgsem_interface_tendency!(device, (Nfp,))(
        balance_law,
        Val(info),
        VerticalDirection(),
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
        ndrange = ndrange,
        dependencies = (comp_stream,),
    )
    
    if communicate
        exchange_state_prognostic = MPIStateArrays.end_ghost_exchange!(
            state_prognostic;
            dependencies = exchange_state_prognostic,
        )
    end

    # mirror surface tendency: exterior
    
    Nfp = info.Nfp_h
    ndrange = Nfp * info.nexteriorelem
    comp_stream = dgsem_interface_tendency!(device, (Nfp,))(
        balance_law,
        Val(info),
        VerticalDirection(),
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
        ndrange = ndrange,
        dependencies = (comp_stream, exchange_state_prognostic),
    )
    
    # The synchronization here through a device event prevents CuArray based and
    # other default stream kernels from launching before the work scheduled in
    # this function is finished.
    wait(device, comp_stream)

end