"""
    BalanceLaw

An abstract type representing a PDE balance law of the form

elements for balance laws of the form

```math
q_{,t} + Σ_{i=1,...d} F_{i,i} = s
```

Subtypes `L` should define the following methods:
- `vars_state_conservative(::L)`: a tuple of symbols containing the state variables
- `vars_state_auxiliary(::L)`: a tuple of symbols containing the auxiliary variables
- `vars_state_gradient(::L)`: a tuple of symbols containing the transformed variables of which gradients are computed
- `vars_state_gradient_flux(::L)`: a tuple of symbols containing the diffusive variables
- `vars_gradient_laplacian(::L)`: a tuple of symbols containing the transformed variables of which gradients of laplacian are computed, they have to be a subset of `vars_state_gradient`
- `vars_hyperdiffusive(::L)`: a tuple of symbols containing the hyperdiffusive variables
- `flux_first_order!(::L, flux::Grad, state_conservative::Vars, state_auxiliary::Vars, t::Real)`
- `flux_second_order!(::L, flux::Grad, state_conservative::Vars, state_gradient_flux::Vars, Qhypervisc_div::Vars, state_auxiliary::Vars, t::Real)`
- `compute_gradient_argument!(::L, transformstate::Vars, state_conservative::Vars, state_auxiliary::Vars, t::Real)`: transformation of state variables to variables of which gradients are computed
- `compute_gradient_flux!(::L, state_gradient_flux::Vars, ∇transformstate::Grad, state_auxiliary::Vars, t::Real)`: transformation of gradients to the diffusive variables
- `transform_post_gradient_laplacian!(::L, Qhypervisc_div::Vars, ∇Δtransformstate::Grad, state_auxiliary::Vars, t::Real)`: transformation of laplacian gradients to the hyperdiffusive variables
- `source!(::L, source::Vars, state_conservative::Vars, diffusive::Vars, state_auxiliary::Vars, t::Real)`
- `wavespeed(::L, n⁻, state_conservative::Vars, state_auxiliary::Vars, t::Real)`
- `boundary_state!(::NumericalFluxGradient, ::L, state_conservative⁺::Vars, state_auxiliary⁺::Vars, normal⁻, state_conservative⁻::Vars, state_auxiliary⁻::Vars, bctype, t)`
- `boundary_state!(::NumericalFluxFirstOrder, ::L, state_conservative⁺::Vars, state_auxiliary⁺::Vars, normal⁻, state_conservative⁻::Vars, state_auxiliary⁻::Vars, bctype, t)`
- `boundary_state!(::NumericalFluxSecondOrder, ::L, state_conservative⁺::Vars, state_gradient_flux⁺::Vars, state_auxiliary⁺::Vars, normal⁻, state_conservative⁻::Vars, state_gradient_flux⁻::Vars, state_auxiliary⁻::Vars, bctype, t)`
- `init_state_auxiliary!(::L, state_auxiliary::Vars, coords, args...)`
- `init_state_conservative!(::L, state_conservative::Vars, state_auxiliary::Vars, coords, args...)`

"""
abstract type BalanceLaw end # PDE part

# function stubs
function vars_state_conservative end
function vars_state_auxiliary end

function vars_state_gradient end
function vars_state_gradient_flux end

vars_gradient_laplacian(::BalanceLaw, FT) = @vars()
vars_hyperdiffusive(::BalanceLaw, FT) = @vars()

vars_integrals(::BalanceLaw, FT) = @vars()
vars_reverse_integrals(::BalanceLaw, FT) = @vars()

number_state_conservative(m::BalanceLaw, FT) =
    varsize(vars_state_conservative(m, FT))
number_state_auxiliary(m::BalanceLaw, FT) = varsize(vars_state_auxiliary(m, FT))

number_state_gradient(m::BalanceLaw, FT) = varsize(vars_state_gradient(m, FT))
number_state_gradient_flux(m::BalanceLaw, FT) =
    varsize(vars_state_gradient_flux(m, FT))

num_gradient_laplacian(m::BalanceLaw, FT) =
    varsize(vars_gradient_laplacian(m, FT))
num_hyperdiffusive(m::BalanceLaw, FT) = varsize(vars_hyperdiffusive(m, FT))

num_integrals(m::BalanceLaw, FT) = varsize(vars_integrals(m, FT))
num_reverse_integrals(m::BalanceLaw, FT) =
    varsize(vars_reverse_integrals(m, FT))


function init_state_conservative! end
function init_state_auxiliary! end

function flux_first_order! end
function flux_second_order! end
function source! end

compute_gradient_argument!(::BalanceLaw, args...) = nothing
compute_gradient_flux!(::BalanceLaw, args...) = nothing
function transform_post_gradient_laplacian! end

function wavespeed end
function boundary_state! end

function update_auxiliary_state! end
function update_auxiliary_state_gradient! end

function integral_load_auxiliary_state! end
function integral_set_auxiliary_state! end
function reverse_integral_load_auxiliary_state! end
function reverse_integral_set_auxiliary_state! end


using ..Courant
"""
    calculate_dt(dg, model, Q, Courant_number, direction, t)

For a given model, compute a time step satisying the nondiffusive Courant number
`Courant_number`
"""
function calculate_dt(dg, model, Q, Courant_number, t, direction)
    Δt = one(eltype(Q))
    CFL = courant(nondiffusive_courant, dg, model, Q, Δt, t, direction)
    return Courant_number / CFL
end


function create_conservative_state(balance_law::BalanceLaw, grid)
    topology = grid.topology
    # FIXME: Remove after updating CUDA
    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    Np = dofs_per_element(grid)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    V = vars_state_conservative(balance_law, FT)
    state_conservative = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        number_state_conservative(balance_law, FT),
        length(topology.elems),
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
        weights = weights,
    )
    return state_conservative
end

function create_auxiliary_state(balance_law, grid)
    topology = grid.topology
    Np = dofs_per_element(grid)

    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    V = vars_state_auxiliary(balance_law, FT)
    state_auxiliary = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        number_state_auxiliary(balance_law, FT),
        length(topology.elems),
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
        weights = weights,
    )

    dim = dimensionality(grid)
    polyorder = polynomialorder(grid)
    vgeo = grid.vgeo
    device = typeof(state_auxiliary.data) <: Array ? CPU() : CUDA()
    nrealelem = length(topology.realelems)
    event = Event(device)
    event = kernel_init_state_auxiliary!(device, min(Np, 1024), Np * nrealelem)(
        balance_law,
        Val(dim),
        Val(polyorder),
        state_auxiliary.data,
        vgeo,
        topology.realelems,
        dependencies = (event,),
    )
    event = MPIStateArrays.begin_ghost_exchange!(
        state_auxiliary;
        dependencies = event,
    )
    event = MPIStateArrays.end_ghost_exchange!(
        state_auxiliary;
        dependencies = event,
    )
    wait(device, event)

    return state_auxiliary
end

function create_gradient_state(balance_law, grid)
    topology = grid.topology
    Np = dofs_per_element(grid)

    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    # TODO: Clean up this MPIStateArray interface...
    V = vars_state_gradient_flux(balance_law, FT)
    state_gradient_flux = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        number_state_gradient_flux(balance_law, FT),
        length(topology.elems),
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
        weights = weights,
    )

    return state_gradient_flux
end

function create_higher_order_states(balance_law, grid)
    topology = grid.topology
    Np = dofs_per_element(grid)

    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    ngradlapstate = num_gradient_laplacian(balance_law, FT)
    # TODO: Clean up this MPIStateArray interface...
    V = vars_gradient_laplacian(balance_law, FT)
    Qhypervisc_grad = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        3ngradlapstate,
        length(topology.elems),
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
        weights = weights,
    )

    V = vars_hyperdiffusive(balance_law, FT)
    Qhypervisc_div = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        ngradlapstate,
        length(topology.elems),
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
        weights = weights,
    )
    return Qhypervisc_grad, Qhypervisc_div
end
