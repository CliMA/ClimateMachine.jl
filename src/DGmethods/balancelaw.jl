"""
    BalanceLaw

An abstract type representing a PDE balance law of the form

elements for balance laws of the form

```math
q_{,t} + Σ_{i=1,...d} F_{i,i} = s
```

Subtypes `L` should define the following methods:
- `vars_aux(::L)`: a tuple of symbols containing the auxiliary variables
- `vars_state(::L)`: a tuple of symbols containing the state variables
- `vars_gradient(::L)`: a tuple of symbols containing the transformed variables of which gradients are computed
- `vars_gradient_laplacian(::L)`: a tuple of symbols containing the transformed variables of which gradients of laplacian are computed, they have to be a subset of `vars_gradient`
- `vars_diffusive(::L)`: a tuple of symbols containing the diffusive variables
- `vars_hyperdiffusive(::L)`: a tuple of symbols containing the hyperdiffusive variables
- `flux_nondiffusive!(::L, flux::Grad, state::State, auxstate::State, t::Real)`
- `flux_diffusive!(::L, flux::Grad, state::State, diffstate::State, hyperdiffstate::State, auxstate::State, t::Real)`
- `gradvariables!(::L, transformstate::State, state::State, auxstate::State, t::Real)`: transformation of state variables to variables of which gradients are computed
- `diffusive!(::L, diffstate::State, ∇transformstate::Grad, auxstate::State, t::Real)`: transformation of gradients to the diffusive variables
- `hyperdiffusive!(::L, hyperdiffstate::State, ∇Δtransformstate::Grad, auxstate::State, t::Real)`: transformation of laplacian gradients to the hyperdiffusive variables
- `source!(::L, source::State, state::State, diffusive::Vars, auxstate::State, t::Real)`
- `wavespeed(::L, n⁻, state::State, aux::State, t::Real)`
- `boundary_state!(::NumericalFluxGradient, ::L, state⁺::State, aux⁺::State, normal⁻, state⁻::State, aux⁻::State, bctype, t)`
- `boundary_state!(::NumericalFluxNonDiffusive, ::L, state⁺::State, aux⁺::State, normal⁻, state⁻::State, aux⁻::State, bctype, t)`
- `boundary_state!(::NumericalFluxDiffusive, ::L, state⁺::State, diff⁺::State, aux⁺::State, normal⁻, state⁻::State, diff⁻::State, aux⁻::State, bctype, t)`
- `init_aux!(::L, aux::State, coords, args...)`
- `init_state!(::L, state::State, aux::State, coords, args...)`

"""
abstract type BalanceLaw end # PDE part

# function stubs
function vars_state end
function vars_aux end
function vars_gradient end
vars_gradient_laplacian(::BalanceLaw, FT) = @vars()
function vars_diffusive end
vars_hyperdiffusive(::BalanceLaw, FT) = @vars()
vars_integrals(::BalanceLaw, FT) = @vars()
vars_reverse_integrals(::BalanceLaw, FT) = @vars()

num_aux(m::BalanceLaw, FT) = varsize(vars_aux(m,FT))
num_state(m::BalanceLaw, FT) = varsize(vars_state(m,FT)) # nstate
num_gradient(m::BalanceLaw, FT) = varsize(vars_gradient(m,FT))  # number_gradient_states
num_gradient_laplacian(m::BalanceLaw, FT) = varsize(vars_gradient_laplacian(m,FT))
num_diffusive(m::BalanceLaw, FT) = varsize(vars_diffusive(m,FT)) # number_viscous_states
num_hyperdiffusive(m::BalanceLaw, FT) = varsize(vars_hyperdiffusive(m,FT))
num_integrals(m::BalanceLaw, FT) = varsize(vars_integrals(m,FT))
num_reverse_integrals(m::BalanceLaw, FT) = varsize(vars_reverse_integrals(m,FT))

function update_aux! end
function integral_load_aux! end
function integral_set_aux! end
function reverse_integral_load_aux! end
function reverse_integral_set_aux! end
function flux_nondiffusive! end
function flux_diffusive! end
function gradvariables! end
function diffusive! end
function hyperdiffusive! end
function source! end 
function wavespeed end
function boundary_state! end
function init_aux! end
function init_state! end

function calculate_dt end

function create_state(bl::BalanceLaw, grid, commtag)
  topology = grid.topology
  # FIXME: Remove after updating CUDA
  h_vgeo = Array(grid.vgeo)
  FT = eltype(h_vgeo)
  Np = dofs_per_element(grid)
  DA = arraytype(grid)

  weights = view(h_vgeo, :, grid.Mid, :)
  weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

  state = MPIStateArray{FT}(topology.mpicomm, DA, Np, num_state(bl,FT),
                            length(topology.elems),
                            realelems=topology.realelems,
                            ghostelems=topology.ghostelems,
                            vmaprecv=grid.vmaprecv, vmapsend=grid.vmapsend,
                            nabrtorank=topology.nabrtorank,
                            nabrtovmaprecv=grid.nabrtovmaprecv,
                            nabrtovmapsend=grid.nabrtovmapsend,
                            weights=weights, commtag=commtag)
  return state
end

function create_auxstate(bl, grid, commtag=222)
  topology = grid.topology
  Np = dofs_per_element(grid)

  h_vgeo = Array(grid.vgeo)
  FT = eltype(h_vgeo)
  DA = arraytype(grid)

  weights = view(h_vgeo, :, grid.Mid, :)
  weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

  auxstate = MPIStateArray{FT}(topology.mpicomm, DA, Np, num_aux(bl,FT),
                               length(topology.elems),
                               realelems=topology.realelems,
                               ghostelems=topology.ghostelems,
                               vmaprecv=grid.vmaprecv,
                               vmapsend=grid.vmapsend,
                               nabrtorank=topology.nabrtorank,
                               nabrtovmaprecv=grid.nabrtovmaprecv,
                               nabrtovmapsend=grid.nabrtovmapsend,
                               weights=weights, commtag=commtag)

  dim = dimensionality(grid)
  polyorder = polynomialorder(grid)
  vgeo = grid.vgeo
  device = typeof(auxstate.data) <: Array ? CPU() : CUDA()
  nrealelem = length(topology.realelems)
  @launch(device, threads=(Np,), blocks=nrealelem,
          initauxstate!(bl, Val(dim), Val(polyorder), auxstate.data, vgeo, topology.realelems))
  MPIStateArrays.start_ghost_exchange!(auxstate)
  MPIStateArrays.finish_ghost_exchange!(auxstate)

  return auxstate
end

function create_diffstate(bl, grid, commtag=111)
  topology = grid.topology
  Np = dofs_per_element(grid)

  h_vgeo = Array(grid.vgeo)
  FT = eltype(h_vgeo)
  DA = arraytype(grid)

  weights = view(h_vgeo, :, grid.Mid, :)
  weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

  # TODO: Clean up this MPIStateArray interface...
  diffstate = MPIStateArray{FT}(topology.mpicomm, DA, Np, num_diffusive(bl,FT),
                                length(topology.elems),
                                realelems=topology.realelems,
                                ghostelems=topology.ghostelems,
                                vmaprecv=grid.vmaprecv,
                                vmapsend=grid.vmapsend,
                                nabrtorank=topology.nabrtorank,
                                nabrtovmaprecv=grid.nabrtovmaprecv,
                                nabrtovmapsend=grid.nabrtovmapsend,
                                weights=weights, commtag=commtag)

  return diffstate
end

function create_hyperdiffstate(bl, grid, commtag=333)
  topology = grid.topology
  Np = dofs_per_element(grid)

  h_vgeo = Array(grid.vgeo)
  FT = eltype(h_vgeo)
  DA = arraytype(grid)

  weights = view(h_vgeo, :, grid.Mid, :)
  weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

  ngradlapstate = num_gradient_laplacian(bl,FT)
  # TODO: Clean up this MPIStateArray interface...
  Qhypervisc_grad = MPIStateArray{FT}(topology.mpicomm, DA, Np, 3ngradlapstate,
                                      length(topology.elems),
                                      realelems=topology.realelems,
                                      ghostelems=topology.ghostelems,
                                      vmaprecv=grid.vmaprecv,
                                      vmapsend=grid.vmapsend,
                                      nabrtorank=topology.nabrtorank,
                                      nabrtovmaprecv=grid.nabrtovmaprecv,
                                      nabrtovmapsend=grid.nabrtovmapsend,
                                      weights=weights, commtag=commtag)

  Qhypervisc_div = MPIStateArray{FT}(topology.mpicomm, DA, Np, ngradlapstate,
                                     length(topology.elems),
                                     realelems=topology.realelems,
                                     ghostelems=topology.ghostelems,
                                     vmaprecv=grid.vmaprecv,
                                     vmapsend=grid.vmapsend,
                                     nabrtorank=topology.nabrtorank,
                                     nabrtovmaprecv=grid.nabrtovmaprecv,
                                     nabrtovmapsend=grid.nabrtovmapsend,
                                     weights=weights, commtag=commtag + 111)
  return Qhypervisc_grad, Qhypervisc_div
end
