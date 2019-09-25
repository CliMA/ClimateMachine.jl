struct DGModel{BL,G,NFND,NFD,GNF,AS,DS}
  balancelaw::BL
  grid::G
  numfluxnondiff::NFND
  numfluxdiff::NFD
  gradnumflux::GNF
  auxstate::AS
  diffstate::DS
end
function DGModel(balancelaw, grid, numfluxnondiff, numfluxdiff, gradnumflux;
                 auxstate=create_auxstate(balancelaw, grid),
                 diffstate=create_diffstate(balancelaw, grid))
  DGModel(balancelaw, grid, numfluxnondiff, numfluxdiff, gradnumflux, auxstate,
          diffstate)
end

function (dg::DGModel)(dQdt, Q, t; increment=false)
  bl = dg.balancelaw
  device = typeof(Q.Q) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Nfp = Nq * Nqk
  nrealelem = length(topology.realelems)

  Qvisc = dg.diffstate
  auxstate = dg.auxstate

  DFloat = eltype(Q)
  nviscstate = num_diffusive(bl, DFloat)

  lgl_weights_vec = grid.ω
  Dmat = grid.D
  vgeo = grid.vgeo
  sgeo = grid.sgeo
  vmapM = grid.vmapM
  vmapP = grid.vmapP
  elemtobndy = grid.elemtobndy
  polyorder = polynomialorder(dg.grid)

  Np = dofs_per_element(grid)

  if hasmethod(update_aux!, Tuple{typeof(dg), typeof(bl), typeof(Q),
                                  typeof(auxstate), typeof(t)})
    update_aux!(dg, bl, Q, auxstate, t)
  end

  ########################
  # Gradient Computation #
  ########################
  MPIStateArrays.start_ghost_exchange!(Q)
  MPIStateArrays.start_ghost_exchange!(auxstate)

  if nviscstate > 0

    @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
            volumeviscterms!(bl, Val(dim), Val(polyorder), Q.Q, Qvisc.Q, auxstate.Q, vgeo, t, Dmat,
                             topology.realelems))

    MPIStateArrays.finish_ghost_recv!(Q)
    MPIStateArrays.finish_ghost_recv!(auxstate)

    @launch(device, threads=Nfp, blocks=nrealelem,
            faceviscterms!(bl, Val(dim), Val(polyorder), dg.gradnumflux,
                           Q.Q, Qvisc.Q, auxstate.Q,
                           vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
                           topology.realelems))

    MPIStateArrays.start_ghost_exchange!(Qvisc)
  end

  ###################
  # RHS Computation #
  ###################
  @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
          volumerhs!(bl, Val(dim), Val(polyorder), dQdt.Q, Q.Q, Qvisc.Q, auxstate.Q,
                     vgeo, t, lgl_weights_vec, Dmat,
                     topology.realelems, increment))

  if nviscstate > 0
    MPIStateArrays.finish_ghost_recv!(Qvisc)
  else
    MPIStateArrays.finish_ghost_recv!(Q)
    MPIStateArrays.finish_ghost_recv!(auxstate)
  end

  # The main reason for this protection is not for the MPI.Waitall!, but the
  # make sure that we do not recopy data to the GPU
  nviscstate > 0 && MPIStateArrays.finish_ghost_recv!(Qvisc)
  nviscstate == 0 && MPIStateArrays.finish_ghost_recv!(Q)

  @launch(device, threads=Nfp, blocks=nrealelem,
          facerhs!(bl, Val(dim), Val(polyorder),
                   dg.numfluxnondiff,
                   dg.numfluxdiff,
                   dQdt.Q, Q.Q, Qvisc.Q,
                   auxstate.Q, vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
                   topology.realelems))

  # Just to be safe, we wait on the sends we started.
  MPIStateArrays.finish_ghost_send!(Qvisc)
  MPIStateArrays.finish_ghost_send!(Q)
end

function init_ode_state(dg::DGModel, args...; commtag=888)
  bl = dg.balancelaw
  grid = dg.grid

  state = create_state(bl, grid, commtag)

  topology = grid.topology
  # FIXME: Remove after updating CUDA
  h_vgeo = Array(grid.vgeo)
  DFloat = eltype(h_vgeo)
  Np = dofs_per_element(grid)

  auxstate = dg.auxstate
  dim = dimensionality(grid)
  polyorder = polynomialorder(grid)
  vgeo = grid.vgeo
  device = typeof(state.Q) <: Array ? CPU() : CUDA()
  nrealelem = length(topology.realelems)
  @launch(device, threads=(Np,), blocks=nrealelem,
          initstate!(bl, Val(dim), Val(polyorder), state.Q, auxstate.Q, vgeo,
                     topology.realelems, args...))
  MPIStateArrays.start_ghost_exchange!(state)
  MPIStateArrays.finish_ghost_exchange!(state)

  return state
end

"""
    dof_iteration!(dof_fun!::Function, R::MPIStateArray, disc::DGBalanceLaw,
                   Q::MPIStateArray)

Iterate over each dof to fill `R` using the `dof_fun!`. The syntax of the
`dof_fun!` is
```
dof_fun!(l_R, l_Q, l_Qvisc, l_aux)
```
where `l_R`, `l_Q`, `l_Qvisc`, and `l_aux` are of type `MArray` filled initially
with the values at a single degree of freedom. After the call the values in
`l_R` will be written back to the degree of freedom of `R`.
"""
function node_apply_aux!(f!::Function, dg::DGModel, Q::MPIStateArray, param)
  bl = dg.balancelaw

  grid = dg.grid
  topology = grid.topology

  @assert size(R)[end] == size(Q)[end] == size(dg.auxstate)[end]
  @assert size(R)[1] == size(Q)[1] == size(dg.auxstate)[1]

  dim = dimensionality(grid)
  N = polynomialorder(grid)

  Qvisc = dg.diffstate
  auxstate = dg.auxstate

  nstate = size(Q, 2)
  nviscstate = size(Qvisc, 2)
  nauxstate = size(auxstate, 2)

  nRstate = size(R, 2)

  Dmat = grid.D
  vgeo = grid.vgeo

  device = typeof(auxstate.Q) <: Array ? CPU() : CUDA()

  nelem = length(topology.elems)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  nrealelem = length(topology.realelems)

  @launch(device, threads=(Np,), blocks=nrealelem,
    knl_node_apply_aux!(bl, Val(dim), Val(N), f!, Q.Q, Qvisc.Q, auxstate.Q, topology.realelems))
end

function indefinite_stack_integral!(dg::DGModel, m::BalanceLaw,
                                    Q::MPIStateArray, auxstate::MPIStateArray,
                                    t::Real)

  device = typeof(Q.Q) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  DFloat = eltype(Q)

  vgeo = grid.vgeo
  polyorder = polynomialorder(dg.grid)

  # do integrals
  nintegrals = num_integrals(m, DFloat)
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_indefinite_stack_integral!(m, Val(dim), Val(polyorder),
                                         Val(nvertelem), Q.Q, auxstate.Q,
                                         vgeo, grid.Imat, 1:nhorzelem,
                                         Val(nintegrals)))
end

function reverse_indefinite_stack_integral!(dg::DGModel, m::BalanceLaw,
                                            auxstate::MPIStateArray, t::Real)

  device = typeof(auxstate.Q) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  DFloat = eltype(auxstate)

  vgeo = grid.vgeo
  polyorder = polynomialorder(dg.grid)

  # do integrals
  nintegrals = num_integrals(m, DFloat)
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_reverse_indefinite_stack_integral!(Val(dim), Val(polyorder),
                                                 Val(nvertelem), auxstate.Q,
                                                 1:nhorzelem,
                                                 Val(nintegrals)))
end

function nodal_update_aux!(f!, dg::DGModel, m::BalanceLaw, Q::MPIStateArray,
                           auxstate::MPIStateArray, t::Real)
  device = typeof(Q.Q) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  nrealelem = length(topology.realelems)

  polyorder = polynomialorder(dg.grid)

  Np = dofs_per_element(grid)

  ### update aux variables
  @launch(device, threads=(Np,), blocks=nrealelem,
          knl_nodal_update_aux!(m, Val(dim), Val(polyorder), f!,
                          Q.Q, auxstate.Q, t, topology.realelems))
end
