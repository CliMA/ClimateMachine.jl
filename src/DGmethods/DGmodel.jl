abstract type Direction end
struct EveryDirection <: Direction end
struct HorizontalDirection <: Direction end
struct VerticalDirection <: Direction end

struct DGModel{BL,G,NFND,NFD,GNF,AS,DS,D}
  balancelaw::BL
  grid::G
  numfluxnondiff::NFND
  numfluxdiff::NFD
  gradnumflux::GNF
  auxstate::AS
  diffstate::DS
  direction::D
end
function DGModel(balancelaw, grid, numfluxnondiff, numfluxdiff, gradnumflux;
                 auxstate=create_auxstate(balancelaw, grid),
                 diffstate=create_diffstate(balancelaw, grid),
                 direction=EveryDirection())
  DGModel(balancelaw, grid, numfluxnondiff, numfluxdiff, gradnumflux, auxstate,
          diffstate, direction)
end

function (dg::DGModel)(dQdt, Q, ::Nothing, t; increment=false)
  bl = dg.balancelaw
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

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

  FT = eltype(Q)
  nviscstate = num_diffusive(bl, FT)

  lgl_weights_vec = grid.ω
  Dmat = grid.D
  vgeo = grid.vgeo
  sgeo = grid.sgeo
  vmapM = grid.vmapM
  vmapP = grid.vmapP
  elemtobndy = grid.elemtobndy
  polyorder = polynomialorder(dg.grid)

  Np = dofs_per_element(grid)

  communicate = !(isstacked(topology) &&
                  typeof(dg.direction) <: VerticalDirection)

  if hasmethod(update_aux!, Tuple{typeof(dg), typeof(bl), typeof(Q),
                                  typeof(auxstate), typeof(t)})
    update_aux!(dg, bl, Q, auxstate, t)
  end

  ########################
  # Gradient Computation #
  ########################
  if communicate
    MPIStateArrays.start_ghost_exchange!(Q)
    MPIStateArrays.start_ghost_exchange!(auxstate)
  end

  if nviscstate > 0

    @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
            volumeviscterms!(bl, Val(dim), Val(polyorder), dg.direction, Q.data,
                             Qvisc.data, auxstate.data, vgeo, t, Dmat,
                             topology.realelems))

    if communicate
      MPIStateArrays.finish_ghost_recv!(Q)
      MPIStateArrays.finish_ghost_recv!(auxstate)
    end

    @launch(device, threads=Nfp, blocks=nrealelem,
            faceviscterms!(bl, Val(dim), Val(polyorder), dg.direction,
                           dg.gradnumflux, Q.data, Qvisc.data, auxstate.data,
                           vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
                           topology.realelems))

    communicate && MPIStateArrays.start_ghost_exchange!(Qvisc)
  end

  ###################
  # RHS Computation #
  ###################
  @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
          volumerhs!(bl, Val(dim), Val(polyorder), dg.direction, dQdt.data,
                     Q.data, Qvisc.data, auxstate.data, vgeo, t,
                     lgl_weights_vec, Dmat, topology.realelems, increment))

  if communicate
    if nviscstate > 0
      MPIStateArrays.finish_ghost_recv!(Qvisc)
    else
      MPIStateArrays.finish_ghost_recv!(Q)
      MPIStateArrays.finish_ghost_recv!(auxstate)
    end
  end

  @launch(device, threads=Nfp, blocks=nrealelem,
          facerhs!(bl, Val(dim), Val(polyorder), dg.direction,
                   dg.numfluxnondiff,
                   dg.numfluxdiff,
                   dQdt.data, Q.data, Qvisc.data,
                   auxstate.data, vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
                   topology.realelems))

  # Just to be safe, we wait on the sends we started.
  if communicate
    MPIStateArrays.finish_ghost_send!(Qvisc)
    MPIStateArrays.finish_ghost_send!(Q)
  end
end

function init_ode_state(dg::DGModel, args...;
                        device=arraytype(dg.grid) <: Array ? CPU() : CUDA(),
                        commtag=888)
  array_device = arraytype(dg.grid) <: Array ? CPU() : CUDA()
  @assert device == CPU() || device == array_device

  bl = dg.balancelaw
  grid = dg.grid

  state = create_state(bl, grid, commtag)

  topology = grid.topology
  Np = dofs_per_element(grid)

  auxstate = dg.auxstate
  dim = dimensionality(grid)
  polyorder = polynomialorder(grid)
  vgeo = grid.vgeo
  nrealelem = length(topology.realelems)

  if device == array_device
    @launch(device, threads=(Np,), blocks=nrealelem,
            initstate!(bl, Val(dim), Val(polyorder), state.data, auxstate.data, vgeo,
                     topology.realelems, args...))
  else
    h_vgeo = Array(vgeo)
    h_state = similar(state, Array)
    h_auxstate = similar(auxstate, Array)
    h_auxstate .= auxstate
    @launch(device, threads=(Np,), blocks=nrealelem,
      initstate!(bl, Val(dim), Val(polyorder), h_state.data, h_auxstate.data, h_vgeo,
          topology.realelems, args...))
    state .= h_state
  end  

  MPIStateArrays.start_ghost_exchange!(state)
  MPIStateArrays.finish_ghost_exchange!(state)

  return state
end

function indefinite_stack_integral!(dg::DGModel, m::BalanceLaw,
                                    Q::MPIStateArray, auxstate::MPIStateArray,
                                    t::Real)

  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  FT = eltype(Q)

  vgeo = grid.vgeo
  polyorder = polynomialorder(dg.grid)

  # do integrals
  nintegrals = num_integrals(m, FT)
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_indefinite_stack_integral!(m, Val(dim), Val(polyorder),
                                         Val(nvertelem), Q.data, auxstate.data,
                                         vgeo, grid.Imat, 1:nhorzelem,
                                         Val(nintegrals)))
end

function reverse_indefinite_stack_integral!(dg::DGModel, m::BalanceLaw,
                                            auxstate::MPIStateArray, t::Real)

  device = typeof(auxstate.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  FT = eltype(auxstate)

  vgeo = grid.vgeo
  polyorder = polynomialorder(dg.grid)

  # do integrals
  nintegrals = num_integrals(m, FT)
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_reverse_indefinite_stack_integral!(Val(dim), Val(polyorder),
                                                 Val(nvertelem), auxstate.data,
                                                 1:nhorzelem,
                                                 Val(nintegrals)))
end

function nodal_update_aux!(f!, dg::DGModel, m::BalanceLaw, Q::MPIStateArray,
                           auxstate::MPIStateArray, t::Real)
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

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
                          Q.data, auxstate.data, t, topology.realelems))
end

function MPIStateArrays.MPIStateArray(dg::DGModel, commtag=888)
  bl = dg.balancelaw
  grid = dg.grid

  state = create_state(bl, grid, commtag)

  return state
end

function banded_matrix(dg::DGModel, Q::MPIStateArray = MPIStateArray(dg),
                       dQ::MPIStateArray = MPIStateArray(dg);
                       single_column = false)
  banded_matrix((dQ, Q) -> dg(dQ, Q, nothing, 0; increment=false),
                dg, Q, dQ; single_column=single_column)
end

function banded_matrix(f!::Function, dg::DGModel,
                       Q::MPIStateArray = MPIStateArray(dg),
                       dQ::MPIStateArray = MPIStateArray(dg);
                       single_column = false)
  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology
  @assert isstacked(topology)
  @assert typeof(dg.direction) <: VerticalDirection

  FT = eltype(Q.data)
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  nstate = num_state(bl, FT)
  N = polynomialorder(grid)
  Nq = N + 1

  # p is lower bandwidth
  # q is upper bandwidth
  eband = num_diffusive(bl, FT) == 0 ? 1 : 2
  p = q = nstate * Nq * eband - 1

  nrealelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nrealelem, nvertelem)

  dim = dimensionality(grid)

  Nqj = dim == 2 ? 1 : Nq

  # first horizontal DOF index
  # second horizontal DOF index
  # band index -q:p
  # vertical DOF index
  # horizontal element index
  A = if single_column
    similar(Q.data, p + q + 1, Nq * nstate * nvertelem)
  else
    similar(Q.data, Nq, Nqj, p + q + 1, Nq * nstate * nvertelem,
            nhorzelem)
  end
  fill!(A, zero(FT))

  # loop through all DOFs in a column and compute the matrix column
  for ev = 1:nvertelem
    for s = 1:nstate
      for k = 1:Nq
        # Set a single 1 per column and rest 0
        @launch(device, threads=(Nq, Nqj, Nq), blocks=(nvertelem, nhorzelem),
                knl_set_banded_data!(bl, Val(dim), Val(N), Val(nvertelem),
                                     Q.data, k, s, ev, 1:nhorzelem,
                                     1:nvertelem))

        # Get the matrix column
        f!(dQ, Q)

        # Store the banded matrix
        @launch(device, threads=(Nq, Nqj, Nq),
                blocks=(2 * eband + 1, nhorzelem),
                knl_set_banded_matrix!(bl, Val(dim), Val(N), Val(nvertelem),
                                       Val(p), Val(q), Val(2eband),
                                       A, dQ.data, k, s, ev, 1:nhorzelem,
                                       -eband:eband))
      end
    end
  end
  A
end

function banded_matrix_vector_product!(dg::DGModel, A, dQ::MPIStateArray,
                                       Q::MPIStateArray)
  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology
  @assert isstacked(topology)
  @assert typeof(dg.direction) <: VerticalDirection

  FT = eltype(Q.data)
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  eband = num_diffusive(bl, FT) == 0 ? 1 : 2
  nstate = num_state(bl, FT)
  N = polynomialorder(grid)
  Nq = N + 1
  p = q = nstate * Nq * eband - 1

  nrealelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nrealelem, nvertelem)

  dim = dimensionality(grid)

  Nqj = dim == 2 ? 1 : Nq

  @launch(device, threads=(Nq, Nqj, Nq),
          blocks=(nvertelem, nhorzelem),
          knl_banded_matrix_vector_product!(bl, Val(dim), Val(N),
                                            Val(nvertelem), Val(p), Val(q),
                                            dQ.data, A, Q.data, 1:nhorzelem,
                                            1:nvertelem))
end
