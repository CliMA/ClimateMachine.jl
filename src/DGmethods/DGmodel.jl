struct DGModel{BL,G,NF,GNF}
  balancelaw::BL
  grid::G
  divnumflux::NF
  gradnumflux::GNF
end

function (dg::DGModel)(dQdt, Q, param, t)
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

  Qvisc = param.diff
  auxstate = param.aux

  nviscstate = num_diffusive(bl)
  ngradstate = num_transform(bl)
  nauxstate = size(auxstate, 2)
  states_grad = map(var -> findfirst(isequal(var), vars_state(bl)), vars_state_for_transform(bl))

  Dmat = grid.D
  vgeo = grid.vgeo
  sgeo = grid.sgeo
  vmapM = grid.vmapM
  vmapP = grid.vmapP
  elemtobndy = grid.elemtobndy


  ########################
  # Gradient Computation #
  ########################
  MPIStateArrays.start_ghost_exchange!(Q)

  if nviscstate > 0

    @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
            volumeviscterms!(dg, Val(dim), Val(N), Val(num_state(bl)), Val(states_grad),
                             Val(ngradstate), Val(nviscstate), Val(nauxstate),
                             Q.Q, Qvisc.Q, auxstate.Q, vgeo, t, Dmat,
                             topology.realelems))

    MPIStateArrays.finish_ghost_recv!(Q)

    @launch(device, threads=Nfp, blocks=nrealelem,
            faceviscterms!(dg, Val(dim), Val(N), Val(num_state(bl)), Val(states_grad),
                           Val(ngradstate), Val(nviscstate), Val(nauxstate),
                           Q.Q, Qvisc.Q, auxstate.Q,
                           vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
                           topology.realelems))

    MPIStateArrays.start_ghost_exchange!(Qvisc)
  end

  ###################
  # RHS Computation #
  ###################

  @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
          volumerhs!(Val(dim), Val(N), Val(num_state(bl)), Val(nviscstate),
                     Val(nauxstate), dQdt.Q, Q.Q,
                     Qvisc.Q, auxstate.Q, vgeo, t, Dmat, topology.realelems))

  MPIStateArrays.finish_ghost_recv!(nviscstate > 0 ? Qvisc : Q)

  # The main reason for this protection is not for the MPI.Waitall!, but the
  # make sure that we do not recopy data to the GPU
  nviscstate > 0 && MPIStateArrays.finish_ghost_recv!(Qvisc)
  nviscstate == 0 && MPIStateArrays.finish_ghost_recv!(Q)

  @launch(device, threads=Nfp, blocks=nrealelem,
          facerhs!(Val(dim), Val(N), Val(num_state(bl)), Val(nviscstate),
                   Val(nauxstate),
                   dQdt.Q, Q.Q, Qvisc.Q,
                   auxstate.Q, vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
                   topology.realelems))

  # Just to be safe, we wait on the sends we started.
  MPIStateArrays.finish_ghost_send!(Qvisc)
  MPIStateArrays.finish_ghost_send!(Q)
end






function init_ode_param(dg::DGModel)
  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology
  Np = dofs_per_element(grid)
  
  h_vgeo = Array(grid.vgeo)
  DFloat = eltype(h_vgeo)
  DA = arraytype(grid)

  
  # TODO: Clean up this MPIStateArray interface...
  diffstate = MPIStateArray{Tuple{Np, num_diffusive(bl)},DFloat, DA}(
    topology.mpicomm,
    length(topology.elems),
    realelems=topology.realelems,
    ghostelems=topology.ghostelems,
    sendelems=topology.sendelems,
    nabrtorank=topology.nabrtorank,
    nabrtorecv=topology.nabrtorecv,
    nabrtosend=topology.nabrtosend,
    weights=view(h_vgeo, :, grid.Mid, :),
    commtag=111)

  auxstate = MPIStateArray{Tuple{Np, num_aux(bl)}, DFloat, DA}(
    topology.mpicomm,
    length(topology.elems),
    realelems=topology.realelems,
    ghostelems=topology.ghostelems,
    sendelems=topology.sendelems,
    nabrtorank=topology.nabrtorank,
    nabrtorecv=topology.nabrtorecv,
    nabrtosend=topology.nabrtosend,
    weights=view(h_vgeo, :, grid.Mid, :),
    commtag=222)

  # if auxiliary_state_initialization! !== nothing
  #   @assert auxiliary_state_length > 0
    dim = dimensionality(grid)
    N = polynomialorder(grid)
    vgeo = grid.vgeo
    device = typeof(auxstate.Q) <: Array ? CPU() : CUDA()
    nrealelem = length(topology.realelems)
    @launch(device, threads=(Np,), blocks=nrealelem,
            initauxstate!(bl, Val(dim), Val(N), Val(num_aux(bl)),
                          auxstate.Q, vgeo, topology.realelems))
    MPIStateArrays.start_ghost_exchange!(auxstate)
    MPIStateArrays.finish_ghost_exchange!(auxstate)
  # end
  return (aux=auxstate, diff=diffstate)
end



function init_ode_state(dg::DGModel, param, args...; commtag=888)
  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology
  # FIXME: Remove after updating CUDA
  h_vgeo = Array(grid.vgeo)
  DFloat = eltype(h_vgeo)
  Np = dofs_per_element(grid)
  DA = arraytype(grid)
  Q = MPIStateArray{Tuple{Np, num_state(bl)}, DFloat, DA}(topology.mpicomm,
                                               length(topology.elems),
                                               realelems=topology.realelems,
                                               ghostelems=topology.ghostelems,
                                               sendelems=topology.sendelems,
                                               nabrtorank=topology.nabrtorank,
                                               nabrtorecv=topology.nabrtorecv,
                                               nabrtosend=topology.nabrtosend,
                                               weights=view(h_vgeo, :,
                                                            grid.Mid, :),
                                               commtag=commtag)



  vgeo = grid.vgeo
  Np = dofs_per_element(grid)
  auxstate = param.aux
  nauxstate = size(auxstate, 2)

  # FIXME: GPUify me
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q, h_auxstate) = host_array ? (vgeo, Q, auxstate) :
                                       (Array(vgeo), Array(Q), Array(auxstate))
  Qdof = MArray{Tuple{num_state(bl)}, eltype(h_Q)}(undef)
  auxdof = MArray{Tuple{nauxstate}, eltype(h_Q)}(undef)
  @inbounds for e = 1:size(Q, 3), i = 1:Np
    coords = (h_vgeo[i, grid.xid, e], h_vgeo[i, grid.yid, e],
                 h_vgeo[i, grid.zid, e])

    for s = 1:nauxstate
      auxdof[s] = h_auxstate[i, s, e]
    end
    init_state!(bl, State{vars_state(bl)}(Qdof), State{vars_aux(bl)}(auxdof), coords, args...)
    for n = 1:num_state(bl)
      h_Q[i, n, e] = Qdof[n]
    end
  end
  if !host_array
    Q .= h_Q
  end

  Q
end

