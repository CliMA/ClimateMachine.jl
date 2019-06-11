struct DGModel{BL,P,G,NF,GNF}
  balancelaw::BL
  poly::P
  grid::G
  divnumflux::NF
  gradnumflus::GNF
end


function (dg::DGModel)(dQdt, Q, param, t)
  device = typeof(Q.Q) <: Array ? CPU() : CUDA()

  grid = disc.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Nfp = Nq * Nqk
  nrealelem = length(topology.realelems)

  Qvisc = param.visc
  auxstate = param.aux

  nstate = disc.nstate
  nviscstate = disc.number_viscous_states
  ngradstate = disc.number_gradient_states
  nauxstate = size(auxstate, 2)
  states_grad = disc.states_for_gradient_transform

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
            volumeviscterms!(dg, Val(dim), Val(N), Val(nstate), Val(states_grad),
                             Val(ngradstate), Val(nviscstate), Val(nauxstate),
                             Q.Q, Qvisc.Q, auxstate.Q, vgeo, t, Dmat,
                             topology.realelems))

    MPIStateArrays.finish_ghost_recv!(Q)

    @launch(device, threads=Nfp, blocks=nrealelem,
            faceviscterms!(dg, Val(dim), Val(N), Val(nstate), Val(states_grad),
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
          volumerhs!(Val(dim), Val(N), Val(nstate), Val(nviscstate),
                     Val(nauxstate), disc.flux!, disc.source!, dQdt.Q, Q.Q,
                     Qvisc.Q, auxstate.Q, vgeo, t, Dmat, topology.realelems))

  MPIStateArrays.finish_ghost_recv!(nviscstate > 0 ? Qvisc : Q)

  # The main reason for this protection is not for the MPI.Waitall!, but the
  # make sure that we do not recopy data to the GPU
  nviscstate > 0 && MPIStateArrays.finish_ghost_recv!(Qvisc)
  nviscstate == 0 && MPIStateArrays.finish_ghost_recv!(Q)

  @launch(device, threads=Nfp, blocks=nrealelem,
          facerhs!(Val(dim), Val(N), Val(nstate), Val(nviscstate),
                   Val(nauxstate),
                   dQdt.Q, Q.Q, Qvisc.Q,
                   auxstate.Q, vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
                   topology.realelems))

  # Just to be safe, we wait on the sends we started.
  MPIStateArrays.finish_ghost_send!(Qvisc)
  MPIStateArrays.finish_ghost_send!(Q)
end
