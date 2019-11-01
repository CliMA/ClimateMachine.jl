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

function (dg::DGModel)(dYdt, Y, param, t; increment=false)
  bl = dg.balancelaw
  FT = eltype(Y)
  device = typeof(Y.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  E    = grid.topology.realelems
  nE   = length(E)
  Nd   = dimensionality(grid)
  N    = polynomialorder(grid)
  Nq   = N + 1
  Nqk  = Nd == 2 ? 1 : Nq
  Nfp  = Nq * Nqk
  Np   = dofs_per_element(grid)

  σ  = param.diff
  α  = param.aux
  nσ = num_diffusive(bl, FT)

  vgeo = grid.vgeo
  sgeo = grid.sgeo
  ω    = grid.ω
  D    = grid.D
  M⁻   = grid.vmapM
  M⁺   = grid.vmapP
  Mᴮ   = grid.elemtobndy

  communicate = !(isstacked(grid.topology) &&
                  typeof(dg.direction) <: VerticalDirection)

  if hasmethod(update_aux!, Tuple{typeof(dg), typeof(bl), typeof(Y), typeof(α),
                                  typeof(t)})
    update_aux!(dg, bl, Y, α, t)
  end

  ########################
  # Gradient Computation #
  ########################
  if communicate
    MPIStateArrays.start_ghost_exchange!(Y)
    MPIStateArrays.start_ghost_exchange!(α)
  end

  if nσ > 0
    @launch(device, threads=(Nq, Nq, Nqk), blocks=nE,
            volume_diffusive_terms!(bl, Val(Nd), Val(N),
            Y.data, σ.data, α.data, vgeo, t, D, E))

    if communicate
      MPIStateArrays.finish_ghost_recv!(Y)
      MPIStateArrays.finish_ghost_recv!(α)
    end

    @launch(device, threads=Nfp, blocks=nE,
            face_diffusive_terms!(bl, Val(Nd), Val(N), dg.gradnumflux,
            Y.data, σ.data, α.data, vgeo, sgeo, t, M⁻, M⁺, Mᴮ, E))

    communicate && MPIStateArrays.start_ghost_exchange!(σ)
  end

  ###################
  # RHS Computation #
  ###################
  @launch(device, threads=(Nq, Nq, Nqk), blocks=nE,
          volume_tendency!(bl, Val(Nd), Val(N),
          dYdt.data, Y.data, σ.data, α.data, vgeo, t, ω, D, E, increment))

  if communicate
    if nσ > 0
      MPIStateArrays.finish_ghost_recv!(σ)
    else
      MPIStateArrays.finish_ghost_recv!(Y)
      MPIStateArrays.finish_ghost_recv!(α)
    end
  end

  @launch(device, threads=Nfp, blocks=nE,
          face_tendency!(bl, Val(Nd), Val(N), dg.numfluxnondiff, dg.numfluxdiff,
                   dYdt.data, Y.data, σ.data, α.data, vgeo, sgeo, t,
                   M⁻, M⁺, Mᴮ, E))

  # Just to be safe, we wait on the sends we started.
  if communicate
    MPIStateArrays.finish_ghost_send!(σ)
    MPIStateArrays.finish_ghost_send!(Y)
  end
end

"""
Initialize the ODE state array.
"""
function init_ode_state(dg::DGModel args...; device==arraytype(dg.grid) <: Array ? CPU() : CUDA(), commtag=888)
  array_device = arraytype(dg.grid) <: Array ? CPU() : CUDA()
  @assert device == CPU() || device == array_device

  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology

  Y = create_state(bl, grid, commtag)
  α = dg.auxstate

  Nd   = dimensionality(grid)
  N    = polynomialorder(grid)
  Np   = dofs_per_element(grid)
  vgeo = grid.vgeo
  E    = topology.realelems
  nE   = length(E)

  if device == array_device
    @launch(device, threads=(Np,), blocks=nE,
            initstate!(bl, Val(Nd), Val(N), Y.data, α.data, vgeo, E, args...))
  else
    h_vgeo = Array(vgeo)
    h_Y = similar(Y, Array)
    h_α = similar(α, Array)
    h_α .= α
    @launch(device, threads=(Np,), blocks=nE,
      initstate!(bl, Val(Nd), Val(N), h_Y.data, h_α.data, h_vgeo, E, args...))
    Y .= h_Y
  end

  MPIStateArrays.start_ghost_exchange!(Y)
  MPIStateArrays.finish_ghost_exchange!(Y)

  return Y
end

function indefinite_stack_integral!(dg::DGModel, m::BalanceLaw,
                                    Y::MPIStateArray, α::MPIStateArray,
                                    t::Real)
  FT = eltype(Y)
  device = typeof(Y.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  Nd   = dimensionality(grid)
  N    = polynomialorder(grid)
  Nq   = N + 1
  Nqk  = Nd == 2 ? 1 : Nq
  vgeo = grid.vgeo

  # do integrals
  topology = grid.topology
  nintegrals = num_integrals(m, FT)
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_indefinite_stack_integral!(m, Val(Nd), Val(N),
                                         Val(nvertelem), Y.data, α.data,
                                         vgeo, grid.Imat, 1:nhorzelem,
                                         Val(nintegrals)))
end

function reverse_indefinite_stack_integral!(dg::DGModel, m::BalanceLaw,
                                            α::MPIStateArray, t::Real)
  FT = eltype(α)
  device = typeof(α.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  Nd   = dimensionality(grid)
  N    = polynomialorder(grid)
  Nq   = N + 1
  Nqk  = Nd == 2 ? 1 : Nq
  vgeo = grid.vgeo

  # do integrals
  topology = grid.topology
  nintegrals = num_integrals(m, FT)
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_reverse_indefinite_stack_integral!(Val(Nd), Val(N),
                                               Val(nvertelem), α.data,
                                                 1:nhorzelem,
                                                 Val(nintegrals)))
end

function nodal_update_aux!(f!, dg::DGModel, m::BalanceLaw, Y::MPIStateArray,
                           α::MPIStateArray, t::Real)
  device = typeof(Y.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  E    = grid.topology.realelems
  nE   = length(E)
  Nd   = dimensionality(grid)
  N    = polynomialorder(grid)
  Np   = dofs_per_element(grid)

  ### update aux variables
  @launch(device, threads=(Np,), blocks=nE,
          knl_nodal_update_aux!(m, Val(Nd), Val(N), f!, Y.data, α.data, t, E))
end
