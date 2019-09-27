struct DGModel{BL,G,NFND,NFD,GNF}
  balancelaw::BL
  grid::G
  numfluxnondiff::NFND
  numfluxdiff::NFD
  gradnumflux::GNF
end
function DGModel(dg::DGModel, bl::BalanceLaw)
  return DGModel(bl, dg.grid, dg.numfluxnondiff, dg.numfluxdiff,
                 dg.gradnumflux,)
end

function (dg::DGModel)(dQdt, Q, param, t; increment=false)
  bl = dg.balancelaw
  DFloat = eltype(Q)
  device = typeof(Q.Q) <: Array ? CPU() : CUDA()

  grid = dg.grid
  Ω    = grid.topology.realelems
  nΩ   = length(Ω)
  Nᵈ   = dimensionality(grid)
  N    = polynomialorder(grid)
  Nq   = N + 1
  Nqk  = Nᵈ == 2 ? 1 : Nq
  Nfp  = Nq * Nqk
  Np   = dofs_per_element(grid)

  σ  = param.diff
  α  = param.aux
  nσ = num_diffusive(bl, DFloat)

  vgeo = grid.vgeo
  sgeo = grid.sgeo
  ω    = grid.ω
  D    = grid.D
  ι⁻   = grid.vmapM
  ι⁺   = grid.vmapP
  ιᴮ   = grid.elemtobndy

  if hasmethod(update_aux!, Tuple{typeof(dg), typeof(bl), typeof(Q), typeof(α),
                                  typeof(t)})
    update_aux!(dg, bl, Q, α, t)
  elseif hasmethod(update_aux!, Tuple{typeof(dg), typeof(bl), typeof(Q),
                                  typeof(α), typeof(t),
                                  typeof(param.blparam)})
    update_aux!(dg, bl, Q, α, t, param.blparam)
  end

  ########################
  # Gradient Computation #
  ########################
  MPIStateArrays.start_ghost_exchange!(Q)
  MPIStateArrays.start_ghost_exchange!(α)

  if nσ > 0
    @launch(device, threads=(Nq, Nq, Nqk), blocks=nΩ,
            volumeviscterms!(bl, Val(Nᵈ), Val(N), Q.Q, σ.Q, α.Q, vgeo, t, D, Ω))

    MPIStateArrays.finish_ghost_recv!(Q)
    MPIStateArrays.finish_ghost_recv!(α)

    @launch(device, threads=Nfp, blocks=nΩ,
            faceviscterms!(bl, Val(Nᵈ), Val(N), dg.gradnumflux, Q.Q, σ.Q, α.Q,
                           vgeo, sgeo, t, ι⁻, ι⁺, ιᴮ, Ω))

    MPIStateArrays.start_ghost_exchange!(σ)
  end

  ###################
  # RHS Computation #
  ###################
  @launch(device, threads=(Nq, Nq, Nqk), blocks=nΩ,
          volumerhs!(bl, Val(Nᵈ), Val(N), dQdt.Q, Q.Q, σ.Q, α.Q, vgeo, t, ω, D,
                     Ω, increment))

  if nσ > 0
    MPIStateArrays.finish_ghost_recv!(σ)
  else
    MPIStateArrays.finish_ghost_recv!(Q)
    MPIStateArrays.finish_ghost_recv!(α)
  end

  # The main reason for this protection is not for the MPI.Waitall!, but the
  # make sure that we do not recopy data to the GPU
  nσ > 0  && MPIStateArrays.finish_ghost_recv!(σ)
  nσ == 0 && MPIStateArrays.finish_ghost_recv!(Q)

  @launch(device, threads=Nfp, blocks=nΩ,
          facerhs!(bl, Val(Nᵈ), Val(N), dg.numfluxnondiff, dg.numfluxdiff,
                   dQdt.Q, Q.Q, σ.Q, α.Q, vgeo, sgeo, t, ι⁻, ι⁺, ιᴮ, Ω))

  # Just to be safe, we wait on the sends we started.
  MPIStateArrays.finish_ghost_send!(σ)
  MPIStateArrays.finish_ghost_send!(Q)
end

"""
    init_ode_param(dg::DGModel)

Initialize the ODE parameter object, containing the auxiliary and diffusive states. The extra `args...` are passed through to `init_state!`.
"""
function init_ode_param(dg::DGModel)
  bl = dg.balancelaw

  grid = dg.grid
  Nᵈ   = dimensionality(grid)
  N    = polynomialorder(grid)
  Np   = dofs_per_element(grid)
  vgeo = grid.vgeo
  topology = grid.topology
  Ω    = topology.realelems
  nΩ   = length(Ω)

  h_vgeo = Array(vgeo)
  DFloat = eltype(h_vgeo)
  DA     = arraytype(grid)

  weights = view(h_vgeo, :, grid.Mid, :)
  weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

  # TODO: Clean up this MPIStateArray interface...
  σ = MPIStateArray{Tuple{Np, num_diffusive(bl,DFloat)},DFloat, DA}(
    topology.mpicomm,
    length(topology.elems),
    realelems=topology.realelems,
    ghostelems=topology.ghostelems,
    vmaprecv=grid.vmaprecv,
    vmapsend=grid.vmapsend,
    nabrtorank=topology.nabrtorank,
    nabrtovmaprecv=grid.nabrtovmaprecv,
    nabrtovmapsend=grid.nabrtovmapsend,
    weights=weights,
    commtag=111)

  α = MPIStateArray{Tuple{Np, num_aux(bl,DFloat)}, DFloat, DA}(
    topology.mpicomm,
    length(topology.elems),
    realelems=topology.realelems,
    ghostelems=topology.ghostelems,
    vmaprecv=grid.vmaprecv,
    vmapsend=grid.vmapsend,
    nabrtorank=topology.nabrtorank,
    nabrtovmaprecv=grid.nabrtovmaprecv,
    nabrtovmapsend=grid.nabrtovmapsend,
    weights=weights,
    commtag=222)

  device = typeof(α.Q) <: Array ? CPU() : CUDA()

  @launch(device, threads=(Np,), blocks=nΩ,
          initauxstate!(bl, Val(Nᵈ), Val(N), α.Q, vgeo, Ω))

  MPIStateArrays.start_ghost_exchange!(α)
  MPIStateArrays.finish_ghost_exchange!(α)

  return (aux=α, diff=σ, blparam=init_ode_param(dg, bl))
end
init_ode_param(::DGModel, ::BalanceLaw) = nothing



"""
    init_ode_state(dg::DGModel, param, args...)

Initialize the ODE state array.
"""
function init_ode_state(dg::DGModel, commtag)
  bl = dg.balancelaw

  grid = dg.grid
  Nᵈ   = dimensionality(grid)
  N    = polynomialorder(grid)
  Np   = dofs_per_element(grid)
  vgeo = grid.vgeo
  topology = grid.topology
  Ω    = topology.realelems
  nΩ   = length(Ω)

  # FIXME: Remove after updating CUDA
  h_vgeo = Array(vgeo)
  DFloat = eltype(h_vgeo)
  DA     = arraytype(grid)

  weights = view(h_vgeo, :, grid.Mid, :)
  weights = reshape(weights, size(weights, 1), 1, size(weights, 2))


  Q = MPIStateArray{Tuple{Np, num_state(bl,DFloat)}, DFloat, DA}(topology.mpicomm,
      length(topology.elems),
      realelems=topology.realelems,
      ghostelems=topology.ghostelems,
      vmaprecv=grid.vmaprecv,
      vmapsend=grid.vmapsend,
      nabrtorank=topology.nabrtorank,
      nabrtovmaprecv=grid.nabrtovmaprecv,
      nabrtovmapsend=grid.nabrtovmapsend,
      weights=weights,
      commtag=commtag)

  return Q
end

function init_ode_state(dg::DGModel, param, args...; commtag=888)
  Q = init_ode_state(dg, commtag)

  bl = dg.balancelaw
  α  = param.aux

  grid = dg.grid
  Nᵈ   = dimensionality(grid)
  N    = polynomialorder(grid)
  Np   = dofs_per_element(grid)
  vgeo = grid.vgeo
  topology = grid.topology
  Ω    = topology.realelems
  nΩ   = length(Ω)

  # FIXME: Remove after updating CUDA
  h_vgeo = Array(vgeo)
  DFloat = eltype(h_vgeo)
  DA     = arraytype(grid)

  device = typeof(Q.Q) <: Array ? CPU() : CUDA()

  @launch(device, threads=(Np,), blocks=nΩ,
          initstate!(bl, Val(Nᵈ), Val(N), Q.Q, α.Q, vgeo, Ω, args...))

  MPIStateArrays.start_ghost_exchange!(Q)
  MPIStateArrays.finish_ghost_exchange!(Q)

  return Q
end


"""
    dof_iteration!(dof_fun!::Function, R::MPIStateArray, disc::DGBalanceLaw,
                   Q::MPIStateArray)

Iterate over each dof to fill `R` using the `dof_fun!`. The syntax of the
`dof_fun!` is
```
dof_fun!(l_R, l_Q, l_σ, l_aux)
```
where `l_R`, `l_Q`, `l_σ`, and `l_α` are of type `MArray` filled initially
with the values at a single degree of freedom. After the call the values in
`l_R` will be written back to the degree of freedom of `R`.
"""
function node_apply_aux!(f!::Function, dg::DGModel, Q::MPIStateArray, param::MPIStateArray)
  bl = dg.balancelaw
  device = typeof(α.Q) <: Array ? CPU() : CUDA()

  σ = param.diff
  α = param.aux

  grid = dg.grid
  Ω    = grid.topology.realelems
  nΩ   = length(Ω)
  Nᵈ   = dimensionality(grid)
  N    = polynomialorder(grid)
  Np   = dofs_per_element(grid)

  @assert size(Q)[end] == size(dg.α)[end]
  @assert size(Q)[1]   == size(dg.α)[1]

  @launch(device, threads=(Np,), blocks=nΩ,
    knl_node_apply_aux!(bl, Val(Nᵈ), Val(N), f!, Q.Q, σ.Q, α.Q, Ω))
end

function indefinite_stack_integral!(dg::DGModel, m::BalanceLaw,
                                    Q::MPIStateArray, α::MPIStateArray,
                                    t::Real)
  DFloat = eltype(Q)
  device = typeof(Q.Q) <: Array ? CPU() : CUDA()

  grid = dg.grid
  Nᵈ   = dimensionality(grid)
  N    = polynomialorder(grid)
  Nq   = N + 1
  Nqk  = Nᵈ == 2 ? 1 : Nq
  vgeo = grid.vgeo

  # do integrals
  topology = grid.topology
  nintegrals = num_integrals(m, DFloat)
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_indefinite_stack_integral!(m, Val(Nᵈ), Val(N),
                                         Val(nvertelem), Q.Q, α.Q,
                                         vgeo, grid.Imat, 1:nhorzelem,
                                         Val(nintegrals)))
end

function reverse_indefinite_stack_integral!(dg::DGModel, m::BalanceLaw,
                                            α::MPIStateArray, t::Real)
  DFloat = eltype(α)
  device = typeof(α.Q) <: Array ? CPU() : CUDA()

  grid = dg.grid
  Nᵈ   = dimensionality(grid)
  N    = polynomialorder(grid)
  Nq   = N + 1
  Nqk  = Nᵈ == 2 ? 1 : Nq
  vgeo = grid.vgeo

  # do integrals
  topology = grid.topology
  nintegrals = num_integrals(m, DFloat)
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_reverse_indefinite_stack_integral!(Val(Nᵈ), Val(N),
                                                 Val(nvertelem), α.Q,
                                                 1:nhorzelem,
                                                 Val(nintegrals)))
end

function nodal_update_aux!(f!, dg::DGModel, m::BalanceLaw, Q::MPIStateArray,
                           α::MPIStateArray, t::Real)
  device = typeof(Q.Q) <: Array ? CPU() : CUDA()

  grid = dg.grid
  Ω    = grid.topology.realelems
  nΩ   = length(Ω)
  Nᵈ   = dimensionality(grid)
  N    = polynomialorder(grid)
  Np   = dofs_per_element(grid)

  ### update aux variables
  @launch(device, threads=(Np,), blocks=nΩ,
          knl_nodal_update_aux!(m, Val(Nᵈ), Val(N), f!, Q.Q, α.Q, t, Ω))
end

function copy_stack_field_down!(dg::DGModel, m::BalanceLaw,
                                auxstate::MPIStateArray, fldin, fldout)

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
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_copy_stack_field_down!(Val(dim), Val(polyorder), Val(nvertelem),
                                     auxstate.Q, 1:nhorzelem, Val(fldin),
                                     Val(fldout)))
end
