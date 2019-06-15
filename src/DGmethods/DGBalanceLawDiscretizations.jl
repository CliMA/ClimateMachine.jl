function DGBalanceLaw(;grid::DiscontinuousSpectralElementGrid,
                      length_state_vector, flux!,
                      numerical_flux!,
                      numerical_boundary_flux! = nothing,
                      states_for_gradient_transform=(),
                      number_gradient_states=0,
                      number_viscous_states=0,
                      gradient_transform! = nothing,
                      viscous_transform! = nothing,
                      viscous_penalty! = nothing,
                      viscous_boundary_penalty! = nothing,
                      auxiliary_state_length=0,
                      auxiliary_state_initialization! = nothing,
                      source! = nothing)

  topology = grid.topology
  Np = dofs_per_element(grid)
  h_vgeo = Array(grid.vgeo)
  DFloat = eltype(h_vgeo)
  DA = arraytype(grid)

  (Topologies.hasboundary(topology) &&
   numerical_boundary_flux! === nothing &&
   error("no `numerical_boundary_flux!` given when topology "*
         "has boundary"))

  if number_viscous_states > 0 || number_gradient_states > 0 ||
    length(states_for_gradient_transform) > 0

    # These should all be true in this case
    @assert number_viscous_states > 0
    @assert number_gradient_states > 0
    @assert length(states_for_gradient_transform) > 0
    @assert gradient_transform! !== nothing
    @assert viscous_transform! !== nothing
    @assert viscous_penalty! !== nothing
    (Topologies.hasboundary(topology)) && (@assert viscous_boundary_penalty! !==
                                           nothing)
  end

  # TODO: Clean up this MPIStateArray interface...
  Qvisc = MPIStateArray{Tuple{Np, number_viscous_states},
                     DFloat, DA
                    }(topology.mpicomm,
                      length(topology.elems),
                      realelems=topology.realelems,
                      ghostelems=topology.ghostelems,
                      sendelems=topology.sendelems,
                      nabrtorank=topology.nabrtorank,
                      nabrtorecv=topology.nabrtorecv,
                      nabrtosend=topology.nabrtosend,
                      weights=view(h_vgeo, :, grid.Mid, :),
                      commtag=111)

  auxstate = MPIStateArray{Tuple{Np, auxiliary_state_length}, DFloat, DA
                          }(topology.mpicomm,
                            length(topology.elems),
                            realelems=topology.realelems,
                            ghostelems=topology.ghostelems,
                            sendelems=topology.sendelems,
                            nabrtorank=topology.nabrtorank,
                            nabrtorecv=topology.nabrtorecv,
                            nabrtosend=topology.nabrtosend,
                            weights=view(h_vgeo, :, grid.Mid, :),
                            commtag=222)

  if auxiliary_state_initialization! !== nothing
    @assert auxiliary_state_length > 0
    dim = dimensionality(grid)
    N = polynomialorder(grid)
    vgeo = grid.vgeo
    device = typeof(auxstate.Q) <: Array ? CPU() : CUDA()
    nrealelem = length(topology.realelems)
    @launch(device, threads=(Np,), blocks=nrealelem,
            initauxstate!(bl, Val(dim), Val(N), Val(auxiliary_state_length),
                          auxiliary_state_initialization!, auxstate.Q, vgeo,
                          topology.realelems))
    MPIStateArrays.start_ghost_exchange!(auxstate)
    MPIStateArrays.finish_ghost_exchange!(auxstate)
  end

  DGBalanceLaw(grid, length_state_vector, flux!,
               numerical_flux!, numerical_boundary_flux!,
               Qvisc, number_gradient_states, number_viscous_states,
               states_for_gradient_transform, gradient_transform!,
               viscous_transform!, viscous_penalty!,
               viscous_boundary_penalty!, auxstate, source!)
end





"""
    MPIStateArray(disc::DGBalanceLaw; nstate=disc.nstate, commtag=888)

Given a discretization `disc` constructs an `MPIStateArrays` for holding a
solution state. The optional 'nstate' arguments allows the user to specify a
specific number of states. The optional `commtag` allows the user to set the tag
to use for communication with this `MPIStateArray`.
"""
function MPIStateArrays.MPIStateArray(disc::DGBalanceLaw; nstate=disc.nstate,
                                      commtag=888)
  grid = disc.grid
  topology = disc.grid.topology
  # FIXME: Remove after updating CUDA
  h_vgeo = Array(disc.grid.vgeo)
  DFloat = eltype(h_vgeo)
  Np = dofs_per_element(grid)
  DA = arraytype(grid)
  MPIStateArray{Tuple{Np, nstate}, DFloat, DA}(topology.mpicomm,
                                               length(topology.elems),
                                               realelems=topology.realelems,
                                               ghostelems=topology.ghostelems,
                                               sendelems=topology.sendelems,
                                               nabrtorank=topology.nabrtorank,
                                               nabrtorecv=topology.nabrtorecv,
                                               nabrtosend=topology.nabrtosend,
                                               weights=view(h_vgeo, :,
                                                            disc.grid.Mid, :),
                                               commtag=commtag)
end

"""
    MPIStateArray(disc::DGBalanceLaw, initialization!::Function; commtag=888)

Given a discretization `disc` constructs an `MPIStateArrays` for holding a
solution state. The optional `commtag` allows the user to set the tag to use
for communication with this `MPIStateArray`.

After allocation the `MPIStateArray` is initialized using the function
`initialization!` which will be called as:
```
initialization!(Q, x, y, z, [aux])
```
where `Q` is an `MArray` with the solution state at a single degree of freedom
(DOF) to initialize and `(x,y,z)` is the coordinate point for the allocation. If
`disc` contains an auxiliary data the values of this at the DOF are passed
through as an `MArray` through the `aux` argument

!!! note

    `Q` is `undef` at start the function (i.e., not initialized to zero)

!!! note

    Modifications of the `aux` array will be discarded.

!!! todo

    GPUify this function to remove `host` and `device` data transfers

"""
function MPIStateArrays.MPIStateArray(disc::DGBalanceLaw,
                                      ic!::Function; commtag=888)
  Q = MPIStateArray(disc; commtag=commtag)

  nvar = disc.nstate
  grid = disc.grid
  vgeo = grid.vgeo
  Np = dofs_per_element(grid)
  auxstate = disc.auxstate
  nauxstate = size(auxstate, 2)

  # FIXME: GPUify me
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q, h_auxstate) = host_array ? (vgeo, Q, auxstate) :
                                       (Array(vgeo), Array(Q), Array(auxstate))
  Qdof = MArray{Tuple{nvar}, eltype(h_Q)}(undef)
  auxdof = MArray{Tuple{nauxstate}, eltype(h_Q)}(undef)
  @inbounds for e = 1:size(Q, 3), i = 1:Np
    (x, y, z) = (h_vgeo[i, grid.xid, e], h_vgeo[i, grid.yid, e],
                 h_vgeo[i, grid.zid, e])
    if nauxstate > 0
      for s = 1:nauxstate
        auxdof[s] = h_auxstate[i, s, e]
      end
      ic!(Qdof, x, y, z, auxdof)
    else
      ic!(Qdof, x, y, z)
    end
    for n = 1:nvar
      h_Q[i, n, e] = Qdof[n]
    end
  end
  if !host_array
    Q .= h_Q
  end

  Q
end

"""
    MPIStateArray(initialization!::Function, disc::DGBalanceLaw; commtag=888)

Wrapper function to allow for calls of the form

```
MPIStateArray(disc) do  Q, x, y, z
  # fill Q
end
```

See also [`MPIStateArray`](@ref)
"""
MPIStateArrays.MPIStateArray(f::Function,
                             d::DGBalanceLaw; commtag=888
                            ) = MPIStateArray(d, f; commtag=commtag)

"""
    grad_auxiliary_state!(disc, i, (ix, iy, iz))

Computes the gradient of a the field `i` of the constant auxiliary state of
`disc` and stores the `x, y, z` compoment in fields `ix, iy, iz` of constant
auxiliary state.

!!! note

    This only computes the element gradient not a DG gradient. If your constant
    auxiliary state is discontinuous this may or may not be what you want!
"""
function grad_auxiliary_state!(disc::DGBalanceLaw, id, (idx, idy, idz))
  grid = disc.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)

  auxstate = disc.auxstate

  nauxstate = size(auxstate, 2)

  @assert nauxstate >= max(id, idx, idy, idz)
  @assert 0 < min(id, idx, idy, idz)
  @assert allunique((idx, idy, idz))

  Dmat = grid.D
  vgeo = grid.vgeo

  device = typeof(auxstate.Q) <: Array ? CPU() : CUDA()

  nelem = length(topology.elems)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  @launch(device, threads=(Nq, Nq, Nqk), blocks=nelem,
          elem_grad_field!(Val(dim), Val(N), Val(nauxstate), auxstate.Q, vgeo,
                           Dmat, topology.elems, id, idx, idy, idz))
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
function dof_iteration!(dof_fun!::Function, R::MPIStateArray, disc::DGBalanceLaw,
                        Q::MPIStateArray)
  grid = disc.grid
  topology = grid.topology

  @assert size(R)[end] == size(Q)[end] == size(disc.auxstate)[end]
  @assert size(R)[1] == size(Q)[1] == size(disc.auxstate)[1]

  dim = dimensionality(grid)
  N = polynomialorder(grid)

  Qvisc = disc.Qvisc
  auxstate = disc.auxstate

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
          knl_dof_iteration!(Val(dim), Val(N), Val(nRstate), Val(nstate),
                             Val(nviscstate), Val(nauxstate), dof_fun!, R.Q,
                             Q.Q, Qvisc.Q, auxstate.Q, topology.realelems))
end

