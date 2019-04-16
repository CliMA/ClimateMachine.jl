"""
DG Balance Law Discretizations module.
Attempts to provide a reasonable implementation of a discontinuous Galerkin
method (in weak form) on tensor product quadrilateral (2D) and hexahedral (3D)
elements for balance laws of the form

```math
q_{,t} + Σ_{i=1,...d} F_{i,i} = s
```

where ``q`` is the state vector, ``F`` is the flux function, and ``s`` is the
source function. ``F`` includes both the "inviscid" and "viscous" fluxes. Note
that this is a space only discretization, time must be advanced using some
ordinary differential equations methods; see [`ODESolvers`](@ref).

Much of the notation used in this module follows Hesthaven and Warburton (2008).

!!! note

    We plan to switch to a skew-symmetric formulation (at which time this note
    will be removed)

!!! references

    ```
    @BOOK{HesthavenWarburton2008,
      title = {Nodal Discontinuous {G}alerkin Methods: {A}lgorithms, Analysis,
               and Applications},
      publisher = {Springer-Verlag New York},
      year = {2008},
      author = {Hesthaven, Jan S. and Warburton, Tim},
      volume = {54},
      series = {Texts in Applied Mathematics},
      doi = {10.1007/978-0-387-72067-8}
    }
    ```
"""
module DGBalanceLawDiscretizations

using MPI
using ...Grids
using ...MPIStateArrays
using Documenter
using StaticArrays
using ...SpaceMethods
using DocStringExtensions

export DGBalanceLaw

# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
       _x, _y, _z = 1:_nvgeo

const _nsgeo = 5
const _nx, _ny, _nz, _sMJ, _vMJI = 1:_nsgeo
# }}}

include("DGBalanceLawDiscretizations_kernels.jl")
include("NumericalFluxes.jl")

"""
    DGBalanceLaw <: AbstractDGMethod

This contains the necessary information for a discontinuous Galerkin method for
balance laws.

See also: Outer constructor [`DGBalanceLaw`](@ref)

# Fields

$(DocStringExtensions.FIELDS)

"""
struct DGBalanceLaw <: AbstractDGMethod
  "computational grid / mesh"
  grid::DiscontinuousSpectralElementGrid

  "number of state"
  nstate::Int

  "tuple of states to take the gradient of"
  gradstates::Tuple

  "physical inviscid flux function"
  inviscid_flux!::Function

  "numerical flux function"
  inviscid_numericalflux!::Function

  "storage for the grad"
  Qgrad::MPIStateArray

  "auxiliary state array"
  auxstate::MPIStateArray

  "source function"
  source!::Union{Nothing, Function}
end

"""
     DGBalanceLaw(; grid::DiscontinuousSpectralElementGrid, length_state_vector,
                  inviscid_flux!, inviscid_numericalflux!,
                  auxiliary_state_length=0,
                  auxiliary_state_initialization! = nothing,
                  source! = nothing)

Constructs a `DGBalanceLaw` spatial discretization type for the physics defined
by `inviscid_flux!` and `source!`. The computational domain is defined by
`grid`. The number of state variables is defined by `length_state_vector`. The
user may also specify an auxiliary state which will be unpacked by the compute
kernel passed on to the user-defined flux and numerical flux functions. The
source function `source!` is optional.

The inviscid flux function is called with data from a degree of freedom (DOF) as
```
inviscid_flux!(F, Q, aux, t)
```
where
- `F` is an `MArray` of size `(dim, length_state_vector)` to be filled (note
  that this is uninitialized so user must set to zero is this desired)
- `Q` is the state to evaluate (`MArray`)
- `aux` is the user-defined auxiliary state (`MArray`)
- `t` is the current simulation time
Warning: Modifications to `Q` or `aux` may have side effects and should not be done

The inviscid numerical flux function is called with data from two DOFs as
```
inviscid_numericalflux!(F, nM, QM, auxM, QP, auxP, t)
```
where
- `F` is an `MArray` of size `(dim, length_state_vector)` to be filled with the
  numerical flux across the face (note that this is uninitialized so user must
  set to zero is this desired)
- `nM` is the unit outward normal to the face with respect to the minus side
  (`MVector` of length `3`)
- `QM` and `QP` are the minus and plus side values (`MArray`)
- `auxM` and `auxP` are the auxiliary states (`MArray`)
- `t` is the current simulation time
Warning: Modifications to `nM`, `QM`, `auxM`, `QP`, or `auxP` may have side
effects and should not be done

If present the source function is called with data from a DOF as
```
source!(S, Q, aux, t)
```
where `S` is an `MVector` of length `length_state_vector` to be filled; other
arguments are the same as `inviscid_flux!` and the same warning concerning `Q`
and `aux` applies.

When `auxiliary_state_initialization! !== nothing` then this is called on the
auxiliary state (assuming `auxiliary_state_length > 0`) as
```
auxiliary_state_initialization!(aux, x, y, z)
```
where `aux` is an `MArray` to fill with the auxiliary state for a DOF located at
Cartesian coordinate locations `(x, y, z)`; see also
[`grad_auxiliary_state!`](@ref) allows the user to take the gradient of a field
stored in the auxiliary state.

!!! note

    If `(x, y, z)`, or data derived from this such as spherical coordinates, is
    needed in the flux or source the user is responsible to storing this in the
    auxiliary state

!!! todo

    - Add support for boundary conditions
    - support viscous fluxes (`gradstates` is in the argument list as part of
      this future interface)

"""
function DGBalanceLaw(;grid::DiscontinuousSpectralElementGrid,
                      length_state_vector, inviscid_flux!,
                      inviscid_numericalflux!, gradstates=(),
                      auxiliary_state_length=0,
                      auxiliary_state_initialization! = nothing,
                      source! = nothing)

  ngradstate = length(gradstates)
  topology = grid.topology
  Np = dofs_per_element(grid)
  h_vgeo = Array(grid.vgeo)
  DFloat = eltype(h_vgeo)
  DA = arraytype(grid)
  # TODO: Clean up this MPIStateArray interface...
  Qgrad = MPIStateArray{Tuple{Np, ngradstate},
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
    initauxstate!(Val(dim), Val(N), Val(auxiliary_state_length),
                  auxiliary_state_initialization!, auxstate, vgeo,
                  topology.realelems)
    MPIStateArrays.start_ghost_exchange!(auxstate)
    MPIStateArrays.finish_ghost_exchange!(auxstate)
  end

  DGBalanceLaw(grid, length_state_vector, gradstates, inviscid_flux!,
               inviscid_numericalflux!, Qgrad, auxstate, source!)
end

"""
    MPIStateArray(disc::DGBalanceLaw; commtag=888)

Given a discretization `disc` constructs an `MPIStateArrays` for holding a
solution state. The optional `commtag` allows the user to set the tag to use for
communication with this `MPIStateArray`.
"""
function MPIStateArrays.MPIStateArray(disc::DGBalanceLaw; commtag=888)
  grid = disc.grid
  topology = disc.grid.topology
  nvar = disc.nstate
  # FIXME: Remove after updating CUDA
  h_vgeo = Array(disc.grid.vgeo)
  DFloat = eltype(h_vgeo)
  Np = dofs_per_element(grid)
  DA = arraytype(grid)
  MPIStateArray{Tuple{Np, nvar}, DFloat, DA}(topology.mpicomm,
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

    `Q` is `unde` at start the function (i.e., not initialized to zero)

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

#TODO: Need to think about where this should really live. Grid? MPIStateArrays?
include("../Mesh/vtk.jl")

"""
    writevtk(prefix, Q::MPIStateArray, disc::DGBalanceLaw [, fieldnames])

Write a vtk file for all the fields in the state array `Q` using geometry and
connectivity information from `disc.grid`. The filename will start with `prefix`
which may also contain a directory path. The names used for each of the fields
in the vtk file can be specified through the collection of strings `fieldnames`;
if not specified the fields names will be `"Q1"` through `"Qk"` where `k` is the
number of states in `Q`, i.e., `k = size(Q,2)`.

"""
function writevtk(prefix, Q::MPIStateArray, disc::DGBalanceLaw,
                  fieldnames=nothing)
  vgeo = disc.grid.vgeo
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q) = host_array ? (vgeo, Q.Q) : (Array(vgeo), Array(Q))
  writevtk_helper(prefix, h_vgeo, h_Q, disc.grid, fieldnames)
end


"""
    writevtk_helper(prefix, vgeo::Array, Q::Array, grid, fieldnames)

Internal helper function for `writevtk`
"""
function writevtk_helper(prefix, vgeo::Array, Q::Array, grid, fieldnames)

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq  = N+1

  nelem = size(Q)[end]
  Xid = (grid.xid, grid.yid, grid.zid)
  X = ntuple(j->reshape((@view vgeo[:, Xid[j], :]),
                        ntuple(j->Nq, dim)...,
                        nelem), dim)
  if fieldnames == nothing
    fields = ntuple(i->("Q$i", reshape((@view Q[:, i, :]),
                                       ntuple(j->Nq, dim)..., nelem)),
                    size(Q, 2))
  else
    fields = ntuple(i->(fieldnames[i], reshape((@view Q[:, i, :]),
                                               ntuple(j->Nq, dim)..., nelem)),
                    size(Q, 2))
  end
  writemesh(prefix, X...; fields=fields, realelems=grid.topology.realelems)
end

"""
    odefun!(disc::DGBalanceLaw, dQ::MPIStateArray, Q::MPIStateArray, t)

Evaluates the right-hand side of the discontinuous Galerkin semi-discretization
defined by `disc` at time `t` with state `Q`. The result is added into
`dQ`. Namely, the semi-discretization is of the form
```math
Q̇ = F(Q, t)
```
and after the call `dQ += F(Q, t)`
"""
function SpaceMethods.odefun!(disc::DGBalanceLaw, dQ::MPIStateArray,
                              Q::MPIStateArray, t)
  grid = disc.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)

  Qgrad = disc.Qgrad
  auxstate = disc.auxstate

  nstate = disc.nstate
  ngradstate = length(disc.gradstates)
  nauxstate = size(auxstate, 2)

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

  if ngradstate > 0
    error("Grad not implemented yet")

    # TODO: volumegrad!

    MPIStateArrays.finish_ghost_exchange!(Q)

    # TODO: facegrad!

    MPIStateArrays.start_ghost_exchange!(Qgrad)
  end

  ###################
  # RHS Computation #
  ###################

  volumerhs!(Val(dim), Val(N), Val(nstate), Val(ngradstate), Val(nauxstate),
             disc.inviscid_flux!, disc.source!, dQ.Q, Q.Q, Qgrad.Q, auxstate.Q,
             vgeo, t, Dmat, topology.realelems)

  MPIStateArrays.finish_ghost_exchange!(ngradstate > 0 ? Qgrad : Q)

  ngradstate > 0 && MPIStateArrays.finish_ghost_exchange!(Qgrad)
  ngradstate == 0 && MPIStateArrays.finish_ghost_exchange!(Q)

  facerhs!(Val(dim), Val(N), Val(nstate), Val(ngradstate), Val(nauxstate),
           disc.inviscid_numericalflux!, dQ.Q, Q.Q, Qgrad.Q, auxstate.Q, vgeo,
           sgeo, t, vmapM, vmapP, elemtobndy, topology.realelems)
end

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

  elem_grad_field!(Val(dim), Val(N), Val(nauxstate), auxstate, vgeo,
                   Dmat, topology.elems, id, idx, idy, idz)
end

end
