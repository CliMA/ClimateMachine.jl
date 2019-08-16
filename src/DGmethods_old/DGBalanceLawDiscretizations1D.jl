module DGBalanceLawDiscretizations1D

using MPI
using ..Mesh.Grids
using ..MPIStateArrays
using StaticArrays
using ..SpaceMethods
using DocStringExtensions
using ..Mesh.Topologies
using GPUifyLoops

include("DGBalanceLawDiscretizations1D_kernels.jl")
include("NumericalFluxes_old.jl")

export DGBalanceLaw1D

"""
    DGBalanceLaw1D <: AbstractDGMethod

This contains the necessary information for a discontinuous Galerkin method for
balance laws.

See also: Outer constructor [`DGBalanceLaw1D`](@ref)

# Fields

$(DocStringExtensions.FIELDS)

"""
struct DGBalanceLaw1D <: AbstractDGMethod
  "computational grid / mesh"
  grid::DiscontinuousSpectralElementGrid

  "number of state"
  nstate::Int

  "physical flux function"
  flux!::Function

  "numerical flux function"
  numerical_flux!::Function

  "numerical boundary flux function"
  numerical_boundary_flux!::Union{Nothing, Function}

  "auxiliary state array"
  auxstate::MPIStateArray

  "source function"
  source!::Union{Nothing, Function}

  "callback function for before the `odefun!`"
  preodefun!::Union{Nothing, Function}
end

function DGBalanceLaw1D(;grid::DiscontinuousSpectralElementGrid,
                      length_state_vector, flux!,
                      numerical_flux!,
                      numerical_boundary_flux! = nothing,
                      auxstate = nothing,
                      source! = nothing,
                      preodefun! = nothing)

  topology = grid.topology
  @assert isstacked(topology)
  Np = dofs_per_element(grid)
  h_vgeo = Array(grid.vgeo)
  DFloat = eltype(h_vgeo)
  DA = arraytype(grid)

  (Topologies.hasboundary(topology) &&
   numerical_boundary_flux! === nothing &&
   error("no `numerical_boundary_flux!` given when topology "*
         "has boundary"))

  weights = view(h_vgeo, :, grid.Mid, :)
  weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

  DGBalanceLaw1D(grid, length_state_vector, flux!,
               numerical_flux!, numerical_boundary_flux!,
               auxstate, source!, preodefun!)
end

"""
    MPIStateArray(disc::DGBalanceLaw1D; nstate=disc.nstate, commtag=888)

Given a discretization `disc` constructs an `MPIStateArrays` for holding a
solution state. The optional 'nstate' arguments allows the user to specify a
specific number of states. The optional `commtag` allows the user to set the tag
to use for communication with this `MPIStateArray`.
"""
function MPIStateArrays.MPIStateArray(disc::DGBalanceLaw1D; nstate=disc.nstate,
                                      commtag=888)
  grid = disc.grid
  topology = disc.grid.topology
  # FIXME: Remove after updating CUDA
  h_vgeo = Array(disc.grid.vgeo)
  DFloat = eltype(h_vgeo)
  Np = dofs_per_element(grid)
  DA = arraytype(grid)

  weights = view(h_vgeo, :, grid.Mid, :)
  weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

  MPIStateArray{Tuple{Np, nstate}, DFloat, DA}(topology.mpicomm,
                                               length(topology.elems),
                                               realelems=topology.realelems,
                                               ghostelems=topology.ghostelems,
                                               sendelems=topology.sendelems,
                                               nabrtorank=topology.nabrtorank,
                                               nabrtorecv=topology.nabrtorecv,
                                               nabrtosend=topology.nabrtosend,
                                               weights=weights,
                                               commtag=commtag)
end

"""
    MPIStateArray(disc::DGBalanceLaw1D, initialization!::Function; commtag=888)

Given a discretization `disc` constructs an `MPIStateArrays` for holding a
solution state. The optional `commtag` allows the user to set the tag to use
for communication with this `MPIStateArray`.

After allocation the `MPIStateArray` is initialized using the function
`initialization!` which will be called as:
```
initialization!(Q, x1, x2, x3, aux)
```
where `Q` is an `MArray` with the solution state at a single degree of freedom
(DOF) to initialize and `(x1, x2, x3)` is the coordinate point for the
allocation.  The auxiliary data the values at the DOF are passed through as an
`MArray` through the `aux` argument; if `disc` does not have auxiliary data then
the length of the `MArray` will be zero.

!!! note

    `Q` is `undef` at start the function (i.e., not initialized to zero)

!!! note

    Modifications of the `aux` array will be discarded.

!!! todo

    Remove `host` and `device` data transfers.

"""
function MPIStateArrays.MPIStateArray(disc::DGBalanceLaw1D,
                                      ic!::Function; commtag=888)
  Q = MPIStateArray(disc; commtag=commtag)

  nvar = disc.nstate
  grid = disc.grid
  topology = grid.topology
  auxstate = disc.auxstate
  nauxstate = size(auxstate, 2)
  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Np = dofs_per_element(grid)
  vgeo = grid.vgeo
  nrealelem = length(topology.realelems)

  # FIXME: initialize directly on the device
  device = CPU()
  h_vgeo = Array(vgeo)
  h_Q = similar(Q, Array)
  h_auxstate = similar(auxstate, Array)
  
  h_auxstate .= auxstate

  @launch(device, threads=(Np,), blocks=nrealelem,
          initstate!(Val(dim), Val(N), Val(nvar), Val(nauxstate),
                     ic!, h_Q.Q, h_auxstate.Q, h_vgeo, topology.realelems))

  Q .= h_Q

  MPIStateArrays.start_ghost_exchange!(Q)
  MPIStateArrays.finish_ghost_exchange!(Q)

  Q
end

"""
    MPIStateArray(initialization!::Function, disc::DGBalanceLaw1D; commtag=888)

Wrapper function to allow for calls of the form

```
MPIStateArray(disc) do  Q, x1, x2, x3
  # fill Q
end
```

See also [`MPIStateArray`](@ref)
"""
MPIStateArrays.MPIStateArray(f::Function,
                             d::DGBalanceLaw1D; commtag=888
                            ) = MPIStateArray(d, f; commtag=commtag)


"""
    odefun!(disc::DGBalanceLaw1D, dQ::MPIStateArray, Q::MPIStateArray, t; increment)

Evaluates the right-hand side of the discontinuous Galerkin semi-discretization
defined by `disc` at time `t` with state `Q`.
The result is either added into
`dQ` if `increment` is true or stored in `dQ` if it is false.
Namely, the semi-discretization is of the form
``
  \\dot{Q} = F(Q, t)
``
and after the call `dQ += F(Q, t)` if `increment == true`
or `dQ = F(Q, t)` if `increment == false`
"""
function SpaceMethods.odefun!(disc::DGBalanceLaw1D, dQ::MPIStateArray,
                              Q::MPIStateArray, param, t; increment)

  device = typeof(Q.Q) <: Array ? CPU() : CUDA()

  grid = disc.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Nfp = Nq * Nqk
  nrealelem = length(topology.realelems)

  auxstate = disc.auxstate

  nstate = disc.nstate
  nauxstate = size(auxstate, 2)

  lgl_weights_vec = grid.ω
  Dmat = grid.D
  vgeo = grid.vgeo
  sgeo = grid.sgeo
  vmapM = grid.vmapM
  vmapP = grid.vmapP
  elemtobndy = grid.elemtobndy

  ################################
  # Allow the user to update aux #
  ################################
  disc.preodefun! !== nothing && disc.preodefun!(disc, Q, t)

  ###################
  # RHS Computation #
  ###################

  @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
          volumerhs!(Val(dim), Val(N), Val(nstate), Val(nauxstate),
                     disc.flux!, disc.source!, dQ.Q, Q.Q, auxstate.Q, vgeo, t,
                     lgl_weights_vec, Dmat, topology.realelems, increment))

  @launch(device, threads=Nfp, blocks=nrealelem,
          facerhs!(Val(dim), Val(N), Val(nstate), Val(nauxstate),
                   disc.numerical_flux!, disc.numerical_boundary_flux!, dQ.Q,
                   Q.Q, auxstate.Q, vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
                   topology.realelems))
end

end # module
