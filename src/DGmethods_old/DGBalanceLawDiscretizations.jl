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

The flux function `F_{i}` is taken to be of the form:
```math
F_{i} := F_{i}(q, σ; a)\\\\
σ = H(q, ∇G(q; a); a)
```
where ``a`` is a set of parameters and viscous terms enter through ``σ``

The source term is of the form:
```math
s := s(q; a)
```

In the code and docs the following terminology is used:
- ``q`` is referred to as the state
- ``σ`` is the viscous state
- ``a`` is the auxiliary state
- ``F`` is the physical flux
- ``H`` is the viscous transform
- ``G`` is the gradient transform

Much of the notation used in this module follows Hesthaven and Warburton (2008).

!!! note

    Currently all the functions take the same parameters and the gradient
    transform can take a user-specified subset of the state vector.

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
using ..Mesh.Grids
using ..Mesh.Filters
using ..MPIStateArrays
using StaticArrays
using ..SpaceMethods
using DocStringExtensions
using ..Mesh.Topologies
using GPUifyLoops

include("DGBalanceLawDiscretizations_kernels.jl")
include("NumericalFluxes_old.jl")

export DGBalanceLaw

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

  "physical flux function"
  flux!::Function

  "numerical flux function"
  numerical_flux!::Function

  "numerical boundary flux function"
  numerical_boundary_flux!::Union{Nothing, Function}

  "storage for the viscous state"
  Qvisc::MPIStateArray

  "number of out states for gradient_transform!"
  number_gradient_states::Int

  "number of out states for the viscous_transform!"
  number_viscous_states::Int

  "transform from state to variables to take gradient of"
  gradient_transform!::Union{Nothing, Function}

  "transform from Q and gradient state to viscous states"
  viscous_transform!::Union{Nothing, Function}

  "penalty for the viscous state computation"
  viscous_penalty!::Union{Nothing, Function}

  "boundary penalty for the viscous state computation (e.g., Dirichlet)"
  viscous_boundary_penalty!::Union{Nothing, Function}

  "auxiliary state array"
  auxstate::MPIStateArray

  "source function"
  source!::Union{Nothing, Function}

  "callback function for before the `odefun!`"
  preodefun!::Union{Nothing, Function}
end

"""
    DGBalanceLaw(;grid::DiscontinuousSpectralElementGrid,
                 length_state_vector,
                 flux!,
                 numerical_flux!,
                 numerical_boundary_flux! = nothing,
                 number_gradient_states = 0,
                 number_viscous_states = 0,
                 gradient_transform! = nothing,
                 viscous_transform! = nothing,
                 viscous_penalty! = nothing,
                 viscous_boundary_penalty! = nothing,
                 auxiliary_state_length = 0,
                 auxiliary_state_initialization! = nothing,
                 source! = nothing,
                 preodefun! = nothing)

Constructs a `DGBalanceLaw` spatial discretization type for the physics defined
by `flux!` and `source!`. The computational domain is defined by `grid`. The
number of state variables is defined by `length_state_vector`. The user may also
specify an auxiliary state which will be unpacked by the compute kernel passed
on to the user-defined flux and numerical flux functions. The source function
`source!` is optional.

The flux function is called with data from a degree of freedom (DOF) as
```
flux!(F, Q, V, aux, t)
```
where
- `F` is an `MArray` of size `(dim, length_state_vector)` to be filled (note
  that this is uninitialized so the user must set to zero if is this desired)
- `Q` is the state to evaluate (`MArray`)
- `V` is the viscous state to evaluate (`MArray`)
- `aux` is the user-defined auxiliary state (`MArray`)
- `t` is the current simulation time
Warning: Modifications to `Q` or `aux` may cause side effects and should be
avoided.

The numerical flux function is called with data from two DOFs as
```
numerical_flux!(F, nM, QM, VM, auxM, QP, VP, auxP, t)
```
where
- `F` is an `MVector` of length `length_state_vector` to be filled with the
  numerical flux across the face (note that this is uninitialized so user must
  set to zero if is this desired)
- `nM` is the unit outward normal to the face with respect to the minus side
  (`MVector` of length `3`)
- `QM` and `QP` are the minus and plus side states (`MArray`)
- `VM` and `VP` are the minus and plus viscous side states (`MArray`)
- `auxM` and `auxP` are the auxiliary states (`MArray`)
- `t` is the current simulation time
Warning: Modifications to `nM`, `QM`, `auxM`, `QP`, or `auxP` may cause side
effects and should be avoided.

If `grid.topology` has a boundary then the function
`numerical_boundary_flux!` must be specified. This function is called
with the data from the neighbouring DOF as
```
numerical_boundary_flux!(F, nM, QM, VM, auxM, QP, VP, auxP, bctype, t)
```
where
- `F` is an `MArray` of size `(dim, length_state_vector)` to be filled with the
  numerical flux across the face (note that this is uninitialized so user must
  set to zero is this desired)
- `nM` is the unit outward normal to the face with respect to the minus side
  (`MVector` of length `3`)
- `QM` and `QP` are the minus and plus side states (`MArray`)
- `VM` and `VP` are the minus and plus viscous side states (`MArray`)
- `auxM` and `auxP` are the auxiliary states (`MArray`)
- `bctype` is the boundary condition flag for the connected face and element of
   `grid.elemtobndy`
- `t` is the current simulation time
Note: `QP` and `auxP` are filled with values based on degrees of freedom
referenced in `grid.vmap⁺`; `QP` and `auxP` may be modified by the calling
function.

Warning: Modifications to `nM`, `QM`, or `auxM` may cause side effects and
should be avoided.

If present the source function is called with data from a DOF as
```
source!(S, Q, aux, t)
```
where `S` is an `MVector` of length `length_state_vector` to be filled; other
arguments are the same as `flux!` and the same warning concerning `Q` and `aux`
applies.

When `auxiliary_state_initialization! !== nothing` then this is called on the
auxiliary state (assuming `auxiliary_state_length > 0`) as
```
auxiliary_state_initialization!(aux, x1, x2, x3)
```
where `aux` is an `MArray` to fill with the auxiliary state for a DOF located at
Cartesian coordinate locations `(x1, x2, x3)`; see also
[`grad_auxiliary_state!`](@ref) allows the user to take the gradient of a field
stored in the auxiliary state.

When viscous terms are needed, the user must specify values for the following
keyword arguments:
- `number_gradient_states` (`Int`)
- `number_viscous_states` (`Int`)
- `gradient_transform!` (`Function`)
- `viscous_transform!` (`Function`)
- `viscous_penalty!` (`Function`)
- `viscous_boundary_penalty!` (`Function`); only required if the topology has a
  boundary

The function `gradient_transform!` is the implementation of the function `G` in
the module docs; see [`DGBalanceLawDiscretizations`](@ref).  It is called on
each DOF as:
```
gradient_transform!(G, Q, aux, t)
```
where `G` is an `MVector` of length `number_gradient_states` to be filled, `Q`
is an `MVector` containing the states, `aux` is the full auxiliary state at the
DOF, and `t` is the simulation time.data

The function `viscous_transform!` is the implementation of the function `H` in
the module docs; see [`DGBalanceLawDiscretizations`](@ref). It transforms the
gradient ``∇G`` and ``q`` into the viscous state ``σ``. It is called on each DOF
as:
```
viscous_transform!(V, gradG, Q, aux, t)
```
where `V` is an `MVector` of length `number_viscous_states` to be filled,
`gradG` is an `MMatrix` containing the DG-gradient of ``G``, `Q` is an `MVector`
containing the states, `aux` is the full auxiliary state at the DOF, and `t` is
the simulation time. Note that `V` is a vector not a matrix so that minimal
storage can be used if symmetry can be exploited.

The function `viscous_penalty!` is the penalty terms to be used for the
DG-gradient calculation. It is called with data from two neighbouring degrees of
freedom as
```
viscous_penalty!(V, nM, GM, QM, auxM, GP, QP, auxP, t)
```
where:
- `V` is an `MVector` of length `number_viscous_states` to be filled with the
  numerical penalty across the face; see below.
- `nM` is the unit outward normal to the face with respect to the minus side
  (`MVector` of length `3`)
- `GM` and `GP` are the minus and plus evaluation of `gradient_transform!` on
  either side of the face
- `QM` and `QP` are the minus and plus side states (`MArray`)
- `auxM` and `auxP` are the auxiliary states (`MArray`)
- `t` is the current simulation time
The viscous penalty function should compute on the faces
```math
n^{-} \\cdot H^{*} - n^{-} \\cdot H^{-}
```
where ``n^{-} \\cdot H^{*}`` is the "numerical-flux" for the viscous state
computation and ``H^{-}`` is the value of `viscous_transform!` evaluated on the
minus side with ``n^{-} \\cdot G^{-}`` as an argument.

If `grid.topology` has a boundary then the function `viscous_boundary_penalty!`
must be specified. This function is called with the data from the neighbouring
DOF as
```
viscous_boundary_penalty!(V, nM, GM, QM, auxM, GP, QP, auxP, bctype, t)
```
where the required behaviour mimics that of `viscous_penalty!` and
`numerical_boundary_flux!`.

If `preodefun!` is called right before the rest of the ODE function, with the
main purpose to allow the user to populate/modify the auxiliary state
`disc.auxstate` to be consistent with the current time `t` and solution vector
`Q`
```
preodefun!(disc, Q, t)
```
where `disc` is the `DGBalanceLaw` structure and `Q` is the current state being
used to evaluate the ODE function.

!!! note "notes on `preodefun!`"

    Unlike the other callbacks, this function is not called at the device (or
    kernel) level but the host level.

    MPI communication of `Q` occurs after the `odefun!` and no MPI communication
    of `auxstate` is performed (if this is needed we will need to determine a
    way to handle it in order to overlap communication and computation as well
    only comm update fields).

!!! note

    If `(x1, x2, x3)`, or data derived from this such as spherical coordinates, is
    needed in the flux or source the user is responsible to storing this in the
    auxiliary state

"""
function DGBalanceLaw(;grid::DiscontinuousSpectralElementGrid,
                      length_state_vector, flux!,
                      numerical_flux!,
                      numerical_boundary_flux! = nothing,
                      number_gradient_states=0,
                      number_viscous_states=0,
                      gradient_transform! = nothing,
                      viscous_transform! = nothing,
                      viscous_penalty! = nothing,
                      viscous_boundary_penalty! = nothing,
                      auxiliary_state_length=0,
                      auxiliary_state_initialization! = nothing,
                      source! = nothing,
                      preodefun! = nothing)

  topology = grid.topology
  Np = dofs_per_element(grid)
  h_vgeo = Array(grid.vgeo)
  FT = eltype(h_vgeo)
  DA = arraytype(grid)

  (Topologies.hasboundary(topology) &&
   numerical_boundary_flux! === nothing &&
   error("no `numerical_boundary_flux!` given when topology "*
         "has boundary"))

  if number_viscous_states > 0 || number_gradient_states > 0

    # These should all be true in this case
    @assert number_viscous_states > 0
    @assert number_gradient_states > 0
    @assert gradient_transform! !== nothing
    @assert viscous_transform! !== nothing
    @assert viscous_penalty! !== nothing
    (Topologies.hasboundary(topology)) && (@assert viscous_boundary_penalty! !==
                                           nothing)
  end

  weights = view(h_vgeo, :, grid.Mid, :)
  weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

  # TODO: Clean up this MPIStateArray interface...
  Qvisc = MPIStateArray{FT}(topology.mpicomm, DA, Np, number_viscous_states,
                            length(topology.elems),
                            realelems=topology.realelems,
                            ghostelems=topology.ghostelems,
                            vmaprecv=grid.vmaprecv, vmapsend=grid.vmapsend,
                            nabrtorank=topology.nabrtorank,
                            nabrtovmaprecv=grid.nabrtovmaprecv,
                            nabrtovmapsend=grid.nabrtovmapsend,
                            weights=weights, commtag=111)

  auxstate = MPIStateArray{FT}(topology.mpicomm, DA, Np, auxiliary_state_length,
                               length(topology.elems),
                               realelems=topology.realelems,
                               ghostelems=topology.ghostelems,
                               vmaprecv=grid.vmaprecv,
                               vmapsend=grid.vmapsend,
                               nabrtorank=topology.nabrtorank,
                               nabrtovmaprecv=grid.nabrtovmaprecv,
                               nabrtovmapsend=grid.nabrtovmapsend,
                               weights=weights, commtag=222)

  if auxiliary_state_initialization! !== nothing
    @assert auxiliary_state_length > 0
    dim = dimensionality(grid)
    N = polynomialorder(grid)
    vgeo = grid.vgeo
    device = typeof(auxstate.data) <: Array ? CPU() : CUDA()
    nrealelem = length(topology.realelems)
    @launch(device, threads=(Np,), blocks=nrealelem,
            initauxstate!(Val(dim), Val(N), Val(auxiliary_state_length),
                          auxiliary_state_initialization!, auxstate.data, vgeo,
                          topology.realelems))
    MPIStateArrays.start_ghost_exchange!(auxstate)
    MPIStateArrays.finish_ghost_exchange!(auxstate)
  end

  DGBalanceLaw(grid, length_state_vector, flux!,
               numerical_flux!, numerical_boundary_flux!,
               Qvisc, number_gradient_states, number_viscous_states,
               gradient_transform!, viscous_transform!, viscous_penalty!,
               viscous_boundary_penalty!, auxstate, source!, preodefun!)
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
  FT = eltype(h_vgeo)
  Np = dofs_per_element(grid)
  DA = arraytype(grid)

  weights = view(h_vgeo, :, grid.Mid, :)
  weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

  MPIStateArray{FT}(topology.mpicomm, DA, Np, nstate, length(topology.elems),
                    realelems=topology.realelems,
                    ghostelems=topology.ghostelems, vmaprecv=grid.vmaprecv,
                    vmapsend=grid.vmapsend, nabrtorank=topology.nabrtorank,
                    nabrtovmaprecv=grid.nabrtovmaprecv,
                    nabrtovmapsend=grid.nabrtovmapsend, weights=weights,
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
function MPIStateArrays.MPIStateArray(disc::DGBalanceLaw,
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
  h_Q = Array(Q.data)
  h_auxstate = Array(auxstate.data)

  @launch(device, threads=(Np,), blocks=nrealelem,
          initstate!(Val(dim), Val(N), Val(nvar), Val(nauxstate),
                     ic!, h_Q, h_auxstate, h_vgeo, topology.realelems))

  copyto!(Q.data, h_Q)

  MPIStateArrays.start_ghost_exchange!(Q)
  MPIStateArrays.finish_ghost_exchange!(Q)

  Q
end

"""
    MPIStateArray(initialization!::Function, disc::DGBalanceLaw; commtag=888)

Wrapper function to allow for calls of the form

```
MPIStateArray(disc) do  Q, x1, x2, x3
  # fill Q
end
```

See also [`MPIStateArray`](@ref)
"""
MPIStateArrays.MPIStateArray(f::Function,
                             d::DGBalanceLaw; commtag=888
                            ) = MPIStateArray(d, f; commtag=commtag)


"""
    odefun!(disc::DGBalanceLaw, dQ::MPIStateArray, Q::MPIStateArray, t; increment)

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
function SpaceMethods.odefun!(disc::DGBalanceLaw, dQ::MPIStateArray,
                              Q::MPIStateArray, ::Nothing, t; increment)

  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  grid = disc.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Nfp = Nq * Nqk
  nrealelem = length(topology.realelems)

  Qvisc = disc.Qvisc
  auxstate = disc.auxstate

  nstate = disc.nstate
  nviscstate = disc.number_viscous_states
  ngradstate = disc.number_gradient_states
  nauxstate = size(auxstate, 2)

  lgl_weights_vec = grid.ω
  Dmat = grid.D
  vgeo = grid.vgeo
  sgeo = grid.sgeo
  vmap⁻ = grid.vmap⁻
  vmap⁺ = grid.vmap⁺
  elemtobndy = grid.elemtobndy

  ################################
  # Allow the user to update aux #
  ################################
  disc.preodefun! !== nothing && disc.preodefun!(disc, Q, t)

  ########################
  # Gradient Computation #
  ########################
  MPIStateArrays.start_ghost_exchange!(Q)

  if nviscstate > 0

    @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
            volumeviscterms!(Val(dim), Val(N), Val(nstate), Val(ngradstate),
                             Val(nviscstate), Val(nauxstate),
                             disc.viscous_transform!, disc.gradient_transform!,
                             Q.data, Qvisc.data, auxstate.data, vgeo, t, Dmat,
                             topology.realelems))

    MPIStateArrays.finish_ghost_recv!(Q)

    @launch(device, threads=Nfp, blocks=nrealelem,
            faceviscterms!(Val(dim), Val(N), Val(nstate), Val(ngradstate),
                           Val(nviscstate), Val(nauxstate),
                           disc.viscous_penalty!,
                           disc.viscous_boundary_penalty!,
                           disc.gradient_transform!, Q.data, Qvisc.data, auxstate.data,
                           vgeo, sgeo, t, vmap⁻, vmap⁺, elemtobndy,
                           topology.realelems))

    MPIStateArrays.start_ghost_exchange!(Qvisc)
  end

  ###################
  # RHS Computation #
  ###################

  @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
          volumerhs!(Val(dim), Val(N), Val(nstate), Val(nviscstate),
                     Val(nauxstate), disc.flux!, disc.source!, dQ.data, Q.data,
                     Qvisc.data, auxstate.data, vgeo, t, lgl_weights_vec, Dmat,
                     topology.realelems, increment))

  MPIStateArrays.finish_ghost_recv!(nviscstate > 0 ? Qvisc : Q)

  # The main reason for this protection is not for the MPI.Waitall!, but the
  # make sure that we do not recopy data to the GPU
  nviscstate > 0 && MPIStateArrays.finish_ghost_recv!(Qvisc)
  nviscstate == 0 && MPIStateArrays.finish_ghost_recv!(Q)

  @launch(device, threads=Nfp, blocks=nrealelem,
          facerhs!(Val(dim), Val(N), Val(nstate), Val(nviscstate),
                   Val(nauxstate), disc.numerical_flux!,
                   disc.numerical_boundary_flux!, dQ.data, Q.data, Qvisc.data,
                   auxstate.data, vgeo, sgeo, t, vmap⁻, vmap⁺, elemtobndy,
                   topology.realelems))

  # Just to be safe, we wait on the sends we started.
  MPIStateArrays.finish_ghost_send!(Qvisc)
  MPIStateArrays.finish_ghost_send!(Q)
end

"""
    grad_auxiliary_state!(disc, i, (ix1, ix2, ix3)

Computes the gradient of a the field `i` of the constant auxiliary state of
`disc` and stores the `x1, x2, x3` compoment in fields `ix1, ix2, ix3` of constant
auxiliary state.

!!! note

    This only computes the element gradient not a DG gradient. If your constant
    auxiliary state is discontinuous this may or may not be what you want!
"""
function grad_auxiliary_state!(disc::DGBalanceLaw, id, (idx1, idx2, idx3))
  grid = disc.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)

  auxstate = disc.auxstate

  nauxstate = size(auxstate, 2)

  @assert nauxstate >= max(id, idx1, idx2, idx3)
  @assert 0 < min(id, idx1, idx2, idx3)
  @assert allunique((idx1, idx2, idx3))

  lgl_weights_vec = grid.ω
  Dmat = grid.D
  vgeo = grid.vgeo

  device = typeof(auxstate.data) <: Array ? CPU() : CUDA()

  nelem = length(topology.elems)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  @launch(device, threads=(Nq, Nq, Nqk), blocks=nelem,
          elem_grad_field!(Val(dim), Val(N), Val(nauxstate), auxstate.data, vgeo,
                           lgl_weights_vec, Dmat, topology.elems,
                           id, idx1, idx2, idx3))
end

"""
    indefinite_stack_integral!(disc, f, Q, out_states, [P=disc.auxstate])

Computes an indefinite line integral along the trailing dimension (`ξ3` in
3-D and `ξ2` in 2-D) up an element stack using state `Q`
```math
∫_{ζ_{0}}^{ζ} f(q; aux, t)
```
and stores the result of the integral in field of `P` indicated by
`out_states`

The syntax of the integral kernel is:
```
f(F, Q, aux)
```
where `F` is an `MVector` of length `length(out_states)`, `Q` and `aux` are
the `MVectors` for the state and auxiliary state at a single degree of freedom.
The function is responsible for filling `F`.

Requires the `isstacked(disc.grid.topology) == true`
"""
function indefinite_stack_integral!(disc::DGBalanceLaw, f, Q, out_states,
                                    P=disc.auxstate)
  grid = disc.grid
  topology = grid.topology
  @assert isstacked(topology)

  dim = dimensionality(grid)
  N = polynomialorder(grid)

  auxstate = disc.auxstate
  nauxstate = size(auxstate, 2)
  nstate = size(Q, 2)

  Imat = grid.Imat
  vgeo = grid.vgeo
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  nelem = length(topology.elems)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)
  @assert nelem == nvertelem * nhorzelem

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_indefinite_stack_integral!(Val(dim), Val(N), Val(nstate),
                                         Val(nauxstate), Val(nvertelem), f, P.data,
                                         Q.data, auxstate.data, vgeo, Imat,
                                         1:nhorzelem, Val(out_states)))
end

"""
    reverse_indefinite_stack_integral!(disc, oustate, instate,
                                       [P=disc.auxstate])

reverse previously computed indefinite integral(s) computed with
`indefinite_stack_integral!` to be
```math
∫_{ζ}^{ζ_{max}} f(q; aux, t)
```

The states `instate[i]` is reverse and stored in `instate[i]`.

Requires the `isstacked(disc.grid.topology) == true`
"""
function reverse_indefinite_stack_integral!(disc::DGBalanceLaw, oustate,
                                            instate, P=disc.auxstate)
  grid = disc.grid
  topology = grid.topology
  @assert isstacked(topology)
  @assert length(oustate) == length(instate)

  dim = dimensionality(grid)
  N = polynomialorder(grid)

  device = typeof(P.data) <: Array ? CPU() : CUDA()

  nelem = length(topology.elems)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)
  @assert nelem == nvertelem * nhorzelem

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_reverse_indefinite_stack_integral!(Val(dim), Val(N),
                                                 Val(nvertelem), P.data,
                                                 1:nhorzelem, Val(oustate),
                                                 Val(instate)))
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

  device = typeof(auxstate.data) <: Array ? CPU() : CUDA()

  nelem = length(topology.elems)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  nrealelem = length(topology.realelems)

  @launch(device, threads=(Np,), blocks=nrealelem,
          knl_dof_iteration!(Val(dim), Val(N), Val(nRstate), Val(nstate),
                             Val(nviscstate), Val(nauxstate), dof_fun!, R.data,
                             Q.data, Qvisc.data, auxstate.data, topology.realelems))
end

end # module
