module Topologies

export BrickTopology, AbstractTopology, AbstractStackedTopology

import Canary
using MPI
abstract type AbstractTopology{dim} end
abstract type AbstractStackedTopology{dim} <: AbstractTopology{dim} end

"""
    BrickTopology{dim, T}(mpicomm, elemrange; boundary, periodicity)

Generate a brick mesh topology with coordinates given by the tuple `elemrange`
and the periodic dimensions given by the `periodicity` tuple.

The elements of the brick are partitioned equally across the MPI ranks based
on a space-filling curve.

By default boundary faces will be marked with a one and other faces with a
zero.  Specific boundary numbers can also be passed for each face of the brick
in `boundary`.  This will mark the nonperiodic brick faces with the given
boundary number.

# Examples

We can build a 3 by 2 element two-dimensional mesh that is periodic in the
\$x_2\$-direction with
```jldoctest brickmesh

using CLIMAAtmosDycore
using CLIMAAtmosDycore.Topologies
using MPI
MPI.Init()
topology = BrickTopology(MPI.COMM_SELF, (2:5,4:6);
                         periodicity=(false,true),
                         boundary=[1 3; 2 4])
MPI.Finalize()
```
This returns the mesh structure for

             x_2

              ^
              |
             6-  +-----+-----+-----+
              |  |     |     |     |
              |  |  3  |  4  |  5  |
              |  |     |     |     |
             5-  +-----+-----+-----+
              |  |     |     |     |
              |  |  1  |  2  |  6  |
              |  |     |     |     |
             4-  +-----+-----+-----+
              |
              +--|-----|-----|-----|--> x_1
                 2     3     4     5

For example, the (dimension by number of corners by number of elements) array
`elemtocoord` gives the coordinates of the corners of each element.
```jldoctes brickmesh
julia> topology.elemtocoord
2×4×6 Array{Int64,3}:
[:, :, 1] =
 2  3  2  3
 4  4  5  5

[:, :, 2] =
 3  4  3  4
 4  4  5  5

[:, :, 3] =
 2  3  2  3
 5  5  6  6

[:, :, 4] =
 3  4  3  4
 5  5  6  6

[:, :, 5] =
 4  5  4  5
 5  5  6  6

[:, :, 6] =
 4  5  4  5
 4  4  5  5
```
Note that the corners are listed in Cartesian order.

The (number of faces by number of elements) array `elemtobndy` gives the
boundary number for each face of each element.  A zero will be given for
connected faces.
```jldoctest brickmesh
julia> topology.elemtobndy
4×6 Array{Int64,2}:
 1  0  1  0  0  0
 0  0  0  0  2  2
 0  0  0  0  0  0
 0  0  0  0  0  0
```
Note that the faces are listed in Cartesian order.
"""
struct BrickTopology{dim, T} <: AbstractTopology{dim}
  """
  mpi communicator use for spatial discretization are using
  """
  mpicomm::MPI.Comm

  """
  range of element indices
  """
  elems::UnitRange{Int64}

  """
  range of real (aka nonghost) element indices
  """
  realelems::UnitRange{Int64}

  """
  range of ghost element indices
  """
  ghostelems::UnitRange{Int64}

  """
  array of send element indices sorted so that
  """
  sendelems::Array{Int64, 1}

  """
  element to vertex coordinates

  `elemtocoord[d,i,e]` is the `d`th coordinate of corner `i` of element `e`

  !!! note
  currently coordinates always are of size 3 for `(x, y, z)`
  """
  elemtocoord::Array{T, 3}

  """
  element to neighboring element; `elemtoelem[f,e]` is the number of the element
  neighboring element `e` across face `f`.  If there is no neighboring element
  then `elemtoelem[f,e] == e`.
  """
  elemtoelem::Array{Int64, 2}

  """
  element to neighboring element face; `elemtoface[f,e]` is the face number of
  the element neighboring element `e` across face `f`.  If there is no
  neighboring element then `elemtoface[f,e] == f`."
  """
  elemtoface::Array{Int64, 2}

  """
  element to neighboring element order; `elemtoordr[f,e]` is the ordering number
  of the element neighboring element `e` across face `f`.  If there is no
  neighboring element then `elemtoordr[f,e] == 1`.
  """
  elemtoordr::Array{Int64, 2}

  """
  element to bounday number; `elemtobndy[f,e]` is the boundary number of face
  `f` of element `e`.  If there is a neighboring element then `elemtobndy[f,e]
  == 0`.
  """
  elemtobndy::Array{Int64, 2}

  """
  list of the MPI ranks for the neighboring processes
  """
  nabrtorank::Array{Int64, 1}

  """
  range in ghost elements to receive for each neighbor
  """
  nabrtorecv::Array{UnitRange{Int64}, 1}

  """
  range in `sendelems` to send for each neighbor
  """
  nabrtosend::Array{UnitRange{Int64}, 1}

  BrickTopology(mpicomm, Nelems::NTuple{N, Integer}; kw...) where N =
  BrickTopology(mpicomm, map(Ne->0:Ne, Nelems); kw...)

  function BrickTopology(mpicomm, elemrange;
                         boundary=ones(Int,2,length(elemrange)),
                         periodicity=ntuple(j->false, length(elemrange)),
                         connectivity=:face, ghostsize=1)

    # We cannot handle anything else right now...
    @assert connectivity == :face
    @assert ghostsize == 1

    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)
    topology = Canary.brickmesh(elemrange, periodicity, part=mpirank+1,
                                numparts=mpisize, boundary=boundary)
    topology = Canary.partition(mpicomm, topology...)
    topology = Canary.connectmesh(mpicomm, topology...)

    dim = length(elemrange)
    T = eltype(topology.elemtocoord)
    new{dim, T}(mpicomm, topology.elems, topology.realelems,
                topology.ghostelems, topology.sendelems, topology.elemtocoord,
                topology.elemtoelem, topology.elemtoface, topology.elemtoordr,
                topology.elemtobndy, topology.nabrtorank, topology.nabrtorecv,
                topology.nabrtosend)
  end
end

end
