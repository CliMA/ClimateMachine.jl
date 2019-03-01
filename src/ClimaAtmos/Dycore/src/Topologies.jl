module Topologies

export AbstractTopology, AbstractStackedTopology
export BrickTopology, StackedBrickTopology

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

"""
    StackedBrickTopology{dim, T}(mpicomm, elemrange; boundary, periodicity)

Generate a stacked brick mesh topology with coordinates given by the tuple
`elemrange` and the periodic dimensions given by the `periodicity` tuple.

The elements are stacked such that the elements associated with range
`elemrange[dim]` are contiguous in the element ordering.

The elements of the brick are partitioned equally across the MPI ranks based
on a space-filling curve.  Further, stacks are not split at MPI boundaries.

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
topology = StackedBrickTopology(MPI.COMM_SELF, (2:5,4:6);
                                periodicity=(false,true),
                                boundary=[1 3; 2 4])
MPI.Finalize()
```
This returns the mesh structure stacked in the \$x_2\$-direction for

             x_2

              ^
              |
             6-  +-----+-----+-----+
              |  |     |     |     |
              |  |  2  |  4  |  6  |
              |  |     |     |     |
             5-  +-----+-----+-----+
              |  |     |     |     |
              |  |  1  |  3  |  5  |
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
 2  3  2  3
 5  5  6  6

[:, :, 3] =
 3  4  3  4
 4  4  5  5

[:, :, 4] =
 3  4  3  4
 5  5  6  6

[:, :, 5] =
 4  5  4  5
 4  4  5  5

[:, :, 6] =
 4  5  4  5
 5  5  6  6
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

struct StackedBrickTopology{dim, T} <: AbstractStackedTopology{dim}
  """
  mpi communicator use for spatial discretization are using
  """
  mpicomm::MPI.Comm

  """
  number of elements in a stack
  """
  stacksize::Int64

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

  StackedBrickTopology(mpicomm, Nelems::NTuple{N, Integer}; kw...) where N =
  StackedBrickTopology(mpicomm, map(Ne->0:Ne, Nelems); kw...)

  function StackedBrickTopology(mpicomm, elemrange;
                         boundary=ones(Int,2,length(elemrange)),
                         periodicity=ntuple(j->false, length(elemrange)),
                         connectivity=:face, ghostsize=1)


    dim = length(elemrange)

    dim <= 1 && error("Stacked brick topology works for 2D and 3D")

    # Build the base topology
    basetopo = BrickTopology(mpicomm, elemrange[1:dim-1];
                       boundary=boundary[:,1:dim-1],
                       periodicity=periodicity[1:dim-1],
                       connectivity=connectivity,
                       ghostsize=ghostsize)


    # Use the base topology to build the stacked topology
    stack = elemrange[dim]
    stacksize = length(stack) - 1

    nvert = 2^dim
    nface = 2dim

    nreal = length(basetopo.realelems)*stacksize
    nghost = length(basetopo.ghostelems)*stacksize

    elems=1:(nreal+nghost)
    realelems=1:nreal
    ghostelems=nreal.+(1:nghost)

    sendelems = similar(basetopo.sendelems,
                        length(basetopo.sendelems)*stacksize)
    for i=1:length(basetopo.sendelems), j=1:stacksize
      sendelems[stacksize*(i-1) + j] = stacksize*(basetopo.sendelems[i]-1) + j
    end

    elemtocoord = similar(basetopo.elemtocoord, dim, nvert, length(elems))

    for i=1:length(basetopo.elems), j=1:stacksize
      e = stacksize*(i-1) + j

      for v = 1:2^(dim-1)
        for d = 1:(dim-1)
          elemtocoord[d, v, e] = basetopo.elemtocoord[d, v, i]
          elemtocoord[d, 2^(dim-1) + v, e] = basetopo.elemtocoord[d, v, i]
        end

        elemtocoord[dim, v, e] = stack[j]
        elemtocoord[dim, 2^(dim-1) + v, e] = stack[j+1]
      end
    end

    elemtoelem = similar(basetopo.elemtoelem, nface, length(elems))
    elemtoface = similar(basetopo.elemtoface, nface, length(elems))
    elemtoordr = similar(basetopo.elemtoordr, nface, length(elems))
    elemtobndy = similar(basetopo.elemtobndy, nface, length(elems))

    for e=1:length(basetopo.elems)*stacksize, f=1:nface
      elemtoelem[f, e] = e
      elemtoface[f, e] = f
      elemtoordr[f, e] = 1
      elemtobndy[f, e] = 0
    end

    for i=1:length(basetopo.realelems), j=1:stacksize
      e1 = stacksize*(i-1) + j

      for f = 1:2(dim-1)
        e2 = stacksize*(basetopo.elemtoelem[f, i]-1) + j

        elemtoelem[f, e1] = e2
        elemtoface[f, e1] = basetopo.elemtoface[f, i]

        # We assume a simple orientation right now
        @assert basetopo.elemtoordr[f, i] == 1
        elemtoordr[f, e1] = basetopo.elemtoordr[f, i]
      end

      et = stacksize*(i-1) + j + 1
      eb = stacksize*(i-1) + j - 1
      ft = 2(dim-1) + 1
      fb = 2(dim-1) + 2
      ot = 1
      ob = 1

      if j == stacksize
        et = periodicity[dim] ? stacksize*(i-1) + 1 : e1
        ft = periodicity[dim] ? ft : 2(dim-1) + 2
      elseif j == 1
        eb = periodicity[dim] ? stacksize*(i-1) + stacksize : e1
        fb = periodicity[dim] ? fb : 2(dim-1) + 1
      end

      elemtoelem[2(dim-1)+1, e1] = eb
      elemtoelem[2(dim-1)+2, e1] = et
      elemtoface[2(dim-1)+1, e1] = fb
      elemtoface[2(dim-1)+2, e1] = ft
      elemtoordr[2(dim-1)+1, e1] = ob
      elemtoordr[2(dim-1)+2, e1] = ot
    end

    for i=1:length(basetopo.elems), j=1:stacksize
      e1 = stacksize*(i-1) + j

      for f = 1:2(dim-1)
        elemtobndy[f, e1] = basetopo.elemtobndy[f, i]
      end

      bt = bb = 0

      if j == stacksize
        bt = periodicity[dim] ? bt : boundary[2,dim]
      elseif j == 1
        bb = periodicity[dim] ? bb : boundary[1,dim]
      end

      elemtobndy[2(dim-1)+1, e1] = bb
      elemtobndy[2(dim-1)+2, e1] = bt
    end

    nabrtorank = basetopo.nabrtorank
    nabrtorecv =
      UnitRange{Int}[UnitRange(stacksize*(first(basetopo.nabrtorecv[n])-1)+1,
                               stacksize*last(basetopo.nabrtorecv[n]))
                     for n = 1:length(nabrtorank)]
    nabrtosend =
      UnitRange{Int}[UnitRange(stacksize*(first(basetopo.nabrtosend[n])-1)+1,
                               stacksize*last(basetopo.nabrtosend[n]))
                     for n = 1:length(nabrtorank)]

    T = eltype(basetopo.elemtocoord)
    new{dim, T}(mpicomm, stacksize, elems, realelems, ghostelems, sendelems,
                elemtocoord, elemtoelem, elemtoface, elemtoordr, elemtobndy,
                nabrtorank, nabrtorecv, nabrtosend)
  end
end

end
