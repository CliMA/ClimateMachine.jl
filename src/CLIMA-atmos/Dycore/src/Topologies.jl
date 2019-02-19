module Topologies

export BrickTopology, AbstractTopology

import Canary
using MPI
abstract type AbstractTopology{dim} end

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
                         periodicity=ntuple(j->false, length(elemrange)),
                         connectivity=:face, ghostsize=1)

    # We cannot handle anything else right now...
    @assert connectivity == :face
    @assert ghostsize == 1

    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)
    topology = Canary.brickmesh(elemrange, periodicity, part=mpirank+1,
                                numparts=mpisize)
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
