using Test
using MPI
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.MPIStateArrays

function main()
  T = Float64
  Nhorz = 3
  Nstack = 5
  N = 4
  DA = Array

  MPI.Initialized() || MPI.Init()

  comm = MPI.COMM_WORLD
  crank = MPI.Comm_rank(comm)
  csize = MPI.Comm_size(comm)

  Rrange = T.(accumulate(+,1:Nstack+1))
  topology = StackedCubedSphereTopology(MPI.COMM_SELF, Nhorz, Rrange;
                                        boundary=(1,2))
  grid = DiscontinuousSpectralElementGrid(topology; FloatType=T,
                                          DeviceArray=DA, polynomialorder=N,
                                          meshwarp=Topologies.cubedshellwarp)

  #=
  @show elems       = topology.elems
  @show realelems   = topology.realelems
  @show ghostelems  = topology.ghostelems
  @show sendelems   = topology.sendelems
  @show elemtocoord = topology.elemtocoord
  @show elemtoelem  = topology.elemtoelem
  @show elemtoface  = topology.elemtoface
  @show elemtoordr  = topology.elemtoordr
  @show elemtobndy  = topology.elemtobndy
  @show nabrtorank  = topology.nabrtorank
  @show nabrtorecv  = topology.nabrtorecv
  @show nabrtosend  = topology.nabrtosend
  =#

  # Check xyz matches before comm
  x = @view grid.vgeo[:, Grids._x, :]
  y = @view grid.vgeo[:, Grids._y, :]
  z = @view grid.vgeo[:, Grids._z, :]

  @test x[grid.vmapM] ≈ x[grid.vmapP]
  @test y[grid.vmapM] ≈ y[grid.vmapP]
  @test z[grid.vmapM] ≈ z[grid.vmapP]

  Np = (N+1)^3
  xyz = MPIStateArray{Tuple{Np, 3}, T, DA}(topology.mpicomm,
                                           length(topology.elems),
                                           realelems=topology.realelems,
                                           ghostelems=topology.ghostelems,
                                           sendelems=topology.sendelems,
                                           nabrtorank=topology.nabrtorank,
                                           nabrtorecv=topology.nabrtorecv,
                                           nabrtosend=topology.nabrtosend)
  xyz.Q[:,:,topology.realelems] .=
        @view grid.vgeo[:, [Grids._x, Grids._y, Grids._z], topology.realelems]
  MPIStateArrays.start_ghost_exchange!(xyz)
  MPIStateArrays.finish_ghost_exchange!(xyz)

  # Check xyz matches after
  x = @view xyz.Q[:, 1, :]
  y = @view xyz.Q[:, 2, :]
  z = @view xyz.Q[:, 3, :]

  @test x[grid.vmapM] ≈ x[grid.vmapP]
  @test y[grid.vmapM] ≈ y[grid.vmapP]
  @test z[grid.vmapM] ≈ z[grid.vmapP]

  nothing
end
isinteractive() || main()
