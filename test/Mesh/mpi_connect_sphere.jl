using Test
using MPI
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.MPIStateArrays

function main()
  FT = Float64
  Nhorz = 3
  Nstack = 5
  N = 4
  DA = Array

  MPI.Initialized() || MPI.Init()

  comm = MPI.COMM_WORLD
  crank = MPI.Comm_rank(comm)
  csize = MPI.Comm_size(comm)

  Rrange = FT.(accumulate(+,1:Nstack+1))
  topology = StackedCubedSphereTopology(MPI.COMM_SELF, Nhorz, Rrange;
                                        boundary=(1,2))
  grid = DiscontinuousSpectralElementGrid(topology; FloatType=FT,
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

  # Check x1x2x3 matches before comm
  x1 = @view grid.vgeo[:, Grids._x1, :]
  x2 = @view grid.vgeo[:, Grids._x2, :]
  x3 = @view grid.vgeo[:, Grids._x3, :]

  @test x1[grid.vmap⁻] ≈ x1[grid.vmap⁺]
  @test x2[grid.vmap⁻] ≈ x2[grid.vmap⁺]
  @test x3[grid.vmap⁻] ≈ x3[grid.vmap⁺]

  Np = (N+1)^3
  x1x2x3 = MPIStateArray{FT}(topology.mpicomm, DA, Np, 3,
                             length(topology.elems),
                             realelems=topology.realelems,
                             ghostelems=topology.ghostelems,
                             vmaprecv=grid.vmaprecv,
                             vmapsend=grid.vmapsend,
                             nabrtorank=topology.nabrtorank,
                             nabrtovmaprecv=grid.nabrtovmaprecv,
                             nabrtovmapsend=grid.nabrtovmapsend)
  x1x2x3.data[:,:,topology.realelems] .=
        @view grid.vgeo[:, [Grids._x1, Grids._x2, Grids._x3], topology.realelems]
  MPIStateArrays.start_ghost_exchange!(x1x2x3)
  MPIStateArrays.finish_ghost_exchange!(x1x2x3)

  # Check x1x2x3 matches after
  x1 = @view x1x2x3.data[:, 1, :]
  x2 = @view x1x2x3.data[:, 2, :]
  x3 = @view x1x2x3.data[:, 3, :]

  @test x1[grid.vmap⁻] ≈ x1[grid.vmap⁺]
  @test x2[grid.vmap⁻] ≈ x2[grid.vmap⁺]
  @test x3[grid.vmap⁻] ≈ x3[grid.vmap⁺]

  nothing
end
isinteractive() || main()
