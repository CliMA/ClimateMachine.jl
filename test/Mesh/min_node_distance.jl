using Test
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.VTK
using Logging
using Printf
using LinearAlgebra

let
  # boiler plate MPI stuff
  CLIMA.init()
  ArrayType = CLIMA.array_type()

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  # Mesh generation parameters
  N = 4
  Nq = N+1
  Neh = 10
  Nev = 4

  @testset "$(@__FILE__) DGModel matrix" begin
    for FT in (Float64, Float32)
      for dim = (2, 3)
        if dim == 2
          brickrange = (range(FT(0); length=Neh+1, stop=1),
                        range(FT(1); length=Nev+1, stop=2))
        elseif dim == 3
          brickrange = (range(FT(0); length=Neh+1, stop=1),
                        range(FT(0); length=Neh+1, stop=1),
                        range(FT(1); length=Nev+1, stop=2))
        end

        topl = StackedBrickTopology(mpicomm, brickrange)

        function warpfun(ξ1, ξ2, ξ3)
          FT = eltype(ξ1)

          ξ1 ≥ FT(1//2) && (ξ1 = FT(1//2) + 2(ξ1 - FT(1//2)))
          if dim == 2
            ξ2 ≥ FT(3//2) && (ξ2 = FT(3//2) + 2(ξ2 - FT(3//2)))
          elseif dim == 3
            ξ2 ≥ FT(1//2) && (ξ2 = FT(1//2) + 2(ξ2 - FT(1//2)))
            ξ3 ≥ FT(3//2) && (ξ3 = FT(3//2) + 2(ξ3 - FT(3//2)))
          end
          (ξ1, ξ2, ξ3)
        end

        grid = DiscontinuousSpectralElementGrid(topl,
                                                FloatType = FT,
                                                DeviceArray = ArrayType,
                                                polynomialorder = N,
                                                meshwarp = warpfun)

        # testname = "grid_poly$(N)_dim$(dim)_$(ArrayType)_$(FT)"
        # filename(rank) = @sprintf("%s_mpirank%04d", testname, rank)
        # writevtk(filename(MPI.Comm_rank(mpicomm)), grid)
        # if MPI.Comm_rank(mpicomm) == 0
        #   writepvtu(testname, filename.(0:MPI.Comm_size(mpicomm)-1), ())
        # end

        ξ = referencepoints(grid)
        hmnd = (ξ[2] - ξ[1]) / (2Neh)
        vmnd = (ξ[2] - ξ[1]) / (2Nev)

        @test hmnd ≈ min_node_distance(grid, EveryDirection())
        @test vmnd ≈ min_node_distance(grid, VerticalDirection())
        @test hmnd ≈ min_node_distance(grid, HorizontalDirection())

      end
    end
end
end

nothing
