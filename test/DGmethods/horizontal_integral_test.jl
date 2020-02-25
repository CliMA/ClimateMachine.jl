using MPI
using StaticArrays
using Logging
using Printf
using LinearAlgebra
using GPUifyLoops
using Test
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.MPIStateArrays
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters

function run_test1(mpicomm, dim, Ne, N, FT, ArrayType)
  warpfun = (ξ1, ξ2, ξ3) -> begin
    x1 = ξ1 + (ξ1 - 1/2) * cos(2 * π * ξ2 * ξ3) / 4
    x2 = ξ2  + (ξ2 - 1/2) * cos(2 * π * ξ2 * ξ3) / 4
    x3 = ξ3 + ξ1 / 4 + sin(2 * π * ξ1) / 16
    return (x1, x2, x3)
  end

  brickrange = (range(FT(0); length=Ne+1, stop=1),
                range(FT(0); length=Ne+1, stop=1),
                range(FT(0); length=2, stop=1))
  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity = (false, false, false))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = warpfun,
                                         )

  nrealelem = length(topl.realelems)
  Nq = N + 1
  Nqk = dimensionality(grid) == 2 ? 1 : Nq
  vgeo = grid.vgeo
  host_array = Array ∈ typeof(vgeo).parameters
  localvgeo = host_array ? vgeo : Array(vgeo)

  S = zeros(Nqk)
  S1 = zeros(Nqk)

  for e in 1:nrealelem
    for k in 1:Nqk
      for j in 1:Nq
        for i in 1:Nq
          ijk = i + Nq * ((j-1)+ Nq * (k-1))
          S[k] += localvgeo[ijk,grid.x1id,e] * localvgeo[ijk,grid.MHid,e]
          S1[k] += localvgeo[ijk,grid.MHid,e]
        end
      end
    end
  end

  Stot = zeros(Nqk)
  S1tot = zeros(Nqk)
  Err = 0.0

  for k in 1:Nqk
    Stot[k] = MPI.Reduce(S[k], +, 0, mpicomm)
    S1tot[k] = MPI.Reduce(S1[k], +, 0, mpicomm)
    Err += (0.5 - Stot[k] / S1tot[k])^2
  end
  Err = sqrt(Err / Nqk)
  @test Err < 2e-15
end

function run_test2(mpicomm, dim, Ne, N, FT, ArrayType)
  warpfun = (ξ1, ξ2, ξ3) -> begin
    x1 = sin(2 * π * ξ3)/16 + ξ1
    x2 = ξ2  + (ξ2 - 1/2) * cos(2 * π * ξ2 * ξ3) / 4
    x3 = ξ3 + sin(2π * (ξ1))/20
    return (x1, x2, x3)
  end

  brickrange = (range(FT(0); length=Ne+1, stop=1),
                range(FT(0); length=Ne+1, stop=1),
                range(FT(0); length=2, stop=1))
  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity = (false, false, false))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = warpfun,
                                         )

  nrealelem = length(topl.realelems)
  Nq = N + 1
  Nqk = dimensionality(grid) == 2 ? 1 : Nq
  vgeo = grid.vgeo
  host_array = Array ∈ typeof(vgeo).parameters
  localvgeo = host_array ? vgeo : Array(vgeo)

  S = zeros(Nqk)
  S1 = zeros(Nqk)
  K = zeros(Nqk)

  for e in 1:nrealelem
    for k in 1:Nqk
      for j in 1:Nq
        for i in 1:Nq
          ijk = i + Nq * ((j-1)+ Nq * (k-1))
          S[k] += localvgeo[ijk,grid.x1id,e] * localvgeo[ijk,grid.MHid,e]
          S1[k] += localvgeo[ijk,grid.MHid,e]
          K[k] = localvgeo[ijk,grid.x3id,e]
        end
      end
    end
  end

  Stot = zeros(Nqk)
  S1tot = zeros(Nqk)
  Err = 0.0

  for k in 1:Nqk
    Stot[k] = MPI.Reduce(S[k], +, 0, mpicomm)
    S1tot[k] = MPI.Reduce(S1[k], +, 0, mpicomm)
    Err += (0.5 + sin(2 * π * K[k]) / 16 - Stot[k] / S1tot[k])^2
  end
  Err = sqrt(Err / Nqk)
  @test 2e-15 > Err
end

function run_test3(mpicomm, dim, Ne, N, FT, ArrayType)
  base_Nhorz = 4
  base_Nvert = 2
  Rinner = 1 // 2
  Router = 1
  expected_result = [-4.5894269717905445e-8  -1.1473566985387151e-8 ;
		                 -2.0621904184281448e-10 -5.155431637149377e-11 ;
	                   -8.72191208145523e-13   -2.1715962361668062e-13]
  for l in 1:3
    Nhorz = 2^(l-1) * base_Nhorz
    Nvert = 2^(l-1) * base_Nvert
    Rrange = grid1d(FT(Rinner), FT(Router); nelem=Nvert)
    topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)
    grid = DiscontinuousSpectralElementGrid(topl,
                                            FloatType = FT,
                                            DeviceArray = ArrayType,
                                            polynomialorder = N,
                                            meshwarp = Topologies.cubedshellwarp,
                                           )

    nrealelem = length(topl.realelems)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    vgeo = grid.vgeo
    host_array = Array ∈ typeof(vgeo).parameters
    localvgeo = host_array ? vgeo : Array(vgeo)

    topology = grid.topology
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    Surfout = 0
    Surfin = 0

    for ev in 1:nvertelem
      for eh in 1:nhorzelem
        e = ev + (eh - 1) * nvertelem
        for i in 1:Nq
          for j in 1:Nq
            for k in 1:Nqk
              ijk = i + Nq * ((j-1) + Nqk * (k-1))
              if (k == Nqk && ev == nvertelem)
                Surfout += localvgeo[ijk,grid.MHid,e]
              end
              if (k == 1 && ev == 1)
                Surfin += localvgeo[ijk,grid.MHid,e]
              end
            end
          end
        end
      end
    end
    Surfouttot = MPI.Reduce(Surfout, +, 0, MPI.COMM_WORLD)
    Surfintot = MPI.Reduce(Surfin, +, 0, MPI.COMM_WORLD)
    @test (4 * π * Router^2 - Surfouttot) ≈ expected_result[l,1] rtol=1e-3 atol=eps(FT) * 4 * π * Router^2
    @test (4 * π * Rinner^2 - Surfintot)  ≈ expected_result[l,2] rtol=1e-3 atol=eps(FT) * 4 * π * Rinner^2
  end
end

# Test for 2D integral
function run_test4(mpicomm, dim, Ne, N, FT, ArrayType)
  brickrange = ntuple(j->range(FT(0); length=Ne+1, stop=1), 2)
  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity=ntuple(j->true, 2))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  nrealelem = length(topl.realelems)
  Nq = N + 1
  vgeo = grid.vgeo
  host_array = Array ∈ typeof(vgeo).parameters
  localvgeo = host_array ? vgeo : Array(vgeo)

  S = zeros(Nq)

  for e in 1:nrealelem
    for i in 1:Nq
      for j in 1:Nq
        ij = i + Nq * (j - 1)
        S[j] += localvgeo[ij,grid.x1id,e] * localvgeo[ij,grid.MHid,e]
      end
    end
  end

  Err = 0
  for j in 1:Nq
    Err += (S[j] - 0.5)^2
  end

  Err = sqrt(Err / Nq)
  @test Err <= 1e-15
end

function run_test5(mpicomm, dim, Ne, N, FT, ArrayType)
  warpfun = (ξ1, ξ2, ξ3) -> begin
    x1 = cos(π * ξ2)/16 + abs(ξ1)
    x2 = ξ2
    x3 = ξ3
    return (x1, x2, x3)
  end

  brickrange = ntuple(j->range(FT(0); length=Ne+1, stop=1), 2)
  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity=ntuple(j->true, 2))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
					  meshwarp = warpfun,
                                         )

  nrealelem = length(topl.realelems)
  Nq = N + 1
  vgeo = grid.vgeo
  host_array = Array ∈ typeof(vgeo).parameters
  localvgeo = host_array ? vgeo : Array(vgeo)

  S = zeros(Nq)
  J = zeros(Nq)

  for e in 1:nrealelem
    for i in 1:Nq
      for j in 1:Nq
        ij = i + Nq * (j - 1)
        S[j] += localvgeo[ij,grid.x1id,e] * localvgeo[ij,grid.MHid,e]
        J[j] = localvgeo[ij,grid.x2id,e]
      end
    end
  end

  Err = 0
  for j in 1:Nq
    Err += (S[j] - 0.5 - cos(π * J[j]) / 16)^2
  end

  Err = sqrt(Err / Nq)
  @test Err < 1e-15
end

let
  CLIMA.init()
  ArrayType = CLIMA.array_type()

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  FT = Float64
  dim = 3
  Ne = 1
  polynomialorder = 4

  @info (ArrayType, FT, dim)

  @testset "horizontal_integral" begin
    run_test1(mpicomm, dim, Ne, polynomialorder, FT, ArrayType)
    run_test2(mpicomm, dim, Ne, polynomialorder, FT, ArrayType)
    run_test3(mpicomm, dim, Ne, polynomialorder, FT, ArrayType)
    run_test4(mpicomm, dim, Ne, polynomialorder, FT, ArrayType)
    run_test5(mpicomm, dim, Ne, polynomialorder, FT, ArrayType)
  end
end

