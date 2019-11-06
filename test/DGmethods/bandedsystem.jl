using CLIMA: haspkg
using MPI
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.VTK: writemesh
using Logging
using LinearAlgebra
using Random
using StaticArrays
using CLIMA.Atmos: AtmosModel, AtmosAcousticLinearModel,
                   DryModel, NoRadiation, NoFluxBC,
                   ConstantViscosityWithDivergence, IsothermalProfile,
                   HydrostaticState, NoOrientation
using CLIMA.VariableTemplates: flattenednames

using CLIMA.PlanetParameters: T_0
using CLIMA.DGmethods: VerticalDirection, DGModel, Vars, vars_state, num_state,
                       banded_matrix, banded_matrix_vector_product!
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxDiffusive,
                                       CentralGradPenalty
using CLIMA.MPIStateArrays: MPIStateArray, euclidean_distance

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray,)
else
  const ArrayTypes = (Array,)
end

using Test
let
  # boiler plate MPI stuff
  MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  Random.seed!(777 + MPI.Comm_rank(mpicomm))

  # Mesh generation parameters
  N = 4
  Nq = N+1
  Neh = 10
  Nev = 4

  @testset "$(@__FILE__) DGModel matrix" begin
    for AT in ArrayTypes
      for FT in (Float64,)
        for dim = (2, 3)
          for single_column in (false, true)
            # Setup the topology
            if dim == 2
              brickrange = (range(FT(0); length=Neh+1, stop=1),
                            range(FT(1); length=Nev+1, stop=2))
            elseif dim == 3
              brickrange = (range(FT(0); length=Neh+1, stop=1),
                            range(FT(0); length=Neh+1, stop=1),
              range(FT(1); length=Nev+1, stop=2))
            end
            topl = StackedBrickTopology(mpicomm, brickrange)

            # Warp mesh
            function warpfun(ξ1, ξ2, ξ3)
              # single column currently requires no geometry warping

              # Even if the warping is in only the horizontal, the way we
              # compute metrics causes problems for the single column approach
              # (possibly need to not use curl-invariant computation)
              if !single_column
                ξ1 = ξ1 + sin(2π * ξ1 * ξ2) / 10
                ξ2 = ξ2 + sin(2π * ξ1) / 5
                if dim == 3
                  ξ3 = ξ3 + sin(8π * ξ1 * ξ2) / 10
                end
              end
              (ξ1, ξ2, ξ3)
            end

            # create the actual grid
            grid = DiscontinuousSpectralElementGrid(topl,
                                                    FloatType = FT,
                                                    DeviceArray = AT,
                                                    polynomialorder = N,
                                                    meshwarp = warpfun)
            model = AtmosModel(NoOrientation(),
                               HydrostaticState(IsothermalProfile(FT(T_0)),
                                                FT(0)),
            ConstantViscosityWithDivergence(0.0),
            DryModel(),
            NoRadiation(),
            nothing,
            NoFluxBC(),
            nothing)
            linear_model = AtmosAcousticLinearModel(model)

            # the nonlinear model is needed so we can grab the auxstate below
            dg = DGModel(model,
                         grid,
                         Rusanov(),
                         CentralNumericalFluxDiffusive(),
                         CentralGradPenalty())
            dg_linear = DGModel(linear_model,
                                grid,
                                Rusanov(),
                                CentralNumericalFluxDiffusive(),
                                CentralGradPenalty();
                                direction=VerticalDirection(),
                                auxstate=dg.auxstate)

            A_banded = banded_matrix(dg_linear; single_column=single_column)

            Q = MPIStateArray(dg_linear)
            dQ1 = MPIStateArray(dg_linear)
            dQ2 = MPIStateArray(dg_linear)
            Q.data .= rand(size(Q.data))
            dg_linear(dQ1, Q, nothing, 0; increment=false)

            banded_matrix_vector_product!(dg_linear, A_banded, dQ2, Q)
            @test dQ1.realdata ≈ dQ2.realdata
          end
        end
      end
    end
  end

  @testset "$(@__FILE__) linear operator matrix" begin
    for AT in ArrayTypes
      for FT in (Float64,)
        for dim = (2, 3)
          for single_column in (false, true)
            # Setup the topology
            if dim == 2
              brickrange = (range(FT(0); length=Neh+1, stop=1),
                            range(FT(1); length=Nev+1, stop=2))
            elseif dim == 3
              brickrange = (range(FT(0); length=Neh+1, stop=1),
                            range(FT(0); length=Neh+1, stop=1),
              range(FT(1); length=Nev+1, stop=2))
            end
            topl = StackedBrickTopology(mpicomm, brickrange)

            # Warp mesh
            function warpfun(ξ1, ξ2, ξ3)
              # single column currently requires no geometry warping

              # Even if the warping is in only the horizontal, the way we
              # compute metrics causes problems for the single column approach
              # (possibly need to not use curl-invariant computation)
              if !single_column
                ξ1 = ξ1 + sin(2π * ξ1 * ξ2) / 10
                ξ2 = ξ2 + sin(2π * ξ1) / 5
                if dim == 3
                  ξ3 = ξ3 + sin(8π * ξ1 * ξ2) / 10
                end
              end
              (ξ1, ξ2, ξ3)
            end

            # create the actual grid
            grid = DiscontinuousSpectralElementGrid(topl,
                                                    FloatType = FT,
                                                    DeviceArray = AT,
                                                    polynomialorder = N,
                                                    meshwarp = warpfun)
            model = AtmosModel(NoOrientation(),
                               HydrostaticState(IsothermalProfile(FT(T_0)),
                                                FT(0)),
            ConstantViscosityWithDivergence(0.0),
            DryModel(),
            NoRadiation(),
            nothing,
            NoFluxBC(),
            nothing)
            linear_model = AtmosAcousticLinearModel(model)

            # the nonlinear model is needed so we can grab the auxstate below
            dg = DGModel(model,
                         grid,
                         Rusanov(),
                         CentralNumericalFluxDiffusive(),
                         CentralGradPenalty())
            dg_linear = DGModel(linear_model,
                                grid,
                                Rusanov(),
                                CentralNumericalFluxDiffusive(),
                                CentralGradPenalty();
                                direction=VerticalDirection(),
                                auxstate=dg.auxstate)

            α = FT(1 // 10)
            function op!(LQ, Q)
              dg_linear(LQ, Q, nothing, 0; increment=false)
              @. LQ = Q + α * LQ
            end

            A_banded = banded_matrix(op!, dg_linear;
                                     single_column=single_column)

            Q = MPIStateArray(dg_linear)
            dQ1 = MPIStateArray(dg_linear)
            dQ2 = MPIStateArray(dg_linear)
            Q.data .= rand(size(Q.data))
            op!(dQ1, Q)
            banded_matrix_vector_product!(dg_linear, A_banded, dQ2, Q)
            @test dQ1.realdata ≈ dQ2.realdata
          end
        end
      end
    end
  end
end

nothing
