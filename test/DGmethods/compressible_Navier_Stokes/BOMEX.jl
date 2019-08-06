using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.Vtk


@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

using CLIMA.Atmos

DFloat = Float32
const μ_exact = 10

function edmf_init_state!(state::Vars, aux::Vars, (x,y,z), t)
end

function edmf_source!(source::Vars, state::Vars, aux::Vars, t::Real)
end

∑a_up = 0.1
n_up = 2
a_initial = ntuple(i -> i==1 ? DFloat(1-∑a_up) : DFloat(∑a_up/n_up), n_up+1)

model = AtmosModel(ConstantViscosityWithDivergence(DFloat(μ_exact)),
                   EDMF{n_up}(ConstantAreaFrac(a_initial),
                              MomentumModel(DFloat),
                              EnergyModel(DFloat),
                              EDMFDryModel(),
                              TKEModel(DFloat),
                              SurfaceModel(DFloat),
                              ConstantMixingLength(DFloat),
                              ConstantEntrDetr(DFloat),
                              PressureModel(DFloat),
                              BuoyancyModel(DFloat)
                              ),
                   DryModel(),
                   NoRadiation(),
                   edmf_source!,
                   InitStateBC(),
                   edmf_init_state!)

