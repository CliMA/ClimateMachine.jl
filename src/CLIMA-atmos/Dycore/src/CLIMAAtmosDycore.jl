module CLIMAAtmosDycore

using Printf: @sprintf
using Canary, MPI, Requires
using Logging

using PlanetParameters: R_d, cp_d, cv_d, grav
using ParametersType

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
  using .CUDAnative.CUDAdrv

  include("setup_cuda.jl")
  include("lsrk_cuda.jl")
  include("vanilla_euler_cuda.jl")
end

include("setup.jl")

include("callbacks.jl")
include("lsrk.jl")
include("vanilla_euler.jl")
include("vtk.jl")

end # module
