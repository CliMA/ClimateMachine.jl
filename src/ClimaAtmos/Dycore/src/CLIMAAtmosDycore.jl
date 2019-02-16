module CLIMAAtmosDycore

using Requires

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
  using .CUDAnative.CUDAdrv

  include("CLIMAAtmosDycore_cuda.jl")

  # }}}
end

include("types.jl")

include("VanillaEuler.jl")

include("LSRK.jl")

include("GenericCallbacks.jl")

end # module
