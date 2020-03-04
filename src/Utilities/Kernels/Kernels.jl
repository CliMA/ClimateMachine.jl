module Kernels
export sync_device

using KernelAbstractions

sync_device(::CPU) = nothing
using Requires
@init @require CUDAdrv = "c5f51814-7f29-56b8-a69c-e4d8f6be1fde" begin
  import .CUDAdrv
  sync_device(::CUDA) = CUDAdrv.synchronize()
end

end
