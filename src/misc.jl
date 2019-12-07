using MPI, CUDAapi, Requires

const cuarray_pkgid = Base.PkgId(Base.UUID("3a865a2d-5b23-5a0f-bc46-62713ec82fae"), "CuArrays")


"""
    CLIMA.array_type()

Returns the default array type to be used with CLIMA, either a `CuArray` if a GPU is
available, or an `Array` otherwise.

Use of GPUs can be explicitly disabled by setting the environment variable `CLIMA_GPU=false`.
"""
function array_type()
  if get(ENV, "CLIMA_GPU", "") != "false" && CUDAapi.has_cuda_gpu()
    CuArrays = Base.require(cuarray_pkgid)
    CuArrays.CuArray
  else
    Array
  end
end

"""
    CLIMA.init(ArrayType)

Initialize MPI and allocate GPUs among MPI ranks if using GPUs.
"""
function init(ArrayType=array_type())
  if !MPI.Initialized()
    MPI.Init()
  end
  _init_array(ArrayType)
  return nothing
end

function _init_array(::Type{Array})
  nothing
end

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  using .CuArrays, .CuArrays.CUDAdrv, .CuArrays.CUDAnative
  function _init_array(::Type{CuArray})
    comm = MPI.COMM_WORLD
    # allocate GPUs among MPI ranks
    local_comm = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED,  MPI.Comm_rank(comm))
    # we intentionally oversubscribe GPUs for testing: may want to disable this for production
    device!(MPI.Comm_rank(local_comm) % length(devices()))
    CuArrays.allowscalar(false)
  end
end

function gpu_allowscalar(val)
  if haskey(Base.loaded_modules, CLIMA.cuarray_pkgid)
    Base.loaded_modules[CLIMA.cuarray_pkgid].allowscalar(val)
  end
  return nothing
end