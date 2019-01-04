# {{{ reshape for CuArray
function Base.reshape(A::CuArray, dims::NTuple{N, Int}) where {N}
  @assert prod(dims) == prod(size(A))
  CuArray{eltype(A), length(dims)}(dims, A.buf)
end
# }}}

# {{{ fill! for CuArray
function Base.fill!(A::CuArray, val)
  threads = 1024
  blocks = div(length(A) + threads-1, threads)
  T = eltype(A)
  @cuda threads=threads blocks=blocks knl_fill!(A, T(val), length(A))
end
function knl_fill!(a, val, nvals)
  n = (blockIdx().x-1) * blockDim().x + threadIdx().x
  n <= nvals && @inbounds a[n] = val
  nothing
end
