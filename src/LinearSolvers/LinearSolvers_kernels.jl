using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

function linearcombination!(Q, cs, Xs, increment::Bool)
  if !increment
    @inbounds @loop for i = (1:length(Q);
                             (blockIdx().x - 1) * blockDim().x + threadIdx().x)
      Q[i] = -zero(eltype(Q))
    end
  end
  @inbounds for j = 1:length(cs)
    @loop for i = (1:length(Q);
                             (blockIdx().x - 1) * blockDim().x + threadIdx().x)
      Q[i] += cs[j] * Xs[j][i]
    end
  end
end

