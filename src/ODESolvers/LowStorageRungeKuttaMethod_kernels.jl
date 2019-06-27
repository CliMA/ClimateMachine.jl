using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

function update!(rhs, Q, rka, rkb, dt)
  @inbounds @loop for i = (1:length(Q);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    Q[i] += rkb * dt * rhs[i]
    rhs[i] *= rka
  end
end
