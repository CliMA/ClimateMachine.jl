using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

function update!(rhs, Q, Qstage, rka1, rka2, rkb, dt)
  @inbounds @loop for i = (1:length(Q);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    Qstage[i] = rka1 * Q[i] + rka2 * Qstage[i] + dt * rkb * rhs[i]
  end
end
