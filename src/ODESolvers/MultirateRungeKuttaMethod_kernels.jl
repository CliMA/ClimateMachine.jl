using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

function update!(fast_dQ, slow_dQ, Q, δ, fast_rka, fast_rkb, dt,
                 slow_rka = nothing)
  @inbounds @loop for i = (1:length(Q);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    fast_dQ[i] += δ * slow_dQ[i]
    Q[i] += fast_rkb * dt * fast_dQ[i]
    fast_dQ[i] *= fast_rka
    if slow_rka !== nothing
      slow_dQ[i] *= slow_rka
    end
  end
end

