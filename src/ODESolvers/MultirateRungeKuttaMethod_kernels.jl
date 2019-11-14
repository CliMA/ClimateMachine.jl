using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

function update!(fast_dQ, slow_dQ, δ, slow_rka = nothing)
  @inbounds @loop for i = (1:length(fast_dQ);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    fast_dQ[i] += δ * slow_dQ[i]
    if slow_rka !== nothing
      slow_dQ[i] *= slow_rka
    end
  end
end

