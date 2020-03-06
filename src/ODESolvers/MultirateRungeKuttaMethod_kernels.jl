
function update!(fast_dQ, slow_dQ, δ, slow_rka = nothing)
  @inbounds @loop for i = (1:length(fast_dQ);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    fast_dQ[i] += δ * slow_dQ[i]
    if slow_rka !== nothing
      slow_dQ[i] *= slow_rka
    end
  end
end

