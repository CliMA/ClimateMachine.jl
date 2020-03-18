
function update!(dQ, Q, Qstage, rka1, rka2, rkb, dt, slow_δ, slow_dQ,
                 slow_scaling)
  @inbounds @loop for i = (1:length(Q);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if slow_δ !== nothing
      dQ[i] += slow_δ * slow_dQ[i]
    end
    Qstage[i] = rka1 * Q[i] + rka2 * Qstage[i] + dt * rkb * dQ[i]
    if slow_scaling !== nothing
      slow_dQ[i] *= slow_scaling
    end
  end
end
