
function update!(Q, offset, ::Val{i}, yn, ΔYnj, fYnj, αi, βi, γi, d_i, dt) where i
  @inbounds @loop for e = (1:length(Q);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if i > 2
      ΔYnj[i-2][e] = Q[e] - yn[e] # is 0 for i == 2
    end
    Q[e] = yn[e] # (1a)
    offset[e] = (βi[1]/d_i) .* fYnj[1][e] # (1b)
    @unroll for j = 2:i-1
      Q[e] += αi[j] .* ΔYnj[j-1][e] # (1a cont.)
      offset[e] += (γi[j]/(d_i*dt)) * ΔYnj[j-1][e] + (βi[j]/d_i) * fYnj[j][e] # (1b cont.)
    end
  end
end
