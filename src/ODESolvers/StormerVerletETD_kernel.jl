using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

function update!(offset, ::Val{iStage}, fYnj, βSiStage, τ, nPhi) where iStage
  @inbounds @loop for e = (1:length(offset);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
  Fac = βSiStage[1][nPhi];

  for k=(nPhi-1):-1:1
    Fac=Fac.*τ.+βSiStage[1][k];
  end
  offset[e]=Fac.*fYnj[1][e];

  for jStage=2:iStage
    Fac = βSiStage[jStage][nPhi];

    for k=(nPhi-1):-1:1
      Fac=Fac.*τ.+βSiStage[jStage][k];
    end
     offset[e]+=Fac.*fYnj[jStage][e];
  end

#=
    if i > 2
      ΔYnj[i-2][e] = Q[e] - yn[e] # is 0 for i == 2
    end
    Q[e] = yn[e] # (1a)
    offset[e] = (βi[1]/d_i) .* fYnj[1][e] # (1b)
    @unroll for j = 2:i-1
      Q[e] += αi[j] .* ΔYnj[j-1][e] # (1a cont.)
      offset[e] += (γi[j]/(d_i*dt)) * ΔYnj[j-1][e] + (βi[j]/d_i) * fYnj[j][e] # (1b cont.)
    end
=#
end

end
