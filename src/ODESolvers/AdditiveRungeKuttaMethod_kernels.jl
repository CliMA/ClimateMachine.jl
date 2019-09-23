using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

function stage_update!(Q, Qstages, Rstages, Qhat, Qtt, RKA_explicit, RKA_implicit, dt,
                       ::Val{is}, ::Val{split_nonlinear_linear}) where {is, split_nonlinear_linear}
  @inbounds @loop for i = (1:length(Q);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    Qhat_i = Q[i]
    Qstages_is_i = -zero(eltype(Q))
    @unroll for js = 1:is-1
      if split_nonlinear_linear
        rkcoeff = RKA_implicit[is, js] / RKA_implicit[is, is]
      else
        rkcoeff = (RKA_implicit[is, js] - RKA_explicit[is, js]) / RKA_implicit[is, is]
      end
      commonterm = rkcoeff * Qstages[js][i]
      Qhat_i += commonterm + dt * RKA_explicit[is, js] * Rstages[js][i]
      Qstages_is_i -= commonterm
    end
    Qstages[is][i] = Qstages_is_i
    Qhat[i] = Qhat_i
    Qtt[i] = Qhat_i
  end
end

function solution_update!(Q, Rstages, RKB, dt, ::Val{nstages}) where nstages
  @inbounds @loop for i = (1:length(Q);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    @unroll for is = 1:nstages
      Q[i] += RKB[is] * dt * Rstages[is][i]
    end
  end
end
