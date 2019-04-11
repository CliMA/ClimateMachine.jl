module NumericalFluxes
using StaticArrays

# Rusanov (or local Lax-Friedrichs) Flux
function rusanov!(F::MArray{Tuple{nstate}}, nM,
                  QM, GM, ϕcM,
                  QP, GP, ϕcP,
                  t, flux!, wavespeed,
                  preflux = (_...) -> (),
                  correctQ! = nothing
                 ) where {nstate}
  PM = preflux(QM, GM, ϕcM, t)
  λM = wavespeed(nM, QM, GM, ϕcM, t, PM...)
  FM = similar(F, Size(3, nstate))
  flux!(FM, QM, GM, ϕcM, t, PM...)

  PP = preflux(QP, GP, ϕcP, t)
  λP = wavespeed(nM, QP, GP, ϕcP, t, PP...)
  FP = similar(F, Size(3, nstate))
  flux!(FP, QP, GP, ϕcP, t, PP...)

  λ  =  max(λM, λP)

  if correctQ! === nothing
    @inbounds for s = 1:nstate
      F[s] = (nM[1] * (FM[1, s] + FP[1, s]) + nM[2] * (FM[2, s] + FP[2, s]) +
              nM[3] * (FM[3, s] + FP[3, s]) + λ * (QM[s] - QP[s])) / 2
    end
  else
    QM_cpy = copy(QM)
    QP_cpy = copy(QP)
    correctQ!(QM_cpy, ϕcM)
    correctQ!(QP_cpy, ϕcP)
    @inbounds for s = 1:nstate
      F[s] = (nM[1] * (FM[1, s] + FP[1, s]) + nM[2] * (FM[2, s] + FP[2, s]) +
              nM[3] * (FM[3, s] + FP[3, s]) + λ * (QM_cpy[s] - QP_cpy[s])) / 2
    end
  end
end

end
