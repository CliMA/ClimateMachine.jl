module NumericalFluxes
using StaticArrays

# Rusanov (or local Lax-Friedrichs) Flux
function rusanov!(F::MArray{Tuple{nstate}}, nM,
                  QM, auxM,
                  QP, auxP,
                  t, flux!, wavespeed,
                  preflux = (_...) -> (),
                  correctQ! = nothing
                 ) where {nstate}
  PM = preflux(QM, auxM, t)
  λM = wavespeed(nM, QM, auxM, t, PM...)
  FM = similar(F, Size(3, nstate))
  flux!(FM, QM, auxM, t, PM...)

  PP = preflux(QP, auxP, t)
  λP = wavespeed(nM, QP, auxP, t, PP...)
  FP = similar(F, Size(3, nstate))
  flux!(FP, QP, auxP, t, PP...)

  λ  =  max(λM, λP)

  if correctQ! === nothing
    @inbounds for s = 1:nstate
      F[s] = (nM[1] * (FM[1, s] + FP[1, s]) + nM[2] * (FM[2, s] + FP[2, s]) +
              nM[3] * (FM[3, s] + FP[3, s]) + λ * (QM[s] - QP[s])) / 2
    end
  else
    QM_cpy = copy(QM)
    QP_cpy = copy(QP)
    correctQ!(QM_cpy, auxM)
    correctQ!(QP_cpy, auxP)
    @inbounds for s = 1:nstate
      F[s] = (nM[1] * (FM[1, s] + FP[1, s]) + nM[2] * (FM[2, s] + FP[2, s]) +
              nM[3] * (FM[3, s] + FP[3, s]) + λ * (QM_cpy[s] - QP_cpy[s])) / 2
    end
  end
end

end
