module NumericalFluxes
using StaticArrays

# Rosonuv (or local Lax-Friedrichs) Flux
function rosanuv!(F::MArray{Tuple{nstate}}, nM,
                  QM, GM, ϕcM, ϕdM,
                  QP, GP, ϕcP, ϕdP,
                  t, flux!, wavespeed, preflux = (_...) -> ()) where nstate
  PM = preflux(QM, GM, ϕcM, ϕdM, t)
  λM = wavespeed(nM, QM, GM, ϕcM, ϕdM, t, PM...)
  FM = similar(F, Size(3, nstate))
  flux!(FM, QM, GM, ϕcM, ϕdM, t, PM...)

  PP = preflux(QP, GP, ϕcP, ϕdP, t)
  λP = wavespeed(nM, QP, GP, ϕcP, ϕdP, t, PP...)
  FP = similar(F, Size(3, nstate))
  flux!(FP, QP, GP, ϕcP, ϕdP, t, PP...)

  λ  =  max(λM, λP)

  @inbounds for s = 1:nstate
    F[s] = (nM[1] * (FM[1, s] + FP[1, s]) + nM[2] * (FM[2, s] + FP[2, s]) +
            nM[3] * (FM[3, s] + FP[3, s]) + λ * (QM[s] - QP[s])) / 2
  end
end

end
