module NumericalFluxes
using StaticArrays

"""
    rusanov!(F::MArray, nM, QM, auxM, QP, auxP, t, flux!, wavespeed,
             [preflux = (_...) -> (), correctQ!])

Calculate the Rusanov (aka local Lax-Friedrichs) numerical flux given the plus
and minus side states `QP` and `QM` using the physical flux function `flux!` and
`wavespeed` calculation.

The `flux!` has almost the same calling convention as `inviscid_flux!` from
[`DGBalanceLaw`](@ref) except that `preflux(Q, aux, t)` is splatted at the end
of the call.

The function `wavespeed` should return the maximum wavespeed for a state and is
called as `wavespeed(nM, QM, auxM, t, preflux(QM, auxM, t)...)` and
`wavespeed(nM, QP, auxP, t, preflux(QP, auxP, t)...)` where `nM` is the outward
unit normal for the minus side.

When present `correctQ!(QM, auxM)` and `correctQ!(QP, auxP)` will be after
`wavespeed` and `flux!` are called to the user can modify `QM` and `QP` before
`QM - QP` is needed; this is useful for correcting `Q` to include discontinuous
reference states.

!!! todo

    We may want to switch to a `computed_jump!` instead of `correctQ!` since
    this would allow the user to better handle round-off error with large
    background states.

"""
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
