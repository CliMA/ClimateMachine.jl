using CLIMA.ODESolvers
using CLIMA.LinearSolvers

const slow_mrrk_methods =
    ((LSRK54CarpenterKennedy, 4), (LSRK144NiegemannDiehlBusch, 4))
const fast_mrrk_methods = (
    (LSRK54CarpenterKennedy, 4),
    (LSRK144NiegemannDiehlBusch, 4),
    (SSPRK33ShuOsher, 3),
    (SSPRK34SpiteriRuuth, 3),
)
const explicit_methods = (
    (LSRK54CarpenterKennedy, 4),
    (LSRK144NiegemannDiehlBusch, 4),
    (SSPRK33ShuOsher, 3),
    (SSPRK34SpiteriRuuth, 3),
    (LSRKEulerMethod, 1),
)

const imex_methods = (
    (ARK2GiraldoKellyConstantinescu, 2),
    (ARK437L2SA1KennedyCarpenter, 4),
    (ARK548L2SA2KennedyCarpenter, 5),
)

const mis_methods =
    ((MIS2, 2), (MIS3C, 2), (MIS4, 3), (MIS4a, 3), (TVDMISA, 2), (TVDMISB, 2))

struct DivideLinearSolver <: AbstractLinearSolver end
function LinearSolvers.prefactorize(
    linearoperator!,
    ::DivideLinearSolver,
    args...,
)
    linearoperator!
end
function LinearSolvers.linearsolve!(
    linearoperator!,
    ::DivideLinearSolver,
    Qtt,
    Qhat,
    args...,
)
    @. Qhat = 1 / Qhat
    linearoperator!(Qtt, Qhat, args...)
    @. Qtt = 1 / Qtt
end
