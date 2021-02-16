using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers

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
    (LS3NRK44Classic, 4),
    (LS3NRK33Heuns, 3),
    (SSPRK22Heuns, 2),
    (SSPRK22Ralstons, 2),
    (SSPRK33ShuOsher, 3),
    (SSPRK34SpiteriRuuth, 3),
    (LSRKEulerMethod, 1),
)

const imex_methods_lowstorage_compatible = (
    # Low-storage variant methods have an assumption that the
    # explicit and implicit rhs/time-scaling coefficients (B/C vectors)
    # in the Butcher tables are the same.
    (ARK1ForwardBackwardEuler, 1),
    (ARK2ImplicitExplicitMidpoint, 2),
    (ARK2GiraldoKellyConstantinescu, 2),
    (ARK437L2SA1KennedyCarpenter, 4),
    (ARK548L2SA2KennedyCarpenter, 5),
    (DBM453VoglEtAl, 3),
)
const imex_methods_naivestorage_compatible = (
    imex_methods_lowstorage_compatible...,
    # Some methods can only be used with the `NaiveVariant` storage
    # scheme since, in general, ARK methods can have different time-scaling/rhs-scaling
    # coefficients (C/B vectors in the Butcher tables). For future reference,
    # any other ARK-type methods that have more general Butcher tables
    # (but with same number of stages) should be tested here:
    (Trap2LockWoodWeller, 2),
)

const mis_methods =
    ((MIS2, 2), (MIS3C, 2), (MIS4, 3), (MIS4a, 3), (TVDMISA, 2), (TVDMISB, 2))


const mrigark_erk_methods = ((MRIGARKERK33aSandu, 3), (MRIGARKERK45aSandu, 4))

const mrigark_irk_methods = (
    (MRIGARKESDIRK24LSA, 2),
    (MRIGARKESDIRK23LSA, 2),
    (MRIGARKIRK21aSandu, 2),
    (MRIGARKESDIRK34aSandu, 3),
    (MRIGARKESDIRK46aSandu, 4),
)

const fast_mrigark_methods =
    ((LSRK54CarpenterKennedy, 4), (LSRK144NiegemannDiehlBusch, 4))

struct DivideLinearSolver <: AbstractSystemSolver end
function SystemSolvers.prefactorize(
    linearoperator!,
    ::DivideLinearSolver,
    args...,
)
    linearoperator!
end
function SystemSolvers.linearsolve!(
    linearoperator!,
    preconditioner,
    ::DivideLinearSolver,
    Qtt,
    Qhat,
    args...,
)
    @. Qhat = 1 / Qhat
    linearoperator!(Qtt, Qhat, args...)
    @. Qtt = 1 / Qtt
end
