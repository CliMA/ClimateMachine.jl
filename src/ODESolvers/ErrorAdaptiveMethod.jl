export ErrorAdaptiveSolver, NoController, IntegralController

using LinearAlgebra: norm

struct ErrorAdaptiveSolver{S, AT, EC} <: AbstractODESolver
    solver::S
    candidate::AT
    error_estimate::AT
    error_controller::EC
end

function ErrorAdaptiveSolver(solver, error_controller, Q)
    AT = typeof(Q)
    candidate = similar(Q)
    error_estimate = similar(Q)
    ErrorAdaptiveSolver(solver, candidate, error_estimate, error_controller)
end

gettime(eas::ErrorAdaptiveSolver) = gettime(eas.solver)
getdt(eas::ErrorAdaptiveSolver) = getdt(eas.solver)
updatedt!(eas::ErrorAdaptiveSolver, dt) = updatedt!(eas.solver, dt)
embedded_order(eas::ErrorAdaptiveSolver) = embedded_order(eas.solver)

function general_dostep!(
    Q,
    eas::ErrorAdaptiveSolver,
    p,
    timeend::Real;
    adjustfinalstep::Bool,
)
    candidate = eas.candidate
    error_estimate = eas.error_estimate
    time = gettime(eas)
    dt = getdt(eas)

    acceptstep = false
    while !acceptstep
        dostep!((candidate, Q, error_estimate), eas.solver, p, time)
        order = embedded_order(eas)
        acceptstep, dt =
            eas.error_controller(candidate, error_estimate, dt, order)
        acceptstep || updatedt!(eas, dt)
    end

    if adjustfinalstep && time + getdt(eas.solver) > timeend
        updatedt!(eas, timeend - time)
        dostep!((candidate, Q, error_estimate), eas.solver, p, time)
    end

    Q .= candidate

    eas.solver.t += getdt(eas.solver)
    updatedt!(eas, dt)
    return eas.solver.t
end

abstract type AbstractErrorController end

# just for testing the machinery
struct NoController <: AbstractErrorController end
function (obj::NoController)(candidate, error_estimate, dt, p)
    return true, dt
end

Base.@kwdef struct IntegralController{FT} <: AbstractErrorController
    safety_factor::FT = 9 // 10
    atol::FT = 0
    rtol::FT = 0
end

function (obj::IntegralController)(candidate, error_estimate, dt, p)
    # TODO: move to the constructor
    @assert obj.rtol >= 0
    @assert obj.atol >= 0
    @assert obj.rtol > 0 || obj.atol > 0

    # FIXME: do this without creating a temporary !
    error_scaled = @. error_estimate / (obj.atol + obj.rtol * abs(candidate))
    δ = norm(error_scaled, Inf, false)

    newdt = obj.safety_factor * dt * (1 / δ)^(1 / (p + 1))
    return δ <= 1, newdt
end
