export ErrorAdaptiveSolver, NoController
export IntegralController
export ProportionalIntegralController
export ProportionalIntegralDerivativeController

using LinearAlgebra: norm

struct ErrorAdaptiveParam{P, AT}
    p::P
    Q::AT
    error_estimate::AT
    dQ_error::AT
end
get_org_param(p) = p
get_org_param(eap::ErrorAdaptiveParam) = eap.p

mutable struct ErrorAdaptiveSolver{S, AT, EC, HT} <: AbstractODESolver
    solver::S
    candidate::AT
    error_estimate::AT
    dQ_error::AT
    error_controller::EC
    dt_history::HT
    nrejections::Int
end

function ErrorAdaptiveSolver(solver, error_controller, Q; save_dt_history=false)
    AT = typeof(Q)
    candidate = similar(Q)
    candidate .= Q
    error_estimate = similar(Q)
    dQ_error = similar(Q)
    dt_history = save_dt_history ? Vector{eltype(Q)}(undef, 0) : nothing
    ErrorAdaptiveSolver(solver, candidate, error_estimate, dQ_error, error_controller, dt_history, 0)
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
    dt_history = eas.dt_history
    
    easp = ErrorAdaptiveParam(p, Q, error_estimate, eas.dQ_error)

    acceptstep = false
    while !acceptstep
        candidate .= Q
        if adjustfinalstep && time + dt > timeend
          updatedt!(eas, timeend - time)
          dostep!(candidate, eas.solver, easp, time)
          break
        end
        dostep!(candidate, eas.solver, easp, time)
        order = embedded_order(eas)
        acceptstep, dt =
            eas.error_controller(candidate, error_estimate, dt, order)
        #@show dt
        if !acceptstep
          updatedt!(eas, dt)
          eas.nrejections += 1
        end
    end

    #if adjustfinalstep && time + getdt(eas.solver) > timeend
    #    updatedt!(eas, timeend - time)
    #    dostep!(candidate, eas.solver, easp, time)
    #end

    Q .= candidate

    #@show eas.solver.t, getdt(eas.solver) 
    eas.solver.t += getdt(eas.solver)
    #@show eas.solver.t

    if dt_history !== nothing
      push!(dt_history, getdt(eas.solver))
    end

    updatedt!(eas, dt)
    return eas.solver.t
end

abstract type AbstractErrorController end

# just for testing the machinery
struct NoController <: AbstractErrorController end
function (obj::NoController)(candidate, error_estimate, dt, p)
    return true, dt
end

Base.@kwdef mutable struct IntegralController{FT} <: AbstractErrorController
    safety_factor::FT = 9 // 10
    atol::FT = 0
    rtol::FT = 0
    clamp_min::FT = 0.2
    clamp_max::FT = 5
    δ::Union{FT, Nothing} = nothing
end

function (obj::IntegralController)(candidate, error_estimate, dt, p)
    # TODO: move to the constructor
    @assert obj.rtol >= 0
    @assert obj.atol >= 0
    @assert obj.rtol > 0 || obj.atol > 0

    # FIXME: do this without creating a temporary !
    error_scaled = @. error_estimate / (obj.atol + obj.rtol * abs(candidate))
    δ = norm(error_scaled, Inf, false)

    accepted = δ <= 1
    accepted && (obj.δ = δ)
    newdt = obj.safety_factor * dt * (1 / δ)^(1 / (p + 1))
    newdt = clamp(newdt, obj.clamp_min * dt, obj.clamp_max * dt)
    return accepted, newdt
end

Base.@kwdef mutable struct ProportionalIntegralController{FT} <: AbstractErrorController
    safety_factor::FT = 9 // 10
    atol::FT = 0
    rtol::FT = 0
    clamp_min::FT = 0.2
    clamp_max::FT = 5
    δ_n::Union{FT, Nothing} = nothing
    δ::Union{FT, Nothing} = nothing
end

function (obj::ProportionalIntegralController{FT})(candidate, error_estimate, dt, p) where {FT}
    @assert obj.rtol >= 0
    @assert obj.atol >= 0
    @assert obj.rtol > 0 || obj.atol > 0

    # on first use we don't have δ_n so just use the integral controller until we get it
    if isnothing(obj.δ_n)
      controller = IntegralController(safety_factor=obj.safety_factor, atol=obj.atol, rtol=obj.rtol,
                                      clamp_min=obj.clamp_min, clamp_max=obj.clamp_max)
      accepted, newdt = controller(candidate, error_estimate, dt, p)
      obj.δ_n = controller.δ
      return accepted, newdt
    else
      @assert !isnothing(obj.δ_n)

      # FIXME: do this without creating a temporary !
      error_scaled = @. error_estimate / (obj.atol + obj.rtol * abs(candidate))
      δ_np1 = norm(error_scaled, Inf, false)
      
      newdt = obj.safety_factor * dt * (1 / δ_np1)^(FT(0.7) / p) * obj.δ_n^(FT(0.4) / p)
      newdt = clamp(newdt, obj.clamp_min * dt, obj.clamp_max * dt)

      accepted = δ_np1 <= 1
      if accepted
        obj.δ = δ_np1
        obj.δ_n = δ_np1
      end
      return accepted, newdt
    end
end

Base.@kwdef mutable struct ProportionalIntegralDerivativeController{FT} <: AbstractErrorController
    safety_factor::FT = 9 // 10
    atol::FT = 0
    rtol::FT = 0
    clamp_min::FT = 0.2
    clamp_max::FT = 5
    δ_n::Union{FT, Nothing} = nothing
    δ_nm1::Union{FT, Nothing} = nothing
end

function (obj::ProportionalIntegralDerivativeController{FT})(candidate, error_estimate, dt, p) where {FT}
    if isnothing(obj.δ_nm1)
      controller = IntegralController(safety_factor=obj.safety_factor, atol=obj.atol, rtol=obj.rtol,
                                      clamp_min=obj.clamp_min, clamp_max=obj.clamp_max)
      accepted, newdt = controller(candidate, error_estimate, dt, p)
      obj.δ_nm1 = controller.δ
      return accepted, newdt
    elseif isnothing(obj.δ_n)
      @assert !isnothing(obj.δ_nm1)
      controller = ProportionalIntegralController(safety_factor=obj.safety_factor,
                                                  atol=obj.atol, rtol=obj.rtol, δ_n = obj.δ_nm1,
                                                  clamp_min=obj.clamp_min, clamp_max=obj.clamp_max)
      accepted, newdt = controller(candidate, error_estimate, dt, p)
      obj.δ_n = controller.δ
      return accepted, newdt
    else
      @assert !isnothing(obj.δ_nm1) && !isnothing(obj.δ_n)

      # FIXME: do this without creating a temporary !
      error_scaled = @. error_estimate / (obj.atol + obj.rtol * abs(candidate))
      δ_np1 = norm(error_scaled, Inf, false)

      α = FT(0.49) / p
      β = FT(0.34) / p
      γ = FT(0.10) / p
      newdt = obj.safety_factor * dt * (1 / δ_np1)^α * obj.δ_n^β * (1 / obj.δ_nm1)^γ
      newdt = clamp(newdt, obj.clamp_min * dt, obj.clamp_max * dt)
      accepted = δ_np1 <= 1
      if accepted
        obj.δ_nm1 = obj.δ_n
        obj.δ_n = δ_np1
      end
      return accepted, newdt
    end
  end
