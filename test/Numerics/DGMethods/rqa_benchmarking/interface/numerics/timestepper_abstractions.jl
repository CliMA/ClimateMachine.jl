abstract type AbstrateRate end

Base.@kwdef struct Explicit{ℳ, ℛ} <: AbstractRate
    model::ℳ
    rate::ℛ = 1
end

function Explicit(model::BalanceLaw; rate = 1) 
    return Explicit(model = model, rate = rate)
end

Base.@kwdef struct Implicit{ℳ, ℛ, 𝒜} <: AbstractRate
    model::ℳ
    rate::ℛ = 1
    method::𝒜 = LinearBackwardEulerSolver(ManyColumnLU(); isadjustable = false)
end

function Implicit(model::BalanceLaw; rate = 1, adjustable = false, method = LinearBackwardEulerSolver(ManyColumnLU(); isadjustable = false)) 
    return Implicit(model = model, rate = rate, adjustable = adjustable)
end

# helper functions
function explicit(models::Tuple)
    explicit_models = []
    for model in models
        if model isa Explicit 
            push!(explicit_models, model)
        end
    end
    return explicit_models
end
# extension
explicit(model) = explicit((model,))
explicit(model::BalanceLaw) = explicit(Explicit(model))

function implicit(models::Tuple)
    explicit_models = []
    for model in models
        if model isa Implicit 
            push!(explicit_models, model)
        end
    end
    return explicit_models
end
# extension
implicit(model) = implicit(())

# methods for constructing odesolvers, TODO: Extend whatever struct an ODE solver is

# Default to Explicit
function construct_odesolver(method, models, state, Δt; t0 = 0, split_explicit_implicit = false)
    # put error checking here 
    explicit_models = explicit(models)
    implicit_models = implicit(models)
    @assert length(explicit_models) = 1
    @assert length(implicit_models) = 0

    explicit_model = explicit_models[1]

    # Instantiate time stepping method    
    odesolver = method(
        explicit_model.model,
        state;
        dt = Δt,
        t0 = t0,
        split_explicit_implicit = split_explicit_implicit
    )
    return odesolver
end


# IMEX
function construct_odesolver(method::AbstractAdditiveRungeKutta, models, state, Δt; t0 = 0, split_explicit_implicit = false)
    # put error checking here 
    implicit_models = implicit(models)
    explicit_models = explicit(models)
    @assert length(implicit_models) = 1
    @assert length(explicit_models) = 1

    explicit_model = explicit_models[1]
    implicit_model = implicit_models[1]

    # Instantiate time stepping method    
    odesolver = method(
        explicit_model.model,
        implicit_model.model,
        implicit_model.method,
        state;
        dt = Δt,
        t0 = t0,
        split_explicit_implicit = split_explicit_implicit
    )
    return odesolver
end

