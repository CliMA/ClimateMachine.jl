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

function implicit(models::Tuple)
    explicit_models = []
    for model in models
        if model isa Implicit 
            push!(explicit_models, model)
        end
    end
    return explicit_models
end

# methods for constructing odesolvers
# TODO: Figure out current supertype for dispatch (or make it)

# IMEX
function construct_odesolver(method::IMEX, models, state, Δt; t0 = 0, split_explicit_implicit = false)
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

# Explicit
function construct_odesolver(method::Explicit, models, state, Δt; t0 = 0, split_explicit_implicit = false)
    # put error checking here 
    explicit_models = explicit(models)
    @assert length(explicit_models) = 1

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