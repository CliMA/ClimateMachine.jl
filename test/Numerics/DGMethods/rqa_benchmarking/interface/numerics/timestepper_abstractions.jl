abstract type AbstractRate end

# TODO: Add more methods here such as MultiRate, Explicit [can't reuse word]
Base.@kwdef struct IMEX{ℱ}
    method::ℱ
end

function IMEX()
    return IMEX(ARK2GiraldoKellyConstantinescu)
end

Base.@kwdef struct Explicit{ℳ, ℛ} <: AbstractRate
    model::ℳ
    rate::ℛ = 1
end

function Explicit(model::BalanceLaw; rate = 1) 
    return Explicit(model = model, rate = rate)
end

function Explicit(model::SpaceDiscretization; rate = 1) 
    return Explicit(model = model, rate = rate)
end

# perhaps "solver_method" instead of "method"?
Base.@kwdef struct Implicit{ℳ, ℛ, 𝒜} <: AbstractRate
    model::ℳ
    rate::ℛ = 1
    method::𝒜 = LinearBackwardEulerSolver(ManyColumnLU(); isadjustable = false)
end

function Implicit(model::BalanceLaw; rate = 1, adjustable = false, method = LinearBackwardEulerSolver(ManyColumnLU(), isadjustable = false)) 
    return Implicit(model = model, rate = rate, method = method)
end

function Implicit(model::SpaceDiscretization; rate = 1, adjustable = false, method = LinearBackwardEulerSolver(ManyColumnLU(), isadjustable = false)) 
    return Implicit(model = model, rate = rate, method = method)
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

# Default to Explicit
# TODO: Extend whatever struct an ODE solver is
function construct_odesolver(method, rhs, state, Δt; t0 = 0, split_explicit_implicit = false)
    # put error checking here 
    explicit_rhs = explicit(rhs)
    implicit_rhs = implicit(rhs)
    number_implicit = length(implicit_rhs)
    number_explicit = length(explicit_rhs) 
    if (number_implicit != 0) | (number_explicit != 1)
        error_string = "Explicit methods require one (and only one) explicit model"
        error_string *= "\n for example, a tuple of the form  (Explicit(model),) or just (model,)"
        @error(error_string)
    end

    explicit_model = explicit_rhs[1]

    # Instantiate time stepping method    
    odesolver = method(
        explicit_rhs.model,
        state;
        dt = Δt,
        t0 = t0,
        split_explicit_implicit = split_explicit_implicit
    )
    return odesolver
end

# IMEX
function construct_odesolver(timestepper::IMEX, rhs, state, Δt; t0 = 0, split_explicit_implicit = false)
    # put error checking here 
    implicit_rhs = implicit(rhs)
    explicit_rhs = explicit(rhs)
    number_implicit = length(implicit_rhs)
    number_explicit = length(explicit_rhs) 
    if (number_implicit != 1) | (number_explicit != 1)
        error_string = "IMEX methods require 1 implicit model and one explicit model"
        error_string *= "\n for example, a tuple of the form  (Explicit(model), Implicit(linear_model),)"
        @error(error_string)
    end
    explicit_rhs = explicit_rhs[1]
    implicit_rhs = implicit_rhs[1]

    # Instantiate time stepping method    
    odesolver = timestepper.method(
        explicit_rhs.model,
        implicit_rhs.model,
        implicit_rhs.method,
        state;
        dt = Δt,
        t0 = t0,
        split_explicit_implicit = split_explicit_implicit
    )
    return odesolver
end