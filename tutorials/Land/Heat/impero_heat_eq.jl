using Impero
include("test_utils.jl")
include("boiler_plate.jl")

@wrapper α=0.1 ρ=10 T=1 ∇T=1
@pde_system heat_equation = [
    ∇T = ∂x(T),
    ∂t(T) = ∂x(α*∇T),
]

#=

typeof_balance_law = HeatModel <: BalanceLaw

# ∇T = α * ∂x(sin(T))) pull out the thing in the derivative
function compute_gradient_argument!(
    m::typeof_balance_law,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.T = sin(state.T)
end;

# ∇T = α * ∂x(sin(T))) pull out the thing outside the derivative
function compute_gradient_flux!(
    m::typeof_balance_law,
    diffusive::Vars,
    ∇G::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.α∇T = - m.α * ∇G.T
end;

# ∂t(T) = ∂x(0 + ∇T) + 0 
# pull out the term with a derivative inside ∂x
function flux_second_order!(
    m::typeof_balance_law,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux.T += diffusive.α∇T
end;

# ∂t(T) = ∂x(0 + ∇T) + 0 
# pull out terms outside ∂x
function source!(m::typeof_balance_law, _...) end;
# pull out the first order term inside ∂x
function flux_first_order!(m::typeof_balance_law, _...) end;
# return nothing now
function nodal_update_auxiliary_state!(
    m::typeof_balance_law,
    state::Vars,
    aux::Vars,
    t::Real,
)
    return nothing
end;
=#

expr = heat_equation
macro my_wonky_macro(expr)
    prog_symbols = prog_function() # T
    grad_symbols = grad_function() # prognostic_state.T
    flux_symbols = flux_function() # α*∇T
end
