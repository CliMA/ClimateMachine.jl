using Impero
include(pwd() * "/tutorials/Land/Heat/test_utils.jl")
# include(pwd() * "/tutorials/Land/Heat/boiler_plate.jl")

@wrapper α=0.1 ρ=10 T=1 ∇T=1
@pde_system heat_equation = [
    ∇T = α*∂x(T),
    ∂t(T) = ∂x(∇T),
]

#=
heat_equation[1] is what we need for
compute_gradient_argument!
compute_gradient_flux!
abstract type HeatModel <: BalanceLaw end

function to_aux_state(eq::Equation, model_name::DataType )
    transform_equ = heat_equation[1].rhs.operand # is T, TODO:(should be get divergence)
    state = transform_equ # TODO: get_state! function returns a list of state variables
    expr_state = :(state) #convert state to expr
    # stuff inside \nabla
    @eval function compute_gradient_argument!(
        m::model_name,
        transform::Vars,
        state::Vars,
        aux::Vars,
        t::Real,
    )
        # loop through states
        transform.$(expr_state) = state.$(expr_state)
    end;

    # stuff outside ∇ of the diffusive flux
    (stuff) * \nabla T
    stuff = grab_stuff(eq) # TODO:
    something = equ.lhs # automate diffusive name
    @eval function compute_gradient_flux!(
        m::model_name,
        diffusive::Vars,
        ∇G::Grad,
        state::Vars,
        aux::Vars,
        t::Real,
    )
        diffusive.$(something) = - $(stuff) * ∇G.$(state)
    end;
    return nothing
end

# ∂t(T) = ∂x(0 + ∇T) + 0 

function to_prog_state(eq::Equation, model_name::DataType )
    prog_equ = heat_equation[2].rhs.operand
    state = transform_equ # TODO: get_state! function returns a list of state variables
    something = equ.lhs # automate diffusive name
    @eval function flux_second_order!(
        m::model_name,
        flux::Grad,
        state::Vars,
        diffusive::Vars,
        hyperdiffusive::Vars,
        aux::Vars,
        t::Real,
    )
        flux.$(state) += diffusive.$(something)
    end;

    function source!(m::model_name, _...) end;
    # pull out the first order term inside ∂x
    function flux_first_order!(m::model_name, _...) end;
    # return nothing now

    function nodal_update_auxiliary_state!(
        m::model_name,
        state::Vars,
        aux::Vars,
        t::Real,
    )
        return nothing
end;
end

=#

expr = heat_equation
macro my_wonky_macro(expr)
    prog_symbols = prog_function() # T
    grad_symbols = grad_function() # prognostic_state.T
    flux_symbols = flux_function() # α*∇T
end
