using ..Ocean: coriolis_parameter
using ..HydrostaticBoussinesq
using ...DGMethods

import ...BalanceLaws
import ..HydrostaticBoussinesq:
    viscosity_tensor,
    coriolis_force!,
    velocity_gradient_argument!,
    velocity_gradient_flux!,
    hydrostatic_pressure!,
    compute_flow_deviation!

@inline function velocity_gradient_argument!(m::HBModel, ::Coupled, G, Q, A, t)
    G.∇u = Q.u
    G.∇uᵈ = A.uᵈ

    return nothing
end

@inline function velocity_gradient_flux!(m::HBModel, ::Coupled, D, G, Q, A, t)
    ν = viscosity_tensor(m)
    ∇u = @SMatrix [
        G.∇uᵈ[1, 1] G.∇uᵈ[1, 2]
        G.∇uᵈ[2, 1] G.∇uᵈ[2, 2]
        G.∇u[3, 1] G.∇u[3, 2]
    ]
    D.ν∇u = -ν * ∇u

    return nothing
end

@inline hydrostatic_pressure!(::HBModel, ::Coupled, _...) = nothing

@inline function coriolis_force!(m::HBModel, ::Coupled, S, Q, A, t)
    # f × u
    f = coriolis_parameter(m, A.y)
    uᵈ, vᵈ = A.uᵈ # Horizontal components of velocity
    S.u -= @SVector [-f * vᵈ, f * uᵈ]

    return nothing
end

# Compute Horizontal Flow deviation from vertical mean
@inline function compute_flow_deviation!(dg, m::HBModel, ::Coupled, Q, t)
    FT = eltype(Q)
    Nq, Nqk, _, _, nelemv, nelemh, nrealelemh, _ = basic_grid_info(dg)

    #### integrate the tendency
    model_int = dg.modeldata.integral_model
    integral = model_int.balance_law
    update_auxiliary_state!(model_int, integral, Q, 0)

    ### properly shape MPIStateArrays
    num_int = number_states(integral, Auxiliary())
    data_int = model_int.state_auxiliary.data
    data_int = reshape(data_int, Nq^2, Nqk, num_int, nelemv, nelemh)

    num_aux = number_states(m, Auxiliary())
    data_aux = dg.state_auxiliary.data
    data_aux = reshape(data_aux, Nq^2, Nqk, num_aux, nelemv, nelemh)

    num_state = number_states(m, Prognostic())
    data_state = reshape(Q.data, Nq^2, Nqk, num_state, nelemv, nelemh)

    ### get vars indices
    index_∫u = varsindex(vars_state(integral, Auxiliary(), FT), :(∫x))
    index_uᵈ = varsindex(vars_state(m, Auxiliary(), FT), :uᵈ)
    index_u = varsindex(vars_state(m, Prognostic(), FT), :u)

    ### get top value (=integral over full depth)
    ∫u = @view data_int[:, end:end, index_∫u, end:end, 1:nrealelemh]
    uᵈ = @view data_aux[:, :, index_uᵈ, :, 1:nrealelemh]
    u = @view data_state[:, :, index_u, :, 1:nrealelemh]

    ## make a copy of horizontal velocity
    ## and remove vertical mean velocity
    uᵈ .= u
    uᵈ .-= ∫u / m.problem.H

    return nothing
end
