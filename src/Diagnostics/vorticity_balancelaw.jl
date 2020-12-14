# A mini balance law for computing vorticity using the DG kernels.
#

using StaticArrays

using ..BalanceLaws
using ..VariableTemplates
using ..MPIStateArrays

import ..BalanceLaws:
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    eq_tends,
    flux,
    source,
    wavespeed,
    boundary_conditions,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    transform_post_gradient_laplacian!,
    init_state_auxiliary!,
    init_state_prognostic!,
    update_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!


# A mini balance law that is used to take the gradient of u and v
# to obtain vorticity.
struct VorticityModel <: BalanceLaw end

# output vorticity vector
vars_state(::VorticityModel, ::Prognostic, FT) = @vars(Ω_bl::SVector{3, FT})
# input velocity vector
vars_state(::VorticityModel, ::Auxiliary, FT) = @vars(u::SVector{3, FT})
vars_state(::VorticityModel, ::Gradient, FT) = @vars()
vars_state(::VorticityModel, ::GradientFlux, FT) = @vars()

# required for each balance law 
function init_state_auxiliary!(
    m::VorticityModel,
    state_auxiliary::MPIStateArray,
    grid,
    direction,
) end
function init_state_prognostic!(
    ::VorticityModel,
    state::Vars,
    aux::Vars,
    localgeo,
    t,
) end
function nodal_update_auxiliary_state!(
    m::VorticityModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end;
function flux_first_order!(
    ::VorticityModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    u = aux.u
    @inbounds begin
        flux.Ω_bl = @SMatrix [
            0 u[3] -u[2]
            -u[3] 0 u[1]
            u[2] -u[1] 0
        ]
    end
end
function compute_gradient_argument!(
    ::VorticityModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) end
flux_second_order!(::VorticityModel, _...) = nothing
source!(::VorticityModel, _...) = nothing
boundary_conditions(::VorticityModel) = ntuple(i -> nothing, 6)
boundary_state!(nf, ::Nothing, ::VorticityModel, _...) = nothing
