struct VorticityModel <: BalanceLaw end
vars_state(::VorticityModel, ::Auxiliary, FT) = @vars(ρ::FT, ρu::SVector{3, FT})
vars_state(::VorticityModel, ::Prognostic, FT) = @vars(ω::SVector{3, FT})
vars_state(::VorticityModel, ::Gradient, FT) = @vars()
vars_state(::VorticityModel, ::GradientFlux, FT) = @vars()
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
function flux_first_order!(
    ::VorticityModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρ = aux.ρ
    ρinv = 1/ρ
    ρu = aux.ρu
    u = ρinv * ρu
    @inbounds begin
      flux.ω = @SMatrix [ 0     u[3] -u[2];
                         -u[3]  0     u[1];
                          u[2] -u[1]  0    ]
    end
end
flux_second_order!(::VorticityModel, _...) = nothing
source!(::VorticityModel, _...) = nothing

boundary_conditions(::VorticityModel) = ntuple(i -> nothing, 6)
boundary_state!(nf, ::Nothing, ::VorticityModel, _...) = nothing
