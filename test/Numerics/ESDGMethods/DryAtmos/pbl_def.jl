include("DryAtmos.jl")

using ClimateMachine.Thermodynamics
using ClimateMachine.VariableTemplates

import CLIMAParameters
CLIMAParameters.Planet.grav(::EarthParameterSet) = 10

struct HeatFlux end
function source!(
    m::DryAtmosModel,
    ::HeatFlux,
    source,
    state,
    aux,
)
    FT = eltype(aux)

    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)
    _MSLP::FT = MSLP(param_set)

    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ = aux.Φ

    z = Φ / _grav

    p = pressure(ρ, ρu, ρe, Φ)

    h0 = FT(1 / 100)
    hscale = FT(25)
    hflux = h0 * exp(-z / hscale) / hscale
    exner = (p / _MSLP) ^ (_R_d / _cp_d)
    source.ρe += _cp_d * ρ * hflux * exner
    
    return nothing
end

struct Absorber{FT}
  tau::FT
  zabs::FT
  ztop::FT
end
function source!(
    m::DryAtmosModel,
    absorber::Absorber,
    source,
    state,
    aux,
)
    FT = eltype(aux)
    _grav::FT = grav(param_set)
    
    z = aux.Φ / _grav

    tau = absorber.tau
    zabs = absorber.zabs
    ztop = absorber.ztop

    zeps = FT(1e-4)
    if z >= (zabs + zeps)
      α = (z - zabs) / (ztop - zabs) / tau
    else
      α = FT(0)
    end
    source.ρu += -α * state.ρu
    source.ρe += -α * (state.ρe - aux.ref_state.ρe)
    return nothing
end

struct Drag end
function drag_source!(m::DryAtmosModel, ::Drag,
                      source::Vars, state::Vars, state_bot::Vars, aux::Vars)
    FT = eltype(source)
    _grav::FT = grav(param_set)
    Φ = aux.Φ
    z = Φ / _grav

    c0 = FT(1 / 10)
    hscale = FT(25)
    u0 = state_bot.ρu / state_bot.ρ
    v0 = @inbounds sqrt(u0[1] ^ 2 + u0[2] ^ 2)
    ρu = state.ρu
    ρu_drag = @inbounds SVector(ρu[1], ρu[2], 0)
    u = ρu / state.ρ

    S_ρu = -c0 * v0 * ρu_drag * exp(-z / hscale) / hscale
    source.ρu +=  S_ρu
    source.ρe += u' * S_ρu
end


Base.@kwdef struct PBL{FT} <: AbstractDryAtmosProblem
    domain_length::FT = 3200
    domain_height::FT = 1500
    θ0::FT = 300
    zm::FT = 500
    st::FT = 1e-4 / grav(param_set)
end

function init_state_prognostic!(bl::DryAtmosModel, 
                                setup::PBL,
                                state, aux, localgeo, t)

    FT = eltype(state)
    (x, y, z) = localgeo.coord
    
    _MSLP::FT = MSLP(param_set)
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _cp_d::FT = cp_d(param_set)

    θ0 = setup.θ0
    zm = setup.zm
    st = setup.st
    
    ρ = aux.ref_state.ρ
    p = aux.ref_state.p

    rnd = rand() - FT(0.5)
    zeps = FT(1e-4)
    if z <= zm - zeps
      fac = rnd * (1 - z / zm)
    else
      fac = FT(0)
    end
    δθ = FT(0.001) * fac
    δT = δθ * (p / _MSLP) ^ (_R_d / _cp_d)
    δw = FT(0.2) * fac
    δe = δw ^ 2 / 2 + _cv_d * δT

    state.ρ = ρ
    state.ρu = ρ * SVector(0, 0, δw)
    state.ρe = aux.ref_state.ρe + ρ * δe
end

struct PBLProfile{S}
  setup::S
end
function (prof::PBLProfile)(param_set, z)
     FT = typeof(z)
     setup = prof.setup
    _MSLP::FT = MSLP(param_set)
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)

    θ0 = setup.θ0
    zm = setup.zm
    st = setup.st
  
    α = _grav / (_cp_d * θ0)
    β = _cp_d / _R_d
    θ = θ0 * (z <= zm ? 1 : 1 + (z - zm) * st)
    zeps = FT(1e-4)
    if z <= zm - zeps
      p = _MSLP * (1 - α * z) ^ β
    else
      p = _MSLP * (1 - α * (zm + log(1 + st * (z - zm)) / st)) ^ β
    end
    T = θ * (p / _MSLP) ^ (_R_d / _cp_d)
    
    T, p
end
