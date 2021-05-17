include("../DryAtmos.jl")

Base.@kwdef struct RisingBubble{FT} <: AbstractDryAtmosProblem
  xmax::FT = 2000
  zmax::FT = 2000
  xc::FT = 0
  rc::FT = 250
  zc::FT = rc + 10
  θref::FT = 300
  δθc::FT = 1 // 2
end

function init_state_prognostic!(bl::DryAtmosModel, 
                                problem::RisingBubble,
                                state, aux, localgeo, t)
    (x, z, _) = localgeo.coord
    FT = eltype(state)

    @unpack xc, rc, zc, θref, δθc = problem

    _grav::FT = grav(param_set)
    _R_d::FT = R_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _p0::FT = MSLP(param_set)

    r = sqrt((x - xc) ^ 2 + (z - zc) ^ 2)

    δθ = r <= rc ? δθc : zero(FT)
    θ = θref + δθ

    π_exner = 1 - _grav / (_cp_d * θ) * z
    ρ = _p0 / (_R_d * θ) * π_exner ^ (_cv_d / _R_d)
    T = θ * π_exner

    state.ρ = ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = ρ * (_cv_d * T + aux.Φ)
end
