include("../DryAtmos.jl")

import CLIMAParameters
CLIMAParameters.Planet.MSLP(::EarthParameterSet) = 1e5

Base.@kwdef struct GravityWave{FT} <: AbstractDryAtmosProblem
  T_ref::FT = 250
  ΔT::FT = 0.0001
  H::FT = 10e3
  u_0::FT = 20
  f::FT = 0
  L::FT
  d::FT
  x_c::FT
  timeend::FT
end
gw_small_setup(FT) = GravityWave{FT}(L=300e3, d=5e3, x_c=100e3, timeend=30*60)
gw_large_setup(FT) = GravityWave{FT}(L=24000e3, d=400e3, x_c=8000e3, timeend=3000*60)

function vars_state(::DryAtmosModel, ::GravityWave, ::Auxiliary, FT)
  @vars begin
    ρ_exact::FT
    ρu_exact::SVector{3, FT}
    ρe_exact::FT
  end
end

function init_state_prognostic!(bl::DryAtmosModel, 
                                problem::GravityWave,
                                state, aux, localgeo, t)
    x, z, _ = localgeo.coord
    FT = eltype(state)
    _R_d::FT = R_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _cv_d::FT = cv_d(param_set)
    p_s::FT = MSLP(param_set)
    g::FT = grav(param_set)

    L = problem.L
    d = problem.d
    x_c = problem.x_c
    u_0 = problem.u_0
    H = problem.H
    T_ref = problem.T_ref
    ΔT = problem.ΔT
    f = problem.f
  
    δ = g / (_R_d * T_ref)
    c_s = sqrt(_cp_d / _cv_d * _R_d * T_ref)
    ρ_s = p_s / (T_ref * _R_d)

    if t == 0
      δT_b = ΔT * exp(-(x - x_c) ^ 2 / d ^ 2) * sin(π * z / H)
      δT = exp(δ * z / 2) * δT_b
      δρ_b = -ρ_s * δT_b / T_ref
      δρ = exp(-δ * z / 2) * δρ_b
      δu, δv, δw = 0, 0, 0
    else
      xp = x - u_0 * t

      δρ_b, δu_b, δv_b, δw_b, δp_b = zeros(SVector{5, Complex{FT}})
      for m in (-1, 1)
        for n in -100:100
          k_x = 2π * n / L
          k_z = π * m / H

          p_1 = c_s ^ 2 * (k_x ^ 2 + k_z ^ 2 + δ ^ 2 / 4) + f ^ 2
          q_1 = g * k_x ^ 2 * (c_s ^ 2 * δ - g) + c_s ^ 2 * f ^ 2 * (k_z ^ 2 + δ ^ 2 / 4)
          
          α = sqrt(p_1 / 2 - sqrt(p_1 ^ 2 / 4 - q_1))
          β = sqrt(p_1 / 2 + sqrt(p_1 ^ 2 / 4 - q_1))

          fac1 = 1 / (β ^ 2 - α ^ 2) 
          L_m1 = (-cos(α * t) / α ^ 2 + cos(β * t) / β ^ 2) * fac1 + 1 / (α ^ 2 * β ^ 2)
          L_0 = (sin(α * t) / α - sin(β * t) / β) * fac1
          L_1 = (cos(α * t) - cos(β * t)) * fac1
          L_2 = (-α * sin(α * t) + β * sin(β * t)) * fac1
          L_3 = (-α ^ 2 * cos(α * t) + β ^ 2 * cos(β * t)) * fac1
          
          if α == 0
            L_m1 = (β ^ 2 * t ^ 2 - 1 + cos(β * t)) / β ^ 4
            L_0 = (β * t - sin(β * t)) / β ^ 3
          end
      
          δρ̃_b0 = -ρ_s / T_ref * ΔT / sqrt(π) * d / L *
                  exp(-d ^ 2 * k_x ^ 2 / 4) * exp(-im * k_x * x_c) * k_z * H / 2im

          δρ̃_b = (L_3 + (p_1 + g * (im * k_z - δ / 2)) * L_1 +
                (c_s ^ 2 * (k_z ^ 2 + δ ^ 2 / 4) + g * (im * k_z - δ / 2)) * f ^ 2 * L_m1) * δρ̃_b0

          δp̃_b = -(g - c_s ^ 2 * (im * k_z + δ / 2)) * (L_1 + f ^ 2 * L_m1) * g * δρ̃_b0

          δũ_b = im * k_x * (g - c_s ^ 2 * (im * k_z + δ / 2)) * L_0 * g * δρ̃_b0 / ρ_s

          δṽ_b = -f * im * k_x * (g - c_s ^ 2 * (im * k_z + δ / 2)) * L_m1 * g * δρ̃_b0 / ρ_s 

          δw̃_b = -(L_2 + (f ^ 2 + c_s ^ 2 * k_x ^ 2) * L_0) * g * δρ̃_b0 / ρ_s 

          expfac = exp(im * (k_x * xp + k_z * z)) 
          
          δρ_b += δρ̃_b * expfac
          δp_b += δp̃_b * expfac

          δu_b += δũ_b * expfac
          δv_b += δṽ_b * expfac
          δw_b += δw̃_b * expfac
        end
      end

      δρ = exp(-δ * z / 2) * real(δρ_b)
      δp = exp(-δ * z / 2) * real(δp_b)

      δu = exp(δ * z / 2) * real(δu_b)
      δv = exp(δ * z / 2) * real(δv_b)
      δw = exp(δ * z / 2) * real(δw_b)

      δT_b = T_ref * (δp_b / p_s - δρ_b / ρ_s)
      δT = exp(δ * z / 2) * real(δT_b)
    end
   
    ρ = ρ_s * exp(-δ * z) + δρ
    T = T_ref + δT
    
    #ρ = aux.ref_state.ρ + δρ
    #T = aux.ref_state.T + δT

    u = SVector{3, FT}(u_0 + δu, δw, 0)
    e_kin = u' * u / 2
    e_pot = aux.Φ
    e_int = _cv_d * T
    ρe_tot = ρ * (e_int + e_kin + e_pot)

    state.ρ = ρ
    state.ρu = ρ * u
    state.ρe = ρe_tot
end
