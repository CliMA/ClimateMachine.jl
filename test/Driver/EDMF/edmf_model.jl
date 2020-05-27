#### EDMF model

#### Entrainment-Detrainment model

Base.@kwdef struct EntrainmentDetrainment{FT}
    "Entrainmnet TKE scale"
    c_λ::FT = 0.3
    "Entrainment factor"
    c_ε::FT = 0.13
    "Detrainment factor"
    c_δ::FT = 0.52
    "Trubulent Entrainment factor"
    c_t::FT = ?
    "Detrainment RH power"
    β::FT = 2
    "Logistic function scale"
    μ_0::FT = 0.0004
    "Updraft mixing fraction"
    χ::FT = 0.25
end

Base.@kwdef struct SurfaceModel{FT}
    "Surface specific humidity [kg/kg]"
    P_surf::FT = 0.0016
    "Surface pressure [pasc]"
    P_surf::FT = 101300.0
    "Surface internal energy []"
    surface_e_int::FT = 300.0*1004.0
    "Surface total specific humidity [kg/kg]"
    surface_q_tot::FT = 0.0
    "Surface I-flux [m^3/s^3]"
    e_int_surface_flux::FT = 0.0
    "Surface q_tot-flux [m/s*kg/kg]"
    q_tot_surface_flux::FT = 0.0
    "Top I-flux [m^3/s^3]"
    e_int_top_flux::FT = 0.0
    "Top q_tot-flux [m/s*kg/kg]"
    q_tot_top_flux::FT = 0.0
    "Sufcae area"
    a_surf::FT = 0.1
    "Ratio of rms turbulent velocity to friction velocity"
    κ_star ::FT = 1.94
    "fixed ustar" # YAIR - need to change this
    ustar::FT = 0.28
end

Base.@kwdef struct PressureModel{FT}
    "Pressure drag"
    α_d::FT = 10.0
    "Pressure advection"
    α_a::FT = 0.1
    "Pressure buoyancy"
    α_b::FT = 0.12
end

Base.@kwdef struct MixingLengthModel{FT}
    "Mixing lengths"
    L::MArray{Tuple{3},FT} = MArray{Tuple{3},FT}([0,0,0])
    "Eddy Viscosity"
    c_m::FT = 0.14
    "Eddy Diffusivity"
    c_k::FT = 0.22
    "Static Stability coefficient"
    c_b::FT = 0.63
    "Empirical stability function coefficient"
    a1 ::FT = -100
    "Empirical stability function coefficient"
    a2 ::FT = -0.2
end

Base.@kwdef struct MicrophysicsModel{FT}
    "enviromental cloud fraction"
    cf_initial::FT = 0.0 # need to define a function for cf
    "quadrature order" # can we code it that if order is 1 than we get mean ?  do we need the gaussian option?
    quadrature_order::FT = 3# yair needs to be a string: "mean", "gaussian quadrature", lognormal quadrature"
end

Base.@kwdef struct Environment{FT} <: BalanceLaw
end

Base.@kwdef struct Updraft{FT} <: BalanceLaw
end

Base.@kwdef struct EDMF{FT, N} <: BalanceLaw
    "Updrafts"
    updraft::NTuple{N,Updraft{FT}} = ntuple(i->Updraft{FT}(), N)
    "Environment"
    environment::Environment{FT} = Environment{FT}()
    "Entrainment-Detrainment model"
    entr_detr::EntrainmentDetrainment{FT} = EntrainmentDetrainment{FT}()
    "Pressure model"
    pressure::PressureModel{FT} = PressureModel{FT}()
    "Surface model"
    surface::SurfaceModel{FT} = SurfaceModel{FT}()
    "Surface model"
    micro_phys::MicrophysicsModel{FT} = MicrophysicsModel{FT}()
    "Mixing length model"
    mix_len::MixingLengthModel{FT} = MixingLengthModel{FT}()
end
