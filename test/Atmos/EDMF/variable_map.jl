#! format: off
var_map(s::String) = var_map(Val(Symbol(s)))
var_map(::Val{T}) where {T} = nothing

var_map(::Val{:prog_ρ}) = ("rho", ())
var_map(::Val{:prog_ρu_1}) = ("u_mean", (:ρ,))
var_map(::Val{:prog_ρu_2}) = ("v_mean", (:ρ,))
var_map(::Val{:prog_moisture_ρq_tot}) = ("qt_mean", (:ρ,))
var_map(::Val{:prog_turbconv_updraft_1_ρa}) = ("updraft_fraction", (:ρ,))
var_map(::Val{:prog_turbconv_updraft_1_ρaw}) = ("updraft_w", (:ρ, :a))
var_map(::Val{:prog_turbconv_updraft_1_ρaq_tot}) = ("updraft_qt", (:ρ, :a))
var_map(::Val{:prog_turbconv_updraft_1_ρaθ_liq}) = ("updraft_thetali", (:ρ, :a))
var_map(::Val{:prog_turbconv_environment_ρatke}) = ("tke_mean", (:ρ, :a))
var_map(::Val{:prog_turbconv_environment_ρaθ_liq_cv}) = ("env_thetali2", (:ρ, :a))
var_map(::Val{:prog_turbconv_environment_ρaq_tot_cv}) = ("env_qt2", (:ρ, :a))
#! format: on
