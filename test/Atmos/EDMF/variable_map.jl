#! format: off
var_map(s::String) = var_map(Val(Symbol(s)))
var_map(::Val{T}) where {T} = nothing

var_map(::Val{Symbol("ρ")}) = ("rho", ())
var_map(::Val{Symbol("ρu[1]")}) = ("u_mean", (:ρ,))
var_map(::Val{Symbol("ρu[2]")}) = ("v_mean", (:ρ,))
var_map(::Val{Symbol("moisture.ρq_tot")}) = ("qt_mean", (:ρ,))
var_map(::Val{Symbol("turbconv.updraft[1].ρa")}) = ("updraft_fraction", (:ρ,))
var_map(::Val{Symbol("turbconv.updraft[1].ρaw")}) = ("updraft_w", (:ρ, :a))
var_map(::Val{Symbol("turbconv.updraft[1].ρaq_tot")}) = ("updraft_qt", (:ρ, :a))
var_map(::Val{Symbol("turbconv.updraft[1].ρaθ_liq")}) = ("updraft_thetali", (:ρ, :a))
var_map(::Val{Symbol("turbconv.environment.ρatke")}) = ("tke_mean", (:ρ, :a))
var_map(::Val{Symbol("turbconv.environment.ρaθ_liq_cv")}) = ("env_thetali2", (:ρ, :a))
var_map(::Val{Symbol("turbconv.environment.ρaq_tot_cv")}) = ("env_qt2", (:ρ, :a))
#! format: on
