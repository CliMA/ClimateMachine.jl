module HyperParameters

export HyperParameterSet

"""
    ValidRange{T}

A valid range, and typical values,
for a given hyper-parameter.
"""
struct ValidRange{T}
  value_min::T
  value_typical::T
  value_max::T
end

"""
    HyperParameterSet

A hyper-parameter set for all variables to be
learned in CLIMA, containing minimum, typical,
and maximum values.
"""
struct HyperParameterSet{T}
  value_min::T
  value_typical::T
  value_max::T
end

"""
    HyperParameterSet

Collect hyper-parameters from
all modules in CLIMA.
"""
function HyperParameterSet()
  hyper_params = merge([
                        AtmosTurbConv(),
                        AtmosSGS(),
                        ]...)
  all_keys = keys(hyper_params)
  value_min     = Dict(k => hyper_params[k].value_min     for k in all_keys)
  value_typical = Dict(k => hyper_params[k].value_typical for k in all_keys)
  value_max     = Dict(k => hyper_params[k].value_max     for k in all_keys)

  return HyperParameterSet(value_min, value_typical, value_max)
end

"""
    AtmosTurbConv

Hyper-parameters in the [`TurbulenceConvection`](@ref) module.
"""
function AtmosTurbConv()
  params = Dict()
  params[:N_subdomains] = ValidRange(2, 2, 10) # Number of sub-domains
  params[:c_ε] = ValidRange(0.0, 0.12, 10.0)   # entr-detr factors
  params[:f_c] = ValidRange(0.0, 0.12, 1.0)    # buoyancy factors

  return params
end

"""
    AtmosTurbConv

Hyper-parameters in the Sub-Grid-Scale Physics.
"""
function AtmosSGS()
  params = Dict()
  params[:Cs] = ValidRange(0.0, 12/100, 10.0)  # Smagorinsky constant

  return params
end

end