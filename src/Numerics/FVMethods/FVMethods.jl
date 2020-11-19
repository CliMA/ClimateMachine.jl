module FVMethods

using StaticArrays
using KernelAbstractions.Extras: @unroll
using ..VariableTemplates

include("FVModel.jl")

end