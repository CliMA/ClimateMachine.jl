using JuliaFormatter

include("clima_formatter_options.jl")

format(@__FILE__; clima_formatter_options...)
