#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using JuliaFormatter

include("clima_formatter_options.jl")

headbranch = get(ARGS, 1, "master")

for filename in
    readlines(`git diff --name-only --diff-filter=AM $headbranch...`)
    endswith(filename, ".jl") || continue

    format(filename; clima_formatter_options...)
end
