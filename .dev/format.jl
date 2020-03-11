#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using JuliaFormatter

headbranch = get(ARGS, 1, "master")

for filename in readlines(`find . -print`)
    endswith(filename, ".jl") || continue

    format(
        filename,
        verbose = true,
        indent = 4,
        margin = 80,
        always_for_in = true,
        whitespace_typedefs = true,
        whitespace_ops_in_indices = true,
        remove_extra_newlines = false,
    )
end
