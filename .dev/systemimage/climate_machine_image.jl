#!/usr/bin/env julia
#
# Called with no arguments will create the system image
#     ClimateMachine.so
# in the `@__DIR__` directory.
#
# Called with a single argument the system image will be placed in the path
# specified by the argument (relative to the callers path)
sysimage_path =
    isempty(ARGS) ? joinpath(@__DIR__, "ClimateMachine.so") : ARGS[1]

using Pkg
Pkg.add("PackageCompiler")
using PackageCompiler
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()
pkgs = Pkg.installed()
delete!(pkgs, "Pkg")

# "TF_BUILD" is check to see if we are running on Azure for CI
if !haskey(ENV, "TF_BUILD")
    PackageCompiler.create_sysimage(
        [Symbol(s) for s in keys(pkgs)];
        sysimage_path = sysimage_path,
        precompile_execution_file = joinpath(
            @__DIR__,
            "..",
            "..",
            "test",
            "Numerics",
            "DGmethods",
            "Euler",
            "isentropicvortex.jl",
        ),
    )
else
    return true
end
