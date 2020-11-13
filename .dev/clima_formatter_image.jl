#!/usr/bin/env julia
#
# Called with no arguments will build the formattter system image 
#   PATH_TO_CLIMATEMACHINE/.git/hooks/JuliaFormatterSysimage.so
#
# Called with a single argument the system image for the formatter will be
# placed in the path specified by the argument (relative to the callers path)

sysimage_path = abspath(
    isempty(ARGS) ?
    joinpath(@__DIR__, "..", ".git", "hooks", "JuliaFormatterSysimage.so") :
    ARGS[1],
)

@info """
Creating system image object file at:
    $(sysimage_path)
"""

using Pkg
Pkg.add("PackageCompiler")
using PackageCompiler

# If a current Manifest exist for the formatter we remove it so that we have the
# latest version
rm(joinpath(@__DIR__, "Manifest.toml"); force = true)

Pkg.activate(joinpath(@__DIR__))
PackageCompiler.create_sysimage(
    :JuliaFormatter;
    precompile_execution_file = joinpath(@__DIR__, "precompile.jl"),
    sysimage_path = sysimage_path,
)
