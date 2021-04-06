#!/usr/bin/env julia
#! format: off
#
# Called with no arguments will replace the default system image with one that
# includes the JuliaFormatter
#
# Called with a single argument the system image for the formatter will be
# placed in the path specified by the argument (relative to the callers path)

using Pkg
Pkg.add("PackageCompiler")
using PackageCompiler

if isfile(PackageCompiler.backup_default_sysimg_path())
    @error """
    A custom default system image already exists.
    Either restore default with:
        julia -e "using PackageCompiler; PackageCompiler.restore_default_sysimage()"
    or use the script
        $(abspath(joinpath(@__DIR__, "..", ".dev", "clima_formatter_image.jl")))
    which will use a custom path for the system image
    """
    exit(1)
end

# If a current Manifest exist for the formatter we remove it so that we have the
# latest version
rm(joinpath(@__DIR__, "Manifest.toml"); force = true)

Pkg.activate(joinpath(@__DIR__))
PackageCompiler.create_sysimage(
    :JuliaFormatter;
    precompile_execution_file = joinpath(@__DIR__, "precompile.jl"),
    replace_default = true,
)
