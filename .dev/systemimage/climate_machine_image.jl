#!/usr/bin/env julia
#
# Called with no arguments will create the system image
#     ClimateMachine.so
# in the `@__DIR__` directory.
#
# Called with a single argument the system image will be placed in the path
# specified by the argument (relative to the callers path)
#
# Called with a specified systemimg path and `true`, the system image will
# compile the climate machine package module (useful for CI)

sysimage_path =
    isempty(ARGS) ? joinpath(@__DIR__, "ClimateMachine.so") : abspath(ARGS[1])

climatemachine_pkg = get(ARGS, 2, "false") == "true"

@info "Creating system image object file at: '$(sysimage_path)'"
@info "Building ClimateMachine into system image: $(climatemachine_pkg)"

start_time = time()

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate(verbose = true)

pkgs = Symbol[]
if climatemachine_pkg
    push!(pkgs, :ClimateMachine)
else
    # TODO: reorg project structure (Project.toml) to use
    # Pkg.dependencies() with PackageCompiler
    #if VERSION >= v"1.4"
    #    append!(pkgs, [Symbol(v.name) for v in values(Pkg.dependencies())])
    #end
    append!(pkgs, [Symbol(name) for name in keys(Pkg.installed())])
end

# use package compiler
Pkg.add("PackageCompiler")
using PackageCompiler
PackageCompiler.create_sysimage(
    pkgs,
    sysimage_path = sysimage_path,
    precompile_execution_file = joinpath(
        @__DIR__,
        "..",
        "..",
        "test",
        "Numerics",
        "DGMethods",
        "Euler",
        "isentropicvortex.jl",
    ),
)

tot_secs = Int(floor(time() - start_time))
@info "Created system image object file at: $(sysimage_path)"
@info "System image build time: $tot_secs sec"
