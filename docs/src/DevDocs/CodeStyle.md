# Coding conventions

For the most part, we follow the
[YASGuide](https://github.com/jrevels/YASGuide). Some key considerations:

- Limit use of Unicode as described in
  [AcceptableUnicode](AcceptableUnicode.md).

- Modules and struct names should follow TitleCase convention.

- Function names should be lowercase with words separated by underscores as
  necessary to improve readability.

- Variable names follow the format used in the [Variable
  List](VariableList.md). In addition, follow [CMIP
  conventions](http://clipc-services.ceda.ac.uk/dreq/) where possible and
  practicable.

- Document design and purpose rather than mechanics and implementation
  (document interfaces and embed documentation in code).

- Avoid variable names that coincide with module and struct names, as well as
  function/variable names that are natively supported.

- Never use the characters `l` (lowercase letter 'el'), `O` (uppercase letter
  'oh'), or `I` (uppercase letter 'eye') as single character variable names.

- Try to limit all lines to a maximum of 78 characters.

- `import`/`using` should be grouped in the following order:
  - Standard library imports.
  - Related third party imports.
  - Local application/library specific imports.
  - Use a blank line between each group of imports.

## Use `JuliaFormatter`

Once you are happy with your PR, apply our
[JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) settings to
all changed files in the repository from the top-level `ClimateMachine`
directory:
```
julia .dev/climaformat.jl <list of changed files>
```
This is easiest done by installing our formatting `githook`.

### Formatting `githook`

A `pre-commit` script can be placed in `$GIT_DIR/hooks/*` which will prevent
commits of incorrectly formatted Julia code.  It will also provide
instructions on how to format the code correctly.

Install the script with:

```
$ ln -s ../../.dev/hooks/pre-commit .git/hooks
```

Then, when you run `git commit`, an error message will be shown for staged
Julia files that are not formatted correctly. For example, if you try to commit
changes to `src/Arrays/MPIStateArrays.jl` that are not formatted correctly:

```
❯ git commit                                                                                                           │
Activating environment at `~/research/code/ClimateMachine.jl/.dev/Project.toml`                                        │
┌ Error: File src/Arrays/MPIStateArrays.jl needs to be indented with:                                                  │
│     julia /home/lucas/research/code/ClimateMachine.jl/.dev/climaformat.jl /home/lucas/research/code/ClimateMachine.jl│
/src/Arrays/MPIStateArrays.jl                                                                                          │
│ and added to the git index via                                                                                       │
│     git add /home/lucas/research/code/ClimateMachine.jl/src/Arrays/MPIStateArrays.jl                                 │
└ @ Main ~/research/code/ClimateMachine.jl/.git/hooks/pre-commit:30
```
Learn more about [`git hooks`](https://www.atlassian.com/git/tutorials/git-hooks).

### Precompiling `JuliaFormatter`

To speed up the formatter and the githook, a custom system image can be
built with the [`PackageCompiler`]. That said, the following [drawback]
from the `PackageCompiler` repository should be noted:

> It should be clearly stated that there are some drawbacks to using a custom
> sysimage, thereby sidestepping the standard Julia package precompilation
> system. The biggest drawback is that packages that are compiled into a
> sysimage (including their dependencies!) are "locked" to the version they
> where at when the sysimage was created. This means that no matter what package
> version you have installed in your current project, the one in the sysimage
> will take precedence. This can lead to bugs where you start with a project
> that needs a specific version of a package, but you have another one compiled
> into the sysimage.

The `PackageCompiler` compiler can be used with `JuliaFormatter` using the
following commands (from the top-level directory of a clone of
`ClimateMachine`):
```
$ julia -q
julia> using Pkg
julia> Pkg.add("PackageCompiler")
julia> using PackageCompiler
julia> Pkg.activate(joinpath(@__DIR__, ".dev"))
julia> using PackageCompiler
julia> PackageCompiler.create_sysimage(:JuliaFormatter; precompile_execution_file=joinpath(@__DIR__, ".dev/precompile.jl"), replace_default=true)
```

If you cannot or do not want to modify the default system image, use the
following instead:

```
$ julia -q
julia> using Pkg
julia> Pkg.add("PackageCompiler")
julia> using PackageCompiler
julia> Pkg.activate(joinpath(@__DIR__, ".dev"))
julia> PackageCompiler.create_sysimage(:JuliaFormatter; precompile_execution_file=joinpath(@__DIR__, ".dev/precompile.jl"), sysimage_path=joinpath(@__DIR__, ".git/hooks/JuliaFormatterSysimage.so"))
```

In this case, use the `pre-commit.sysimage` `git hook` with:

```
$ ln -s ../../.dev/hooks/pre-commit.sysimage .git/hooks/pre-commit
```

Note: Putting the system image in `.git/hooks` protects it from calls to
`git clean -x`.
