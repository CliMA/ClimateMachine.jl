# System Image

To speed up the start time of `ClimateMachine` a custom system image can be
built with the
[`PackageCompiler`](https://github.com/JuliaLang/PackageCompiler.jl).

A helper script for doing this is provided at
[`.dev/systemimage/climate_machine_image.jl`](https://github.com/CliMA/ClimateMachine.jl/blob/master/.dev/systemimage/climate_machine_image.jl)

If called with

 - **no arguments**: will create the system image `ClimateMachine.so` in the
   `@__DIR__` directory (i.e., the directory in which the script is located)

 - **a single argument**: the system image will be placed in the path specified
   by the argument (relative to the callers path)

 - **a specified systemimg path and `true`**: the system image will compile the
   climate machine package module (useful for CI). This option should not be
   used when actually developing the climate machine package; see the
   [drawback](https://julialang.github.io/PackageCompiler.jl/dev/sysimages/#Drawbacks-to-custom-sysimages-1)
   from the `PackageCompiler` repository.

To run julia using the newly created system image use:
```
julia -J/PATH/TO/SYSTEM/IMAGE/ClimateMachine.so --project
```

!!! tip

    If you put the system image in your `.git` directory, your system image will
    not be remove by calls to `git clean`.

!!! warning

    If the climate machine `Manifest.toml` is updated you must build a new
    system image for these changes to be seen.
