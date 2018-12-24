# `Parameters`
A simple type which allows for the specification of constant parameters in
CLIMA.

## Setup
```sh
julia --project -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
```

## Example Usage
Launch Julia with

```sh
julia --project
```

Then one can execute:

```julia
julia> using Parameters

julia> @parameter p1 3//5 "This is a parameter without units"
p1

julia> p1
3//5

help?> p1
search: p1 exp10 expm1 @macroexpand1 ComplexF16

  p1

  This is a parameter without units

  Examples
  ≡≡≡≡≡≡≡≡≡≡

  julia> p1
  3//5
```

One of the major reason for this module is to ensure that constant parameters
come into functions with the correct type

```julia
julia> @code_typed 1.0 + p1
CodeInfo(
313 1 ─ %1 = (Base.add_float)(x, 0.6)::Float64                                                                │╻ +
    └──      return %1                                                                                        │
) => Float64

julia> @code_typed Float32(1.0) + p1
CodeInfo(
313 1 ─ %1 = (Base.add_float)(x, 0.6)::Float32                                                                │╻ +
    └──      return %1                                                                                        │
) => Float32
```

## Usage in other CLIMA modules

To use this module in another part of CLIMA, you can load the module by using:
```julia
]dev PATH_TO_CLIMA_SRC/Parameters/
```
where `PATH_TO_CLIMA_SRC` is the relative path to the CLIMA source directory.
