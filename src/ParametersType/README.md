# `ParametersType`
A simple type which allows for the specification of constant parameters in
CLIMA. This package uses [`Unitful`](https://github.com/ajkeller34/Unitful.jl)
to allow the parameters to have units.

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
julia> using ParametersType

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

julia> using Unitful

julia> @parameter p2 4.5u"m/kg" "This parameter has units"
p2

julia> p2
4.5 m kg^-1

help?> p2
search: p2 exp2 ispow2 ComplexF32

  p2

  This parameter has units

  Examples
  ≡≡≡≡≡≡≡≡≡≡

  julia> p2
  4.5 m kg^-1
```

One of the major reason for this module is to ensure that constant parameters
come into functions with the correct type

```julia
julia> @code_typed 1.0 + p1
CodeInfo(
313 1 ─ %1 = (Base.add_float)(x, 0.6)::Float64                                                                │╻ +
    └──      return %1                                                                                        │
) => Float64

julia> @code_typed 1.0 + p2
CodeInfo(
313 1 ─ %1 = (Base.add_float)(x, 4.5)::Float64                                                                │╻ +
    └──      return %1                                                                                        │
) => Float64

julia> @code_typed Float32(1.0) + p2
CodeInfo(
313 1 ─ %1 = (Base.add_float)(x, 4.5)::Float32                                                                │╻ +
    └──      return %1                                                                                        │
) => Float32

julia> @code_typed Float32(1.0) + p1
CodeInfo(
313 1 ─ %1 = (Base.add_float)(x, 0.6)::Float32                                                                │╻ +
    └──      return %1                                                                                        │
) => Float32
```

## Usage in other CLIMA modules

To use this module in another part of CLIMA, you can load the module by using:
```julia
]dev PATH_TO_CLIMA_SRC/ParameterType/
```
where `PATH_TO_CLIMA_SRC` is the relative path to the CLIMA source directory.
