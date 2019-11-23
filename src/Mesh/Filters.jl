module Filters

using LinearAlgebra, GaussQuadrature, GPUifyLoops
using ..Grids

export AbstractSpectralFilter, AbstractFilter
export ExponentialFilter, CutoffFilter, TMARFilter

abstract type AbstractFilter end
abstract type AbstractSpectralFilter <: AbstractFilter end

include("Filters_kernels.jl")

"""
    spectral_filter_matrix(r, Nc, σ)

Returns the filter matrix that takes function values at the interpolation
`N+1` points, `r`, converts them into Legendre polynomial basis coefficients,
multiplies
```math
σ((n-N_c)/(N-N_c))
```
against coefficients `n=Nc:N` and evaluates the resulting polynomial at the
points `r`.
"""
function spectral_filter_matrix(r, Nc, σ)
  N = length(r)-1
  T = eltype(r)

  @assert N >= 0
  @assert 0 <= Nc <= N

  a, b = GaussQuadrature.legendre_coefs(T, N)
  V = GaussQuadrature.orthonormal_poly(r, a, b)

  Σ = ones(T, N+1)
  Σ[(Nc:N).+1] .= σ.(((Nc:N).-Nc)./(N-Nc))

  V*Diagonal(Σ)/V
end

"""
    ExponentialFilter(grid, Nc=0, s=32, α=-log(eps(eltype(grid))))

Returns the spectral filter with the filter function
```math
σ(η) = \exp(-α η^s)
```
where `s` is the filter order (must be even), the filter starts with
polynomial order `Nc`, and `alpha` is a parameter controlling the smallest
value of the filter function.
"""
struct ExponentialFilter <: AbstractSpectralFilter
  "filter matrix"
  filter

  function ExponentialFilter(grid, Nc=0, s=32,
                             α=-log(eps(eltype(grid))))
    AT = arraytype(grid)
    N = polynomialorder(grid)
    ξ = referencepoints(grid)

    @assert iseven(s)
    @assert 0 <= Nc <= N

    σ(η) = exp(-α*η^s)
    filter = spectral_filter_matrix(ξ, Nc, σ)

    new(AT(filter))
  end
end

"""
    CutoffFilter(grid, Nc=polynomialorder(grid))

Returns the spectral filter that zeros out polynomial modes greater than or
equal to `Nc`.
"""
struct CutoffFilter <: AbstractSpectralFilter
  "filter matrix"
  filter

  function CutoffFilter(grid, Nc=polynomialorder(grid))
    AT = arraytype(grid)
    ξ = referencepoints(grid)

    σ(η) = 0
    filter = spectral_filter_matrix(ξ, Nc, σ)

    new(AT(filter))
  end
end

"""
    TMARFilter()

Returns the truncation and mass aware rescaling nonnegativity preservation
filter.  The details of this filter are described in

    @article{doi:10.1175/MWR-D-16-0220.1,
      author = {Light, Devin and Durran, Dale},
      title = {Preserving Nonnegativity in Discontinuous Galerkin
               Approximations to Scalar Transport via Truncation and Mass
               Aware Rescaling (TMAR)},
      journal = {Monthly Weather Review},
      volume = {144},
      number = {12},
      pages = {4771-4786},
      year = {2016},
      doi = {10.1175/MWR-D-16-0220.1},
    }

Note this needs to be used with a restrictive time step or a flux correction
to ensure that grid integral is conserved.

## Examples

This filter can be applied to the 3rd and 4th fields of an `MPIStateArray` `Q`
with the code

```julia
Filters.apply!(Q, (3, 4), grid, TMARFilter())
```

where `grid` is the associated `DiscontinuousSpectralElementGrid`.
"""
struct TMARFilter <: AbstractFilter
end

"""
    apply!(Q, states, grid::DiscontinuousSpectralElementGrid,
           filter::AbstractSpectralFilter; horizontal = true,
           vertical = true)

Applies `filter` to the `states` of `Q`.

The arguments `horizontal` and `vertical` are used to control if the filter
is applied in the horizontal and vertical reference directions, respectively.
Note, it is assumed that the trailing dimension is the vertical dimension and
the rest are horizontal.
"""
function apply!(Q, states, grid::DiscontinuousSpectralElementGrid,
                filter::AbstractSpectralFilter;
                horizontal = true, vertical = true)
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)

  nstate = size(Q, 2)

  filtermatrix = filter.filter
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  nelem = length(topology.elems)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  nrealelem = length(topology.realelems)

  @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
          knl_apply_filter!(Val(dim), Val(N), Val(nstate), Val(horizontal), Val(vertical),
                            Q.data, Val(states), filtermatrix, topology.realelems))
end

"""
    apply!(Q, states, grid::DiscontinuousSpectralElementGrid, ::TMARFilter)

Applies the truncation and mass aware rescaling to `states` of `Q`.  This
rescaling keeps the states nonegative while keeping the element average
the same.
"""
function apply!(Q, states, grid::DiscontinuousSpectralElementGrid,
                ::TMARFilter)
  topology = grid.topology

  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  nrealelem = length(topology.realelems)
  nreduce = 2^ceil(Int, log2(Nq*Nqk))

  @launch(device, threads=(Nq, Nqk, 1), blocks=nrealelem,
          knl_apply_TMAR_filter!(Val(nreduce), Val(dim), Val(N), Q.data,
                                 Val(states), grid.vgeo, topology.realelems))
end

end
