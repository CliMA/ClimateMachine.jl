module Filters

using SpecialFunctions
using LinearAlgebra, GaussQuadrature, KernelAbstractions
using KernelAbstractions.Extras: @unroll
using StaticArrays
using ..Grids
using ..Grids: Direction, EveryDirection, HorizontalDirection, VerticalDirection

using ...MPIStateArrays
using ...VariableTemplates: @vars, varsize, Vars, varsindices

export AbstractSpectralFilter, AbstractFilter
export ExponentialFilter, CutoffFilter, TMARFilter, BoydVandevenFilter

abstract type AbstractFilter end
abstract type AbstractSpectralFilter <: AbstractFilter end

"""
    AbstractFilterTarget

An abstract type representing variables that the filter
will act on
"""
abstract type AbstractFilterTarget end

"""
    vars_state_filtered(::AbstractFilterTarget, FT)

A tuple of symbols containing variables that the filter
will act on given a float type `FT`
"""
function vars_state_filtered end

"""
    compute_filter_argument!(::AbstractFilterTarget,
                             state_filter::Vars,
                             state::Vars,
                             state_auxiliary::Vars)

Compute filter argument `state_filter` based on `state`
and `state_auxiliary`
"""
function compute_filter_argument! end
"""
    compute_filter_result!(::AbstractFilterTarget,
                           state::Vars,
                           state_filter::Vars,
                           state_auxiliary::Vars)

Compute filter result `state` based on the filtered state
`state_filter` and `state_auxiliary`
"""
function compute_filter_result! end

number_state_filtered(t::AbstractFilterTarget, FT) =
    varsize(vars_state_filtered(t, FT))

"""
    FilterIndices(I)

Filter variables based on their indices `I` where `I` can
be a range or a list of indices

## Examples
```julia
FiltersIndices(1:3)
FiltersIndices(1, 3, 5)
```
"""
struct FilterIndices{I} <: AbstractFilterTarget
    FilterIndices(I::Integer...) = new{I}()
    FilterIndices(I::AbstractRange) = new{I}()
end
vars_state_filtered(::FilterIndices{I}, FT) where {I} =
    @vars(_::SVector{length(I), FT})

function compute_filter_argument!(
    ::FilterIndices{I},
    filter_state::Vars,
    state::Vars,
    aux::Vars,
) where {I}
    @unroll for s in 1:length(I)
        @inbounds parent(filter_state)[s] = parent(state)[I[s]]
    end
end

function compute_filter_result!(
    ::FilterIndices{I},
    state::Vars,
    filter_state::Vars,
    aux::Vars,
) where {I}
    @unroll for s in 1:length(I)
        @inbounds parent(state)[I[s]] = parent(filter_state)[s]
    end
end


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
    N = length(r) - 1
    T = eltype(r)

    @assert N >= 0
    @assert 0 <= Nc <= N

    a, b = GaussQuadrature.legendre_coefs(T, N)
    V = (N == 0 ? ones(T, 1, 1) : GaussQuadrature.orthonormal_poly(r, a, b))

    Σ = ones(T, N + 1)
    Σ[(Nc:N) .+ 1] .= σ.(((Nc:N) .- Nc) ./ (N - Nc))

    V * Diagonal(Σ) / V
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
struct ExponentialFilter{FM} <: AbstractSpectralFilter
    "filter matrices in all directions (tuple of filter matrices)"
    filter_matrices::FM

    function ExponentialFilter(
        grid,
        Nc = 0,
        s = 32,
        α = -log(eps(eltype(grid))),
    )
        dim = dimensionality(grid)

        # Support different filtering thresholds in different
        # directions (default behavior is to apply the same threshold
        # uniformly in all directions)
        if Nc isa Integer
            Nc = ntuple(i -> Nc, dim)
        elseif Nc isa NTuple{2} && dim == 3
            Nc = (Nc[1], Nc[1], Nc[2])
        end
        @assert length(Nc) == dim

        # Tuple of polynomial degrees (N₁, N₂, N₃)
        N = polynomialorders(grid)
        # In 2D, we assume same polynomial order in the horizontal
        @assert dim == 2 || N[1] == N[2]
        @assert iseven(s)
        @assert all(0 .<= Nc .<= N)

        σ(η) = exp(-α * η^s)

        AT = arraytype(grid)
        ξ = referencepoints(grid)
        filter_matrices =
            ntuple(i -> AT(spectral_filter_matrix(ξ[i], Nc[i], σ)), dim)
        new{typeof(filter_matrices)}(filter_matrices)
    end
end

"""
    BoydVandevenFilter(grid, Nc=0, s=32)

Returns the spectral filter using the logorithmic error function of
the form:
```math
σ(η) = 1/2 erfc(2*sqrt(s)*χ(η)*(abs(η)-0.5))
```
whenever s ≤ i ≤ N, and 1 otherwise. The function `χ(η)` is defined
as
```math
χ(η) = sqrt(-log(1-4*(abs(η)-0.5)^2)/(4*(abs(η)-0.5)^2))
```
if `x != 0.5` and `1` otherwise. Here, `s` is the filter order,
the filter starts with polynomial order `Nc`, and `alpha` is a parameter
controlling the smallest value of the filter function.

### References
 - [Boyd1996](@cite)
"""
struct BoydVandevenFilter{FM} <: AbstractSpectralFilter
    "filter matrices in all directions (tuple of filter matrices)"
    filter_matrices::FM

    function BoydVandevenFilter(grid, Nc = 0, s = 32)
        dim = dimensionality(grid)

        # Support different filtering thresholds in different
        # directions (default behavior is to apply the same threshold
        # uniformly in all directions)
        if Nc isa Integer
            Nc = ntuple(i -> Nc, dim)
        elseif Nc isa NTuple{2} && dim == 3
            Nc = (Nc[1], Nc[1], Nc[2])
        end
        @assert length(Nc) == dim

        # Tuple of polynomial degrees (N₁, N₂, N₃)
        N = polynomialorders(grid)
        # In 2D, we assume same polynomial order in the horizontal
        @assert dim == 2 || N[1] == N[2]
        @assert iseven(s)
        @assert all(0 .<= Nc .<= N)

        function σ(η)
            a = 2 * abs(η) - 1
            χ = iszero(a) ? one(a) : sqrt(-log1p(-a^2) / a^2)
            return erfc(sqrt(s) * χ * a) / 2
        end

        AT = arraytype(grid)
        ξ = referencepoints(grid)
        filter_matrices =
            ntuple(i -> AT(spectral_filter_matrix(ξ[i], Nc[i], σ)), dim)
        new{typeof(filter_matrices)}(filter_matrices)
    end
end

"""
    CutoffFilter(grid, Nc=polynomialorders(grid))

Returns the spectral filter that zeros out polynomial modes greater than or
equal to `Nc`.
"""
struct CutoffFilter{FM} <: AbstractSpectralFilter
    "filter matrices in all directions (tuple of filter matrices)"
    filter_matrices::FM

    function CutoffFilter(grid, Nc = polynomialorders(grid))
        dim = dimensionality(grid)

        # Support different filtering thresholds in different
        # directions (default behavior is to apply the same threshold
        # uniformly in all directions)
        if Nc isa Integer
            Nc = ntuple(i -> Nc, dim)
        elseif Nc isa NTuple{2} && dim == 3
            Nc = (Nc[1], Nc[1], Nc[2])
        end
        @assert length(Nc) == dim

        # Tuple of polynomial degrees (N₁, N₂, N₃)
        N = polynomialorders(grid)
        # In 2D, we assume same polynomial order in the horizontal
        @assert dim == 2 || N[1] == N[2]
        @assert all(0 .<= Nc .<= N)

        σ(η) = 0

        AT = arraytype(grid)
        ξ = referencepoints(grid)
        filter_matrices =
            ntuple(i -> AT(spectral_filter_matrix(ξ[i], Nc[i], σ)), dim)
        new{typeof(filter_matrices)}(filter_matrices)
    end
end

"""
    TMARFilter()

Returns the truncation and mass aware rescaling nonnegativity preservation
filter.  The details of this filter are described in [Light2016](@cite)

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
struct TMARFilter <: AbstractFilter end

"""
    Filters.apply!(Q::MPIStateArray,
        target,
        grid::DiscontinuousSpectralElementGrid,
        filter::AbstractSpectralFilter;
        kwargs...)

Applies `filter` to `Q` given a `grid` and a custom `target`.

A `target` can be any of the following:
 - a tuple or range of indices
 - a tuple of symbols or strings of variable names
 - a colon (`:`) to apply to all variables
 - a custom [`AbstractFilterTarget`]

The following keyword arguments are supported for some filters:
- `direction`: for `AbstractSpectralFilter` controls if the filter is
  applied in the horizontal and/or vertical directions. It is assumed that the
  trailing dimension on the reference element is the vertical dimension and the
  rest are horizontal.
- `state_auxiliary`: if `target` requires auxiliary state to compute its argument or results.

# Examples

Specifying the `target` via indices:
```julia
Filters.apply!(Q, :, grid, TMARFilter())
Filters.apply!(Q, (1, 3), grid, CutoffFilter(grid); direction=VerticalDirection())
```

Speciying `target` via symbols or strings:
```julia
Filters.apply!(Q, (:ρ, "energy.ρe"), grid, TMARFilter())
Filters.apply!(Q, ("moisture.ρq_tot",), grid, CutoffFilter(grid);
               direction=VerticalDirection())
```
"""
function apply!(
    Q,
    target,
    grid::DiscontinuousSpectralElementGrid,
    filter::AbstractFilter;
    kwargs...,
)
    device = typeof(Q.data) <: Array ? CPU() : CUDADevice()
    event = Event(device)
    event =
        apply_async!(Q, target, grid, filter; dependencies = event, kwargs...)
    wait(device, event)
end


"""
    Filters.apply_async!(Q, target, grid::DiscontinuousSpectralElementGrid,
        filter::AbstractFilter;
        dependencies,
        kwargs...)

An asynchronous version of [`Filters.apply!`](@ref), returning an `Event`
object. `dependencies` should be an `Event` or tuple of `Event`s which need to
finish before applying the filter.

```julia
compstream = Filters.apply_async!(Q, :, grid, CutoffFilter(grid); dependencies=compstream)
wait(compstream)
```
"""
function apply_async! end

function apply_async!(
    Q,
    target::AbstractFilterTarget,
    grid::DiscontinuousSpectralElementGrid,
    filter::AbstractSpectralFilter;
    dependencies,
    state_auxiliary = nothing,
    direction = EveryDirection(),
)
    topology = grid.topology

    # Tuple of polynomial degrees (N₁, N₂, N₃)
    N = polynomialorders(grid)
    # In 2D, we assume same polynomial order in the horizontal
    dim = dimensionality(grid)
    # Currently only support same polynomial in both horizontal directions
    @assert N[1] == N[2]

    device = typeof(Q.data) <: Array ? CPU() : CUDADevice()

    nelem = length(topology.elems)
    # Number of Gauss-Lobatto quadrature points in each direction
    Nq = N .+ 1
    Nq1 = Nq[1]
    Nq2 = Nq[2]
    Nq3 = dim == 2 ? 1 : Nq[dim]

    nrealelem = length(topology.realelems)
    event = dependencies

    if direction isa EveryDirection || direction isa HorizontalDirection
        @assert dim == 2 || Nq1 == Nq2
        filtermatrix = filter.filter_matrices[1]
        event = kernel_apply_filter!(device, (Nq1, Nq2, Nq3))(
            Val(dim),
            Val(N),
            Val(vars(Q)),
            Val(isnothing(state_auxiliary) ? nothing : vars(state_auxiliary)),
            HorizontalDirection(),
            Q.data,
            isnothing(state_auxiliary) ? nothing : state_auxiliary.data,
            target,
            filtermatrix,
            ndrange = (nrealelem * Nq1, Nq2, Nq3),
            dependencies = event,
        )
    end
    if direction isa EveryDirection || direction isa VerticalDirection
        filtermatrix = filter.filter_matrices[end]
        event = kernel_apply_filter!(device, (Nq1, Nq2, Nq3))(
            Val(dim),
            Val(N),
            Val(vars(Q)),
            Val(isnothing(state_auxiliary) ? nothing : vars(state_auxiliary)),
            VerticalDirection(),
            Q.data,
            isnothing(state_auxiliary) ? nothing : state_auxiliary.data,
            target,
            filtermatrix,
            ndrange = (nrealelem * Nq1, Nq2, Nq3),
            dependencies = event,
        )
    end
    return event
end


function apply_async!(
    Q,
    target::AbstractFilterTarget,
    grid::DiscontinuousSpectralElementGrid,
    ::TMARFilter;
    dependencies,
)
    topology = grid.topology

    device = typeof(Q.data) <: Array ? CPU() : CUDADevice()

    dim = dimensionality(grid)
    N = polynomialorders(grid)
    # Currently only support same polynomial in both horizontal directions
    @assert dim == 2 || N[1] == N[2]
    Nqs = N .+ 1
    Nq = Nqs[1]
    Nqj = dim == 2 ? 1 : Nqs[2]

    nrealelem = length(topology.realelems)
    nreduce = 2^ceil(Int, log2(Nq * Nqj))

    event = dependencies
    event = kernel_apply_TMAR_filter!(device, (Nq, Nqj), (nrealelem * Nq, Nqj))(
        Val(nreduce),
        Val(dim),
        Val(N),
        Q.data,
        target,
        grid.vgeo,
        dependencies = event,
    )
    return event
end

function apply_async!(
    Q,
    indices::Union{Colon, AbstractRange, Tuple{Vararg{Integer}}},
    grid::DiscontinuousSpectralElementGrid,
    filter::AbstractFilter;
    kwargs...,
)
    if indices isa Colon
        indices = 1:size(Q, 2)
    end
    apply_async!(Q, FilterIndices(indices...), grid, filter; kwargs...)
end

function apply_async!(
    Q,
    vs::Tuple,
    grid::DiscontinuousSpectralElementGrid,
    filter::AbstractFilter;
    kwargs...,
)
    apply_async!(
        Q,
        FilterIndices(varsindices(vars(Q), vs)...),
        grid,
        filter;
        kwargs...,
    )
end

const _M = Grids._M

@doc """
    kernel_apply_filter!(::Val{dim}, ::Val{N}, direction,
                         Q, state_auxiliary, target, filtermatrix
                        ) where {dim, N}

Computational kernel: Applies the `filtermatrix` to `Q` given a
custom target `target`.

The `direction` argument is used to control if the filter is applied in the
horizontal and/or vertical reference directions.
""" kernel_apply_filter!
@kernel function kernel_apply_filter!(
    ::Val{dim},
    ::Val{N},
    ::Val{vars_Q},
    ::Val{vars_state_auxiliary},
    direction,
    Q,
    state_auxiliary,
    target::AbstractFilterTarget,
    filtermatrix,
) where {dim, N, vars_Q, vars_state_auxiliary}
    @uniform begin
        FT = eltype(Q)

        Nqs = N .+ 1
        Nq1 = Nqs[1]
        Nq2 = Nqs[2]
        Nq3 = dim == 2 ? 1 : Nqs[dim]

        if direction isa EveryDirection
            filterinξ1 = filterinξ2 = true
            filterinξ3 = dim == 2 ? false : true
        elseif direction isa HorizontalDirection
            filterinξ1 = true
            filterinξ2 = dim == 2 ? false : true
            filterinξ3 = false
        elseif direction isa VerticalDirection
            filterinξ1 = false
            filterinξ2 = dim == 2 ? true : false
            filterinξ3 = dim == 2 ? false : true
        end

        nstates = varsize(vars_Q)
        nfilterstates = number_state_filtered(target, FT)
        nfilteraux =
            isnothing(state_auxiliary) ? 0 : varsize(vars_state_auxiliary)

        # ugly workaround around problems with @private
        # hopefully will be soon fixed in KA
        l_Q2 = MVector{nstates, FT}(undef)
        l_Qfiltered2 = MVector{nfilterstates, FT}(undef)
    end

    s_Q = @localmem FT (Nq1, Nq2, Nq3, nfilterstates)
    l_Q = @private FT (nstates,)
    l_Qfiltered = @private FT (nfilterstates,)
    l_aux = @private FT (nfilteraux,)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

        @unroll for s in 1:nstates
            l_Q[s] = Q[ijk, s, e]
        end

        @unroll for s in 1:nfilteraux
            l_aux[s] = state_auxiliary[ijk, s, e]
        end

        fill!(l_Qfiltered2, -zero(FT))

        compute_filter_argument!(
            target,
            Vars{vars_state_filtered(target, FT)}(l_Qfiltered2),
            Vars{vars_Q}(l_Q[:]),
            Vars{vars_state_auxiliary}(l_aux[:]),
        )

        @unroll for fs in 1:nfilterstates
            l_Qfiltered[fs] = zero(FT)
        end

        @unroll for fs in 1:nfilterstates
            s_Q[i, j, k, fs] = l_Qfiltered2[fs]
        end

        if filterinξ1
            @synchronize
            @unroll for n in 1:Nq1
                @unroll for fs in 1:nfilterstates
                    l_Qfiltered[fs] += filtermatrix[i, n] * s_Q[n, j, k, fs]
                end
            end

            if filterinξ2 || filterinξ3
                @synchronize
                @unroll for fs in 1:nfilterstates
                    s_Q[i, j, k, fs] = l_Qfiltered[fs]
                    l_Qfiltered[fs] = zero(FT)
                end
            end
        end

        if filterinξ2
            @synchronize
            @unroll for n in 1:Nq2
                @unroll for fs in 1:nfilterstates
                    l_Qfiltered[fs] += filtermatrix[j, n] * s_Q[i, n, k, fs]
                end
            end

            if filterinξ3
                @synchronize
                @unroll for fs in 1:nfilterstates
                    s_Q[i, j, k, fs] = l_Qfiltered[fs]
                    l_Qfiltered[fs] = zero(FT)
                end
            end
        end

        if filterinξ3
            @synchronize
            @unroll for n in 1:Nq3
                @unroll for fs in 1:nfilterstates
                    l_Qfiltered[fs] += filtermatrix[k, n] * s_Q[i, j, n, fs]
                end
            end
        end

        @unroll for s in 1:nstates
            l_Q2[s] = l_Q[s]
        end

        compute_filter_result!(
            target,
            Vars{vars_Q}(l_Q2),
            Vars{vars_state_filtered(target, FT)}(l_Qfiltered[:]),
            Vars{vars_state_auxiliary}(l_aux[:]),
        )

        # Store result
        ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
        @unroll for s in 1:nstates
            Q[ijk, s, e] = l_Q2[s]
        end

        @synchronize
    end
end

@kernel function kernel_apply_TMAR_filter!(
    ::Val{nreduce},
    ::Val{dim},
    ::Val{N},
    Q,
    target::FilterIndices{I},
    vgeo,
) where {nreduce, dim, N, I}
    @uniform begin
        FT = eltype(Q)

        Nqs = N .+ 1
        Nq1 = Nqs[1]
        Nq2 = dim == 2 ? 1 : Nqs[2]
        Nq3 = Nqs[end]

        nfilterstates = number_state_filtered(target, FT)
        nelemperblock = 1
    end

    l_Q = @private FT (nfilterstates, Nq1)
    l_MJ = @private FT (Nq1,)

    s_MJQ = @localmem FT (Nq1 * Nq2, nfilterstates)
    s_MJQclipped = @localmem FT (Nq1 * Nq2, nfilterstates)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        # loop up the pencil and load Q and MJ
        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            @unroll for sf in 1:nfilterstates
                s = I[sf]
                l_Q[sf, k] = Q[ijk, s, e]
            end

            l_MJ[k] = vgeo[ijk, _M, e]
        end

        @unroll for sf in 1:nfilterstates
            MJQ, MJQclipped = zero(FT), zero(FT)

            @unroll for k in 1:Nq3
                MJ = l_MJ[k]
                Qs = l_Q[sf, k]
                Qsclipped = Qs ≥ 0 ? Qs : zero(Qs)

                MJQ += MJ * Qs
                MJQclipped += MJ * Qsclipped
            end

            ij = i + Nq1 * (j - 1)

            s_MJQ[ij, sf] = MJQ
            s_MJQclipped[ij, sf] = MJQclipped
        end
        @synchronize

        @unroll for n in 11:-1:1
            if nreduce ≥ 2^n
                ij = i + Nq1 * (j - 1)
                ijshift = ij + 2^(n - 1)
                if ij ≤ 2^(n - 1) && ijshift ≤ Nq1 * Nq2
                    @unroll for sf in 1:nfilterstates
                        s_MJQ[ij, sf] += s_MJQ[ijshift, sf]
                        s_MJQclipped[ij, sf] += s_MJQclipped[ijshift, sf]
                    end
                end
                @synchronize
            end
        end

        @unroll for sf in 1:nfilterstates
            qs_average = s_MJQ[1, sf]
            qs_clipped_average = s_MJQclipped[1, sf]

            r = qs_average > 0 ? qs_average / qs_clipped_average : zero(FT)

            s = I[sf]
            @unroll for k in 1:Nq3
                ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

                Qs = l_Q[sf, k]
                Q[ijk, s, e] = Qs ≥ 0 ? r * Qs : zero(Qs)
            end
        end
    end
end

end
