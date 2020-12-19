module FVReconstructions
using KernelAbstractions.Extras: @unroll
import StaticArrays: SUnitRange, SVector

"""
    AbstractReconstruction

Supertype for FV reconstructions.

Concrete types must provide implementions of
    - `width(recon)`
       returns the width of the reconstruction. Total number of points used in
       reconstruction of top and bottom states is `2width(recon) + 1`
       - (::AbstractReconstruction)(state_top, state_bottom, cell_states::SVector,
                                    cell_weights)
      compute the reconstruction

```
(::AbstractReconstruction)(
        state_top,
        state_bottom,
        cell_states,
        cell_weights,
    )
```

Perform the finite volume reconstruction for the top and bottom states using the
tuple of `cell_states` values using the `cell_weights`.
"""
abstract type AbstractReconstruction end
function (::AbstractReconstruction) end

"""
    AbstractSlopeLimiter

Supertype for FV slope limiter

Given two values `Δ1` and `Δ2`, should return the value of
```
Δ2 * ϕ(Δ1 / Δ2)
```
where `0 ≤ ϕ(r) ≤ 2` is the slope slope limiter
"""
abstract type AbstractSlopeLimiter end

"""
    width(recon::AbstractReconstruction)

Returns the width of the stencil need for the FV reconstruction `recon`. Total
number of values used in the reconstruction are `2width(recon) + 1`
"""
width(recon::AbstractReconstruction) = throw(MethodError(width, (recon,)))

"""
    FVConstant <: AbstractReconstruction

Reconstruction type for cell centered finite volume methods (e.g., constants)
"""
struct FVConstant <: AbstractReconstruction end

width(::FVConstant) = 0

function (::FVConstant)(state_bot, state_top, cell_states::SVector{1}, _)
    @inbounds state_top .= cell_states[1]
    @inbounds state_bot .= cell_states[1]
end
"""
    FVLinear{W = 1} <: AbstractReconstruction

Reconstruction type for limited linear reconstruction finite volume methods.

!!! note
   The optional type parameter `W` is mainly for debuggin purposes and allows
   the stencil to be artificially widened to make sure the kernels work with
   wide stencils.

    FVLinear(limiter = VanLeer())
    FVLinear{W}(limiter = VanLeer())

Construct the `FVLinear` reconstruction type with the given slope `limiter`
"""
struct FVLinear{W, L} <: AbstractReconstruction
    limiter::L

end

"""
    FVLinear(limiter = VanLeer())
    FVLinear{W}(limiter = VanLeer())

Construct the `FVLinear` reconstruction type with the given slope `limiter` and
optional width `W`.
"""
function FVLinear{W}(limiter = VanLeer()) where {W}
    @assert W > 0
    FVLinear{W, typeof(limiter)}(limiter)
end
FVLinear(limiter = VanLeer()) = FVLinear{1}(limiter)

width(::FVLinear{W}) where {W} = W

function (fvrecon::FVLinear)(
    state_bot,
    state_top,
    cell_states::SVector{3},
    cell_weights,
)
    @inbounds wi_top = 1 / (cell_weights[3] + cell_weights[2])
    @inbounds wi_bot = 1 / (cell_weights[2] + cell_weights[1])
    @inbounds @unroll for s in 1:length(state_top)
        # Compute the edge gradient approximations
        Δ_top = wi_top * (cell_states[3][s] - cell_states[2][s])
        Δ_bot = wi_bot * (cell_states[2][s] - cell_states[1][s])

        # Compute the limited slope
        Δ = fvrecon.limiter(Δ_top, Δ_bot)

        # Compute the reconstructions at the cell edges
        state_top[s] = cell_states[2][s] + Δ * cell_weights[2]
        state_bot[s] = cell_states[2][s] - Δ * cell_weights[2]
    end
end

function (fvrecon::FVLinear)(
    state_bot,
    state_top,
    cell_states::SVector{1},
    cell_weights,
)
    FVConstant()(state_bot, state_top, cell_states, cell_weights)
end

function (fvrecon::FVLinear)(
    state_bot,
    state_top,
    cell_states::SVector{D},
    cell_weights,
) where {D}
    W = div(D - 1, 2)
    rng = SUnitRange(W, W + 2)
    @inbounds fvrecon(state_bot, state_top, cell_states[rng], cell_weights[rng])
end

"""
    VanLeer <: AbstractSlopeLimiter

Classic Van-Leer limiter
```
   ϕ(r) = (r + |r|) / (1 + r)
```

### References
    @article{van1974towards,
      title = {Towards the ultimate conservative difference scheme. II.
               Monotonicity and conservation combined in a second-order scheme},
      author = {Van Leer, Bram},
      journal = {Journal of computational physics},
      volume = {14},
      number = {4},
      pages = {361--370},
      year = {1974},
      doi = {10.1016/0021-9991(74)90019-9}
    }
"""
struct VanLeer <: AbstractSlopeLimiter end

function (::VanLeer)(Δ_top, Δ_bot)
    FT = eltype(Δ_top)
    if Δ_top * Δ_bot > 0
        return 2Δ_top * Δ_bot / (Δ_top + Δ_bot)
    else
        return FT(0)
    end
end

end
