using ClimateMachine.Mesh.Filters
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
import ClimateMachine.Mesh.Filters: apply_async!
import ClimateMachine.Mesh.Filters: AbstractFilterTarget
import ClimateMachine.Mesh.Filters: number_state_filtered, vars_state_filtered, compute_filter_argument!, compute_filter_result!

function modified_filter_matrix(r, Nc, σ)
    N = length(r) - 1
    T = eltype(r)

    @assert N >= 0
    @assert 0 <= Nc 

    a, b = GaussQuadrature.legendre_coefs(T, N)
    V = (N == 0 ? ones(T, 1, 1) : GaussQuadrature.orthonormal_poly(r, a, b))

    Σ = ones(T, N + 1) 
    if Nc ≤ N
        Σ[(Nc:N) .+ 1] .= σ.(((Nc:N) .- Nc) ./ (N - Nc))
    end

    V * Diagonal(Σ) / V
end


abstract type AbstractMassPreservingSpectralFilter <: AbstractSpectralFilter end
"""
    CutoffFilter(grid, Nc=polynomialorders(grid))
Returns the spectral filter that zeros out polynomial modes greater than or
equal to `Nc`.
"""
struct MassPreservingCutoffFilter{FM} <: AbstractMassPreservingSpectralFilter
    "filter matrices in all directions (tuple of filter matrices)"
    filter_matrices::FM

    function MassPreservingCutoffFilter(grid, Nc = polynomialorders(grid))
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
        @assert all(0 .<= Nc )

        σ(η) = 0

        AT = arraytype(grid)
        ξ = referencepoints(grid)
        filter_matrices =
            ntuple(i -> AT(modified_filter_matrix(ξ[i], Nc[i], σ)), dim)
        new{typeof(filter_matrices)}(filter_matrices)
    end
end


function apply_async!(
    Q,
    target::AbstractFilterTarget,
    grid::DiscontinuousSpectralElementGrid,
    filter::MassPreservingCutoffFilter;
    dependencies,
    state_auxiliary = nothing,
    direction = EveryDirection(),
)
    topology = grid.topology

    device = typeof(Q.data) <: Array ? CPU() : CUDADevice()

    dim = dimensionality(grid)
    N = polynomialorders(grid)
    # Currently only support same polynomial in both horizontal directions
    @assert dim == 2 || N[1] == N[2]
    Nq = N .+ 1
    Nq1 = Nq[1]
    Nq2 = Nq[2]
    Nq3 = dim == 2 ? 1 : Nq[dim]

    nrealelem = length(topology.realelems)
    # parallel sum info
    nreduce = 2^ceil(Int, log2(Nq1 * Nq2 * Nq3)) 
    event = dependencies

    if direction isa EveryDirection || direction isa HorizontalDirection
        @assert dim == 2 || Nq1 == Nq2
        filtermatrix = filter.filter_matrices[1]
        event = kernel_apply_mp_filter!(device, (Nq1, Nq2, Nq3))(
            Val(nreduce),
            Val(dim),
            Val(N),
            Val(vars(Q)),
            Val(isnothing(state_auxiliary) ? nothing : vars(state_auxiliary)),
            HorizontalDirection(),
            Q.data,
            isnothing(state_auxiliary) ? nothing : state_auxiliary.data,
            target,
            filtermatrix,
            grid.vgeo,
            ndrange = (nrealelem * Nq1, Nq2, Nq3),
            dependencies = event,
        )
    end
    if direction isa EveryDirection || direction isa VerticalDirection
        filtermatrix = filter.filter_matrices[end]
        event = kernel_apply_mp_filter!(device, (Nq1, Nq2, Nq3))(
            Val(nreduce),
            Val(dim),
            Val(N),
            Val(vars(Q)),
            Val(isnothing(state_auxiliary) ? nothing : vars(state_auxiliary)),
            VerticalDirection(),
            Q.data,
            isnothing(state_auxiliary) ? nothing : state_auxiliary.data,
            target,
            filtermatrix,
            grid.vgeo,
            ndrange = (nrealelem * Nq1, Nq2, Nq3),
            dependencies = event,
        )
    end
    return event
end


const _M = Grids._M
# mp for masspreserving
"""
    kernel_apply_mp_filter!(::Val{dim}, ::Val{N}, direction,
                         Q, state_auxiliary, target, filtermatrix
                        ) where {dim, N}
Computational kernel: Applies the `filtermatrix` to `Q` given a
custom target `target`.
The `direction` argument is used to control if the filter is applied in the
""" 
@kernel function kernel_apply_mp_filter!(
    ::Val{nreduce},
    ::Val{dim},
    ::Val{N},
    ::Val{vars_Q},
    ::Val{vars_state_auxiliary},
    direction,
    Q,
    state_auxiliary,
    target::AbstractFilterTarget,
    filtermatrix,
    vgeo,
) where {nreduce, dim, N, vars_Q, vars_state_auxiliary}
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

    s_Q = @localmem FT (Nq1, Nq2, Nq3, nfilterstates) # element local 
    s_MQᴮ = @localmem FT (Nq1 * Nq2 * Nq3, nstates) # before applying filter
    s_MQᴬ = @localmem FT (Nq1 * Nq2 * Nq3, nstates) # after applying filter
    s_M  = @localmem FT (Nq1 * Nq2 * Nq3) # local mass matrix

    l_Q = @private FT (nstates,)
    l_Qfiltered = @private FT (nfilterstates,) # scratch space for storing mat mul
    l_aux = @private FT (nfilteraux,)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)
    ijk = @index(Local, Linear)

    @inbounds begin

        @unroll for s in 1:nstates
            l_Q[s] = Q[ijk, s, e]
        end

        @unroll for s in 1:nfilteraux
            l_aux[s] = state_auxiliary[ijk, s, e]
        end

         # Load mass weighted quantities to shared memory
         s_M[ijk] = vgeo[ijk, _M, e]
         @unroll for s in 1:nstates
             s_MQᴮ[ijk, s] = s_M[ijk] * l_Q[s]
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
        @unroll for s in 1:nstates
            l_Q[s] = l_Q2[s]
            s_MQᴬ[ijk, s] = s_M[ijk] * l_Q[s]
        end

        @synchronize
        @unroll for n in 11:-1:1
            if nreduce ≥ (1 << n)     
                ijkshift = ijk + (1 << (n - 1))
                if ijk ≤ (1 << (n - 1)) && ijkshift ≤ Nq1 * Nq2 * Nq3
                    s_M[ijk] += s_M[ijkshift]
                    @unroll for s in 1:nstates
                        s_MQᴮ[ijk, s] += s_MQᴮ[ijkshift, s]
                        s_MQᴬ[ijk, s] += s_MQᴬ[ijkshift, s]
                    end
                end
                @synchronize
            end
        end

        @synchronize
        M⁻¹ = FT(1) / s_M[1]
        @unroll for s in 1:nstates
            Q[ijk, s, e] = l_Q[s] + M⁻¹ * (s_MQᴮ[1, s] - s_MQᴬ[1, s])
        end

        @synchronize
    end

end
