#### Custom Filter
using ..Mesh.Filters: AbstractFilter
import ..Mesh.Filters: apply!
export AbstractCustomFilter

"""
    AbstractCustomFilter <: AbstractFilter

Dispatch type for a pointwise custom filter; see [`custom_filter!`](@ref)

!!! warning
    Modifying the prognostic state with this filter
    does not guarantee conservation of the modified
    state.
"""
abstract type AbstractCustomFilter <: AbstractFilter end

"""
    custom_filter!(
        ::AbstractCustomFilter,
        balance_law,
        state,
        aux,
    )

Apply the custom filter `AbstractCustomFilter`,
to the prognostic state for a given balance law.
"""
function custom_filter!(::AbstractCustomFilter, balance_law, state, aux) end

function apply!(
    custom_filter::AbstractCustomFilter,
    grid::DiscontinuousSpectralElementGrid,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    state_auxiliary::MPIStateArray,
) where {F!}
    elems = grid.topology.realelems
    device = array_device(state_prognostic)
    Np = dofs_per_element(grid)
    N = polynomialorders(grid)
    knl_custom_filter! = kernel_custom_filter!(device, min(Np, 1024))
    event = Event(device)
    event = knl_custom_filter!(
        m,
        Val(dimensionality(grid)),
        Val(N),
        custom_filter,
        state_prognostic.data,
        state_auxiliary.data,
        elems,
        grid.activedofs;
        ndrange = Np * length(elems),
        dependencies = (event,),
    )
    wait(device, event)
end

@kernel function kernel_custom_filter!(
    bl::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    custom_filter,
    state_prognostic,
    state_auxiliary,
    elems,
    activedofs,
) where {dim, N}
    @uniform begin
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(bl, Prognostic())
        num_state_auxiliary = number_states(bl, Auxiliary())
        vs_p = Vars{vars_state(bl, Prognostic(), FT)}
        vs_a = Vars{vars_state(bl, Auxiliary(), FT)}
        Nq = N .+ 1
        Nqk = dim == 2 ? 1 : Nq[dim]
        Np = Nq[1] * Nq[2] * Nqk

        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
    end

    I = @index(Global, Linear)
    eI = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        e = elems[eI]

        active = activedofs[n + (e - 1) * Np]

        if active
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[n, s, e]
            end

            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[n, s, e]
            end

            custom_filter!(
                custom_filter,
                bl,
                vs_p(local_state_prognostic),
                vs_a(local_state_auxiliary),
            )

            @unroll for s in 1:num_state_prognostic
                state_prognostic[n, s, e] = local_state_prognostic[s]
            end
        end
    end
end
