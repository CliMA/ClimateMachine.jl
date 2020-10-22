##### Iterators for traversing the mesh

@kernel function kernel_traverse_mesh!(
    ::Pointwise,
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    f!,
    elems,
    activedofs,
    ::Type{FT},
    states...,
) where {dim, N, FT}
    bl = balance_law
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Np = Nq * Nq * Nqk
    all_states = (
        (
            st,
            number_states(bl, st),
            vars_state(balance_law, st, FT),
            MArray{Tuple{number_states(bl, st)}, FT}(undef),
            state,
        ) for (st, state) in states
    )

    # @show all_states
    I = @index(Global, Linear)
    eI = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1
    @inbounds begin
        e = elems[eI]
        if activedofs[n + (e - 1) * Np]
            println("-------------------")
            # @unroll for (st, ns, vs, local_state, state) in all_states
            #     @unroll for s in 1:ns
            #         local_state[s] = state[n, s, e]
            #     end
            #     @show st, local_state
            # end
            # vs_local_state = (
            #     Vars{vs}(local_state) for
            #     (st, ns, vs, local_state, state) in all_states
            # )
            # # @show vs_local_state
            # f!(balance_law, vs_local_state...)
            @unroll for (st, ns, vs, local_state, state) in all_states
                @unroll for s in 1:ns
                    state[n, s, e] = 1
                end
            end
        end
    end
end
