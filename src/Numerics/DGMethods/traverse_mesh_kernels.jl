##### Iterators for traversing the mesh
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

@kernel function kernel_traverse_mesh!(
    ::Pointwise,
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    ::Val{N_states},
    f!,
    elems,
    activedofs,
    ::Type{FT},
    states...,
) where {dim, N, N_states, FT}
    bl = balance_law
    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Np = Nq * Nq * Nqk
    all_states = ntuple(Val(N_states)) do i
        st = states[i][1]
        state = states[i][2]
        (
            st,
            number_states(bl, st),
            vars_state(balance_law, st, FT),
            MArray{Tuple{number_states(bl, st)}, FT}(undef),
            state,
        )
    end
    I = @index(Global, Linear)
    eI = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1
    @inbounds begin
        # e = elems[eI]
        # if activedofs[n + (e - 1) * Np]
        #     @unroll for (st, ns, vs, local_state, state) in all_states
        #         @unroll for s in 1:ns
        #             local_state[s] = state[n, s, e]
        #         end
        #     end
        #     vs_local_state = (
        #         Vars{vs}(local_state) for
        #         (st, ns, vs, local_state, state) in all_states
        #     )
        #     f!(balance_law, vs_local_state...)
        #     @unroll for (st, ns, vs, local_state, state) in all_states
        #         @unroll for s in 1:ns
        #             state[n, s, e] = local_state[s]
        #         end
        #     end
        # end
    end
end
