using KernelAbstractions
using ClimateMachine.MPIStateArrays: array_device, weightedsum
using KernelAbstractions.Extras: @unroll

function entropy_integral(dg, balance_law, entropy, state_prognostic)
  state_auxiliary = dg.state_auxiliary
  device = array_device(state_prognostic)
  grid = dg.grid
  topology = grid.topology
  Np = dofs_per_element(grid)
  dim = dimensionality(grid)
  # XXX: Needs updating for multiple polynomial orders
  N = polynomialorders(grid)
  # Currently only support single polynomial order
  @assert all(N[1] .== N)
  N = N[1]
  
  realelems = topology.realelems

  event = esdg_compute_entropy!(device, min(Np, 1024))(
      balance_law,
      Val(dim),
      Val(N),
      entropy.data,
      state_prognostic.data,
      state_auxiliary.data,
      realelems,
      ndrange = Np * length(realelems)
  )
  wait(event)
  
  weightedsum(entropy)
end

@kernel function esdg_compute_entropy!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    entropy,
    state_prognostic,
    state_auxiliary,
    elems,
) where {dim, N}

    FT = eltype(state_prognostic)
    num_state_prognostic = number_states(balance_law, Prognostic())
    num_state_auxiliary = number_states(balance_law, Auxiliary())

    Nq = N + 1

    Nqk = dim == 2 ? 1 : Nq

    Np = Nq * Nq * Nqk

    local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

    I = @index(Global, Linear)
    eI = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        e = elems[eI]
        @unroll for s in 1:num_state_prognostic
            local_state_prognostic[s] = state_prognostic[n, s, e]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end

        entropy[n, 1, e] = state_to_entropy(
            balance_law,
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                local_state_prognostic,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                local_state_auxiliary,
            ),
        )
    end
end
