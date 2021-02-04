using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using ClimateMachine.MPIStateArrays: array_device

pbl_diagnostic_vars(FT) = @vars(θ::FT, u::FT, w::FT, ρ::FT)
function pbl_diagnostics!(atmos, diag::Vars, state::Vars, aux::Vars)
  FT = eltype(state)
  param_set = atmos.param_set
  e_int = internal_energy(atmos, state, aux)
  ts = PhaseDry(param_set, e_int, state.ρ)
  _MSLP::FT = MSLP(param_set)
  _R_d::FT = R_d(param_set)
  _cp_d::FT = cp_d(param_set)
  p = air_pressure(ts)
  T = air_temperature(ts)
  diag.θ = T * (_MSLP / p) ^ (_R_d / _cp_d)
  diag.u = state.ρu[2] / state.ρ
  diag.w = state.ρu[3] / state.ρ
  diag.ρ = state.ρ
end

function nodal_diagnostics!(diagnostic_fun!, diagnostic_vars, 
                            dg, state_diagnostic, state_prognostic)
  balance_law = dg.balance_law
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

  event = Event(device)
  event = kernel_nodal_diagnostics!(device, min(Np, 1024))(
      balance_law,
      Val(dim),
      Val(N),
      Val(diagnostic_vars),
      diagnostic_fun!,
      state_diagnostic.data,
      state_prognostic.data,
      state_auxiliary.data,
      realelems,
      ndrange = Np * length(realelems),
      dependencies = event
  )
  wait(event)
end
@kernel function kernel_nodal_diagnostics!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    ::Val{diagnostic_vars},
    diagnostic_fun!,
    state_diagnostic,
    state_prognostic,
    state_auxiliary,
    elems,
) where {dim, N, diagnostic_vars}
    FT = eltype(state_prognostic)
    num_state_prognostic = number_states(balance_law, Prognostic())
    num_state_auxiliary = number_states(balance_law, Auxiliary())
    num_state_diagnostic = varsize(diagnostic_vars)

    Nq = N + 1

    Nqk = dim == 2 ? 1 : Nq

    Np = Nq * Nq * Nqk

    local_state_diagnostic = MArray{Tuple{num_state_diagnostic}, FT}(undef)
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

        diagnostic_fun!(
            balance_law,
            Vars{diagnostic_vars}(
                local_state_diagnostic,
            ),
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                local_state_prognostic,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                local_state_auxiliary,
            ),
        )
       
        @unroll for s in 1:num_state_diagnostic
          state_diagnostic[n, s, e] = local_state_diagnostic[s]
        end
    end
end

function profiles(diagnostic_vars, dg, state_diagnostic)
    # TODO: GPU, MPI
    state_diagnostic = Array(state_diagnostic.data)

    FT = eltype(state_diagnostic)

    num_state_diagnostic = varsize(diagnostic_vars)

    grid = dg.grid
    topology = grid.topology
    nrealelem = length(topology.realelems)
    Nq = polynomialorders(grid)[1] + 1
    Nqk = Nq
    vgeo = grid.vgeo
    localvgeo = Array(vgeo)

    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    scaling = zeros(FT, Nqk, nvertelem) 
    profs = zeros(FT, Nqk, nvertelem, num_state_diagnostic)

    for ev in 1:nvertelem
        for eh in 1:nhorzelem
            e = ev + (eh - 1) * nvertelem
            for i in 1:Nq
                for j in 1:Nq
                    for k in 1:Nqk
                        ijk = i + Nq * ((j - 1) + Nqk * (k - 1))
                        scaling[k, ev] += localvgeo[ijk, grid.MHid, e]
                        for var in fieldnames(diagnostic_vars)
                          s = varsindex(diagnostic_vars, var)
                          profs[k, ev, s] += state_diagnostic[ijk, s, e] * localvgeo[ijk, grid.MHid, e]
                        end
                    end
                end
            end
        end
    end
    
    for ev in 1:nvertelem
      for k in 1:Nqk
        for s in 1:num_state_diagnostic
          profs[k, ev, s] /= scaling[k, ev]
        end
      end
    end

    (; zip(fieldnames(diagnostic_vars),
           (profs[:, :, s] for s in 1:num_state_diagnostic))...)
end
