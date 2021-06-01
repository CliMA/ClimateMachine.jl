using KernelAbstractions
using ClimateMachine.MPIStateArrays: array_device, weightedsum
using KernelAbstractions.Extras: @unroll
using ClimateMachine.Mesh.Elements: interpolationmatrix, lglpoints
using ClimateMachine.Mesh.Grids: _x1, _x3

function entropy_integral(dg, entropy, state_prognostic)
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
  event = esdg_compute_entropy!(device, min(Np, 1024))(
      balance_law,
      Val(dim),
      Val(N),
      entropy.data,
      state_prognostic.data,
      state_auxiliary.data,
      realelems,
      ndrange = Np * length(realelems),
      dependencies = event
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

function entropy_product(dg, entropy, state_prognostic, tendency)
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
  event = esdg_compute_entropy_product!(device, min(Np, 1024))(
      balance_law,
      Val(dim),
      Val(N),
      entropy.data,
      state_prognostic.data,
      tendency.data,
      state_auxiliary.data,
      realelems,
      ndrange = Np * length(realelems),
      dependencies = event
  )
  wait(event)
  
  weightedsum(entropy)
end

@kernel function esdg_compute_entropy_product!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    entropy,
    state_prognostic,
    tendency,
    state_auxiliary,
    elems,
) where {dim, N}

    FT = eltype(state_prognostic)
    num_state_prognostic = number_states(balance_law, Prognostic())
    num_state_entropy = number_states(balance_law, Entropy())
    num_state_auxiliary = number_states(balance_law, Auxiliary())

    Nq = N + 1

    Nqk = dim == 2 ? 1 : Nq

    Np = Nq * Nq * Nqk

    local_state_entropy = MArray{Tuple{num_state_entropy}, FT}(undef)
    local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
    local_tendency = MArray{Tuple{num_state_prognostic}, FT}(undef)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

    I = @index(Global, Linear)
    eI = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        e = elems[eI]
        
        @unroll for s in 1:num_state_prognostic
            local_state_prognostic[s] = state_prognostic[n, s, e]
        end
        
        @unroll for s in 1:num_state_prognostic
            local_tendency[s] = tendency[n, s, e]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end

        state_to_entropy_variables!(
            balance_law,
            Vars{vars_state(balance_law, Entropy(), FT)}(
                local_state_entropy,
            ),
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                local_state_prognostic,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                local_state_auxiliary,
            ),
        )
       
        local_product = -zero(FT)
        # not that tendency related to the last entropy variable is assumed zero
        @unroll for s in 1:num_state_prognostic
          local_product += local_state_entropy[s] * local_tendency[s]
        end
        entropy[n, 1, e] = local_product
    end
end

function nodal_diagnostics(diagnostic_fun!, diagnostic_vars,
                           model, state_prognostic, state_auxiliary, vgeo)
  FT = eltype(state_prognostic)
  diagnostic_vars = diagnostic_vars(FT)
  num_state_diagnostic = varsize(diagnostic_vars)
  Np = size(state_prognostic, 1)
  Ne = size(state_prognostic, 3)
  state_diagnostic = similar(state_prognostic,
                             (Np, num_state_diagnostic, Ne))

  num_state_prognostic = number_states(model, Prognostic())
  num_state_auxiliary = number_states(model, Auxiliary())

  local_state_diagnostic = MArray{Tuple{num_state_diagnostic}, FT}(undef)
  local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
  local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

  @inbounds @views for e in 1:Ne
    for ijk in 1:Np
       local_state_prognostic .= state_prognostic[ijk, :, e]
       local_state_auxiliary .= state_auxiliary[ijk, :, e]
       local_coord = SVector{3, FT}(vgeo[ijk, _x1:_x3, e])
       diagnostic_fun!(
           model,
           Vars{diagnostic_vars}(
               local_state_diagnostic,
           ),
           Vars{vars_state(model, Prognostic(), FT)}(
               local_state_prognostic,
           ),
           Vars{vars_state(model, Auxiliary(), FT)}(
               local_state_auxiliary,
           ),
           local_coord
       )
       state_diagnostic[ijk, :, e] .= local_state_diagnostic
    end
  end
  
  state_diagnostic
end

function rcParams!(rcParams)
  rcParams["font.size"] = 20
  rcParams["xtick.labelsize"] = 20
  rcParams["ytick.labelsize"] = 20
  rcParams["legend.fontsize"] = 20
  rcParams["figure.titlesize"] = 32
  rcParams["axes.titlepad"] = 10
  rcParams["axes.labelpad"] = 10
end

function interpolate_equidistant(state_diagnostic, vgeo, dim, N, K)
    FT = eltype(state_diagnostic)
    Np = size(state_diagnostic, 1)
    Ns = size(state_diagnostic, 2)
    Ne = size(state_diagnostic, 3)

    ξsrc, _ = lglpoints(FT, N)
    
    Nqi = 4 * (N + 1)
    Npi = Nqi ^ dim
    dξi = 2 / Nqi
    ξdst = [-1 + (j - 1 / 2) * dξi for j in 1:Nqi]

    I1d = interpolationmatrix(ξsrc, ξdst)
    I = kron(ntuple(_->I1d, dim)...)
   
    state_diagnostic_i = ntuple(_->Array{FT}(undef, Nqi .* K), Ns)
    x_i = ntuple(_->Array{FT}(undef, Nqi .* K), dim)

    @views for e in 1:Ne
      xe_i = ntuple(d -> I * vgeo[:, _x1 + d - 1, e], dim)
      dx_i = ntuple(dim) do d
        xd_i = reshape(xe_i[d], (Nqi for dd in 1:dim)...)
        C0 = CartesianIndex((1 for dd in 1:dim)...)
        Cd = CartesianIndex((d == dd for dd in 1:dim)...)
        xd_i[C0 + Cd] - xd_i[C0]
      end
      de_i = ntuple(s -> I * state_diagnostic[:, s, e], Ns)
      ie_i = ntuple(d -> round.(Int, (xe_i[d] .+ dx_i[d] / 2) ./ dx_i[d]), dim)
      C = CartesianIndex.(ie_i...)
      for ijk in 1:Npi
        for s in 1:Ns
          state_diagnostic_i[s][C[ijk]] = de_i[s][ijk]
        end
        for d in 1:dim
          x_i[d][C[ijk]] = xe_i[d][ijk]
        end
      end
    end
    (x_i..., state_diagnostic_i...)
end

function interpolate_horz(state_diagnostic, vgeo, dim, N, K; Nqi)
    FT = eltype(state_diagnostic)
    Np = size(state_diagnostic, 1)
    Ns = size(state_diagnostic, 2)
    Ne = size(state_diagnostic, 3)

    ξsrc, _ = lglpoints(FT, N)
    Nq = N + 1
    
    Npi = Nq * Nqi ^ 2
    dξi = 2 / Nqi
    ξdst_h = [-1 + (j - 1 / 2) * dξi for j in 1:Nqi]
    ξdst_v = range(-FT(1), stop=FT(1), length = Nqi)

    I1d_h = interpolationmatrix(ξsrc, ξdst_h)
    I1d_v = interpolationmatrix(ξsrc, ξdst_v)

    I = kron(LinearAlgebra.I(Nq), I1d_h, I1d_h)
   
    state_diagnostic_i = Array{FT}(undef, Npi, Ns, Ne)

    @views for e in 1:Ne
      for s in 1:Ns
        state_diagnostic_i[:, s, e] .= I * state_diagnostic[:, s, e]
      end
    end
    state_diagnostic_i
end
