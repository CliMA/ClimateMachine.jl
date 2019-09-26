#### DomainDecompFuncs
# Provides a set of helper functions that operate on StateVec.

export domain_average!,
       distribute!,
       diagnose_environment!,
       total_covariance!

"""
    domain_average!(sv::StateVec, weight::StateVec, sv_idx, weight_idx, grid::Grid)

Compute domain average of field `sv_idx`, using weights `weight_idx`,
in state vector `sv`, the grid `grid` respectively.

Formulaically, a domain-averaged variable ``⟨ϕ⟩`` is computed from

``⟨ϕ⟩ = Σ_i a_i \\overline{ϕ}_i``

Where variable ``\\overline{ϕ}_i`` represents ``ϕ`` decomposed across multiple
sub-domains, which are weighted by area fractions ``a_i``.

Note that `domain_average!` is the inverse function of `distribute!`.
"""
function domain_average!(sv::StateVec, weight::StateVec, sv_idx, weight_idx, grid::Grid)
  gm, en, ud, sd, al = allcombinations(DomainIdx(sv))
  @inbounds for k in over_elems_real(grid)
    sv[sv_idx, k, gm] = 0
    @inbounds for i in sd
      sv[sv_idx, k, gm] += sv[sv_idx, k, i]*weight[weight_idx, k, i]
    end
  end
end

"""
    distribute!(sv::StateVec, grid::Grid, var_names)

Distributes values in the state vector `sv`, from the grid-mean
domain to the sub-domains.

Formulaically, a domain-decomposed variable ``\\overline{ϕ}_i`` is computed from

``\\overline{ϕ}_i = ⟨ϕ⟩``

Where variable ``⟨ϕ⟩`` is the domain-averaged variable, computed
across multiple sub-domains.

Note that `distribute!` is the inverse function of `domain_average!`.
"""
function distribute!(sv::StateVec, grid::Grid, var_names)
  !(var_names isa Tuple) && (var_names = (var_names,))
  gm, en, ud, sd, al = allcombinations(DomainIdx(sv))
  @inbounds for k in over_elems(grid)
    @inbounds for i in sd
      @inbounds for ϕ in var_names
        sv[ϕ, k, i] = sv[ϕ, k, gm]
      end
    end
  end
end

"""
    diagnose_environment!(sv::StateVec, grid::Grid, names)

Diagnose environment values in the state vector `sv`, from the grid-mean
domain and updraft variables from the decomposition constraint.

Formulaically:

``ϕ_en = (⟨ϕ⟩-Σ_{i=ud} a_i ϕ_i)/a_en``
"""
function diagnose_environment!(sv::StateVec, grid::Grid, weight_idx, var_names)
  !(var_names isa Tuple) && (var_names = (var_names,))
  gm, en, ud, sd, al = allcombinations(DomainIdx(sv))
  @inbounds for k in over_elems(grid)
    sv[weight_idx, k, en] = sv[weight_idx, k, gm] - sum([sv[weight_idx, k, i] for i in ud]...)
    a_en = sv[weight_idx, k, en]
    @inbounds for ϕ in var_names
      sv[ϕ, k, en] = (sv[ϕ, k, gm] - sum([sv[weight_idx, k, i]*sv[ϕ, k, i] for i in ud]...))/a_en
    end
  end
end

"""
    total_covariance!(dst::StateVec, src::StateVec, cv::StateVec, weights::StateVec,
                      dst_idxs, src_idxs, cv_idxs, weight_idx,
                      grid::Grid, decompose_ϕ_ψ::Function)

Computes the total covariance in state vector `dst`, given
 - `src` source state vector
 - `cv` state vector containing co-variances
 - `weights` state vector containing weights
 - `dst_idxs` indexes for destination state vector
 - `cv_idxs` indexes for state vector containing co-variances
 - `weight_idx` indexes for state vector containing weights
 - `grid` the grid
 - `decompose_ϕ_ψ` a function that receives the covariance index and
                   returns the indexes for each variable. For example:
                   `:ϕ_idx, :ψ_idx = decompose_ϕ_ψ(:cv_ϕ_ψ)`

Formulaically, a total covariance between variables ``ϕ`` and ``ψ`` is computed from

``⟨ϕ^*ψ^*⟩ = Σ_i a_i \\overline{ϕ_i'ψ_i'} + Σ_i Σ_j a_i a_j \\overline{ϕ}_i (\\overline{ψ}_i - \\overline{ψ}_j)``

Where variable ``\\overline{ϕ}_i`` represents ``ϕ`` decomposed across multiple
sub-domains, which are weighted by area fractions ``a_i``.
"""
function total_covariance!(tmp::StateVec, sv::StateVec, cv::StateVec,
                           tcv_ϕψ::Symbol, cv_ϕψ::Symbol,
                           a::Symbol, grid::Grid, decompose_ϕ_ψ::Function)
  gm, en, ud, sd, al = allcombinations(DomainIdx(sv))
  @inbounds for k in over_elems(grid)
    _ϕ, _ψ = decompose_ϕ_ψ(tcv_ϕψ)
    tmp[tcv_ϕψ, k] = 0
    @inbounds for i in sd
      tmp[tcv_ϕψ, k] += sv[a, k, i]*cv[cv_ϕψ, k, i]
      @inbounds for j in sd
        tmp[tcv_ϕψ, k] += sv[a, k, i]*sv[a, k, j]*sv[_ϕ, k, i]*(sv[_ψ, k, i] - sv[_ψ, k, j])
      end
    end
  end
end
