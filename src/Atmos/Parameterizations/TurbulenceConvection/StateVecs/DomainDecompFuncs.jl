#### DomainDecompFuncs
# Provides a set of helper functions that operate on StateVec.

export grid_mean!,
       distribute!,
       diagnose_environment!,
       total_covariance!

"""
    grid_mean!(sv::StateVec, sv_a::StateVec, ϕ, a, grid::Grid)

Compute domain average of field(s) `ϕ`, using weights `a`,
in state vector `sv`, the grid `grid` respectively.

A domain-averaged variable ``⟨ϕ⟩`` is computed from

``⟨ϕ⟩ = Σ_i a_i \\overline{ϕ}_i``

Where variable ``\\overline{ϕ}_i`` represents ``ϕ`` decomposed across multiple
sub-domains, which are weighted by area fractions ``a_i``.

Note that `grid_mean!` is the inverse function of `distribute!`.
"""
function grid_mean!(sv::StateVec, sv_a::StateVec, a, var_names, grid::Grid)
  gm, en, ud, sd, al = allcombinations(sv)
  !(var_names isa Tuple) && (var_names = (var_names,))
  @inbounds for k in over_elems_real(grid)
    for ϕ in var_names
      sv[ϕ, k, gm] = 0
      @inbounds for i in sd
        sv[ϕ, k, gm] += sv[ϕ, k, i]*sv_a[a, k, i]
      end
    end
  end
  extrap_0th_order!(sv, var_names, grid, gm)
end

"""
    distribute!(sv::StateVec, grid::Grid, ϕ)

Distributes values in the state vector `sv`, from the grid-mean
domain to the sub-domains.

A domain-decomposed variable ``\\overline{ϕ}_i`` is computed from

``\\overline{ϕ}_i = ⟨ϕ⟩``

Where variable ``⟨ϕ⟩`` is the grid-mean variable, computed
across multiple sub-domains.

Note that `distribute!` is the inverse function of `grid_mean!`.
"""
function distribute!(sv::StateVec, grid::Grid, var_names)
  !(var_names isa Tuple) && (var_names = (var_names,))
  gm, en, ud, sd, al = allcombinations(sv)
  @inbounds for k in over_elems_real(grid)
    @inbounds for i in sd
      @inbounds for ϕ in var_names
        sv[ϕ, k, i] = sv[ϕ, k, gm]
      end
    end
  end
  @inbounds for i in sd
    extrap_0th_order!(sv, var_names, grid, i)
  end
end

"""
    diagnose_environment!(sv::StateVec, grid::Grid, names)

Diagnose environment values in the state vector `sv`, from the grid-mean
domain and updraft variables from the decomposition constraint.

Formulaically:

``ϕ_en = (⟨ϕ⟩-Σ_{i=ud} a_i ϕ_i)/a_en``
"""
function diagnose_environment!(sv::StateVec, grid::Grid, a, var_names)
  !(var_names isa Tuple) && (var_names = (var_names,))
  gm, en, ud, sd, al = allcombinations(sv)
  @inbounds for k in over_elems(grid)
    sv[a, k, en] = sv[a, k, gm] - sum([sv[a, k, i] for i in ud]...)
    a_en = sv[a, k, en]
    @inbounds for ϕ in var_names
      sv[ϕ, k, en] = (sv[ϕ, k, gm] - sum([sv[a, k, i]*sv[ϕ, k, i] for i in ud]...))/a_en
    end
  end
  extrap_0th_order!(sv, var_names, grid, en)
end

"""
    total_covariance!(dst::StateVec, src::StateVec, cv::StateVec, weights::StateVec,
                      dst_idxs, src_idxs, cv_idxs, a,
                      grid::Grid, decompose_ϕ_ψ::Function)

Computes the total covariance in state vector `dst`, given
 - `src` source state vector
 - `cv` state vector containing co-variances
 - `weights` state vector containing weights
 - `dst_idxs` indexes for destination state vector
 - `cv_idxs` indexes for state vector containing co-variances
 - `a` indexes for state vector containing weights
 - `grid` the grid
 - `decompose_ϕ_ψ` a function that receives the covariance index and
                   returns the indexes for each variable. For example:
                   `:ϕ_idx, :ψ_idx = decompose_ϕ_ψ(:cv_ϕ_ψ)`

A total covariance between variables ``ϕ`` and ``ψ`` is computed from

``⟨ϕ^*ψ^*⟩ = Σ_i a_i \\overline{ϕ_i'ψ_i'} + Σ_i Σ_j a_i a_j \\overline{ϕ}_i (\\overline{ψ}_i - \\overline{ψ}_j)``

Where variable ``\\overline{ϕ}_i`` represents ``ϕ`` decomposed across multiple
sub-domains, which are weighted by area fractions ``a_i``.
"""
function total_covariance!(tmp::StateVec, sv::StateVec, cv::StateVec,
                           tcv_ϕψ::Symbol, cv_ϕψ::Symbol,
                           a::Symbol, grid::Grid, decompose_ϕ_ψ::Function)
  gm, en, ud, sd, al = allcombinations(sv)
  @inbounds for k in over_elems(grid)
    _ϕ, _ψ = decompose_ϕ_ψ(tcv_ϕψ)
    tmp[tcv_ϕψ, k, gm] = 0
    @inbounds for i in sd
      tmp[tcv_ϕψ, k, gm] += sv[a, k, i]*cv[cv_ϕψ, k, i]
      @inbounds for j in sd
        tmp[tcv_ϕψ, k, gm] += sv[a, k, i]*sv[a, k, j]*sv[_ϕ, k, i]*(sv[_ψ, k, i] - sv[_ψ, k, j])
      end
    end
  end
  extrap_0th_order!(tmp, tcv_ϕψ, grid, gm)
end
