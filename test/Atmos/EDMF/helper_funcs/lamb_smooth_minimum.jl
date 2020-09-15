using LambertW

"""
    lambertw_gpu(N)

Returns `real(LambertW.lambertw(Float64(N - 1) / MathConstants.e))`,
valid for `N = 2` and `N = 3`.

TODO: add `LambertW.lambertw` support to KernelAbstractions.
"""
function lambertw_gpu(N)
    if !(N == 2 || N == 3)
        error("Bad N in lambertw_gpu")
    end
    return (0.2784645427610738, 0.46305551336554884)[N - 1]
end

"""
    lamb_smooth_minimum(
        l::AbstractArray{FT};
        frac_upper_bound::FT,
        reg_min::FT,
    ) where {FT}

Returns the smooth minimum of the elements of an array
following the formulation of
Lopez-Gomez et al. (JAMES, 2020), Appendix A, given:
 - `l`, an array of candidate elements
 - `frac_upper_bound`, defines the upper bound of the
        smooth minimum as `smin(x) = min(x)*(1+frac_upper_bound)`
 - `reg_min`, defines the minimum value of the regularizer Λ
"""
function lamb_smooth_minimum(
    l::AbstractArray{FT},
    frac_upper_bound::FT,
    reg_min::FT,
) where {FT}

    xmin = minimum(l)

    # Get regularizer for exponential weights
    N_l = length(l)
    denom = FT(lambertw_gpu(N_l))
    Λ = max(FT(xmin) * frac_upper_bound / denom, reg_min)

    num = sum(i -> l[i] * exp(-(l[i] - xmin) / Λ), 1:N_l)
    den = sum(i -> exp(-(l[i] - xmin) / Λ), 1:N_l)
    smin = num / den

    return smin
end
