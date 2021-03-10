"""
    filter_w(w::FT, w_min::FT) where {FT}

Return velocity such that
`abs(filter_w(w, w_min)) >= abs(w_min)`
while preserving the sign of `w`.
"""
filter_w(w::FT, w_min::FT) where {FT} =
    max(abs(w), abs(w_min)) * (w < 0 ? sign(w) : 1)

"""
    enforce_unit_bounds(a_up_i::FT, a_min::FT, a_max::FT) where {FT}

Enforce variable to be positive.

Ideally, this safety net will be removed once we
have robust positivity preserving methods. For now,
we need this to avoid domain error in certain
circumstances.
"""
enforce_unit_bounds(a_up_i::FT, a_min::FT, a_max::FT) where {FT} =
    clamp(a_up_i, a_min, a_max)

"""
    enforce_positivity(x::FT) where {FT}

Enforce variable to be positive.

Ideally, this safety net will be removed once we
have robust positivity preserving methods. For now,
we need this to avoid domain error in certain
circumstances.
"""
enforce_positivity(x::FT) where {FT} = max(x, FT(0))

"""
    fix_void_up(ρa_up_i::FT, val::FT, fallback = FT(0)) where {FT}

Substitute value by a consistent fallback in case of
negligible area fraction (void updraft).
"""
function fix_void_up(ρa_up_i::FT, val::FT, fallback = FT(0)) where {FT}
    tol = sqrt(eps(FT))
    return ρa_up_i > tol ? val : fallback
end
