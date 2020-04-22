# Miscellaneous helper macros and functions
#

# Helper macro to iterate over the DG grid. Generates the needed loops
# and indices: `eh`, `ev`, `e`, `k,`, `j`, `i`, `ijk`.
macro visitQ(nhorzelem, nvertelem, Nqk, Nq, expr)
    return esc(
        quote
            for eh in 1:($nhorzelem)
                for ev in 1:($nvertelem)
                    e = ev + (eh - 1) * $nvertelem
                    for k in 1:($Nqk)
                        for j in 1:($Nq)
                            for i in 1:($Nq)
                                ijk = i + $Nq * ((j - 1) + $Nq * (k - 1))
                                $expr
                            end
                        end
                    end
                end
            end
        end,
    )
end

# Helper macro to iterate over a 3D array. Used for walking the
# interpolated grid.
macro visitI(nlong, nlat, nlevel, expr)
    return esc(quote
        for lo in 1:($nlong)
            for la in 1:($nlat)
                for le in 1:($nlevel)
                    $expr
                end
            end
        end
    end)
end

# Helpers to extract data from the various state arrays
function extract_state_conservative(dg, state_conservative, ijk, e)
    bl = dg.balance_law
    FT = eltype(state_conservative)
    num_state_conservative = number_state_conservative(bl, FT)
    local_state_conservative = MArray{Tuple{num_state_conservative}, FT}(undef)
    for s in 1:num_state_conservative
        local_state_conservative[s] = state_conservative[ijk, s, e]
    end
    return Vars{vars_state_conservative(bl, FT)}(local_state_conservative)
end
function extract_state_auxiliary(dg, state_auxiliary, ijk, e)
    bl = dg.balance_law
    FT = eltype(state_auxiliary)
    num_state_auxiliary = number_state_auxiliary(bl, FT)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
    for s in 1:num_state_auxiliary
        local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
    end
    return Vars{vars_state_auxiliary(bl, FT)}(local_state_auxiliary)
end
function extract_state_gradient_flux(dg, state_gradient_flux, ijk, e)
    bl = dg.balance_law
    FT = eltype(state_gradient_flux)
    num_state_gradient_flux = number_state_gradient_flux(bl, FT)
    local_state_gradient_flux =
        MArray{Tuple{num_state_gradient_flux}, FT}(undef)
    for s in 1:num_state_gradient_flux
        local_state_gradient_flux[s] = state_gradient_flux[ijk, s, e]
    end
    return Vars{vars_state_gradient_flux(bl, FT)}(local_state_gradient_flux)
end
