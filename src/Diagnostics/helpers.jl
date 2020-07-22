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
function extract_state(dg, state, ijk, e, st::AbstractStateType)
    bl = dg.balance_law
    FT = eltype(state)
    num_state = number_states(bl, st, FT)
    local_state = MArray{Tuple{num_state}, FT}(undef)
    for s in 1:num_state
        local_state[s] = state[ijk, s, e]
    end
    return Vars{vars_state(bl, st, FT)}(local_state)
end
