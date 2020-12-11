# Miscellaneous helper macros and functions
#

# Helper macro to iterate over the DG grid. Generates the needed loops
# and indices: `eh`, `ev`, `e`, `k,`, `j`, `i`, `ijk`.
macro traverse_dg_grid(grid_info, topl_info, expr)
    return esc(
        quote
            for eh in 1:($(topl_info).nhorzrealelem)
                for ev in 1:($(topl_info).nvertelem)
                    e = ev + (eh - 1) * $(topl_info).nvertelem
                    for k in 1:($(grid_info).Nqk)
                        evk = $(grid_info).Nqk * (ev - 1) + k
                        for j in 1:$(grid_info).Nq[2]
                            for i in 1:$(grid_info).Nq[1]
                                ijk =
                                    i +
                                    $(grid_info).Nq[1] *
                                    ((j - 1) + $(grid_info).Nq[2] * (k - 1))
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
# interpolated (GCM) grid.
macro traverse_interpolated_grid(nlong, nlat, nlevel, expr)
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
    num_state = number_states(bl, st)
    local_state = MArray{Tuple{num_state}, FT}(undef)
    for s in 1:num_state
        local_state[s] = state[ijk, s, e]
    end
    return Vars{vars_state(bl, st, FT)}(local_state)
end
