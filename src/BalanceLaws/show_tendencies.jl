##### Show tendencies

export show_tendencies

using PrettyTables

format_tend(tend_type) = "$(nameof(typeof(tend_type)))"

format_tends(tend_types) = "(" * join(format_tend.(tend_types), ", ") * ")"

"""
    show_tendencies(
        bl;
        include_module = false,
        table_complete = false,
    )

Show a table of the tendencies for each
prognostic variable for the given balance law.

## Arguments
 - `include_module[ = false]` will print not remove the module where each
   prognostic variable is defined (e.g., `Atmos.Mass`).
 - `table_complete[ = false]` will print a warning (if false) that the
   tendency table is incomplete.

Requires definitions for
 - [`prognostic_vars`](@ref)
 - [`eq_tends`](@ref)
for the balance law.
"""
function show_tendencies(bl; include_module = false, table_complete = false)
    prog_vars = prognostic_vars(bl)
    if !isempty(prog_vars)
        header = [
            "Equation" "Flux{FirstOrder}" "Flux{SecondOrder}" "Source"
            "(Y_i)" "(F_1)" "(F_2)" "(S)"
        ]
        if include_module
            eqs = collect(string.(typeof.(prog_vars)))
        else
            eqs = collect(last.(split.(string.(typeof.(prog_vars)), ".")))
        end
        fmt_tends(tt) = map(prog_vars) do pv
            format_tends(eq_tends(pv, bl, tt))
        end |> collect
        F1 = fmt_tends(Flux{FirstOrder}())
        F2 = fmt_tends(Flux{SecondOrder}())
        S = fmt_tends(Source())
        data = hcat(eqs, F1, F2, S)
        table_complete || @warn "This table is temporarily incomplete"
        println("\nPDE: ∂_t Y_i + (∇•F_1(Y))_i + (∇•F_2(Y,G)))_i = (S(Y,G))_i")
        pretty_table(
            data,
            header,
            header_crayon = crayon"yellow bold",
            subheader_crayon = crayon"green bold",
            crop = :none,
        )
        println("")
    else
        msg = "Defining `prognostic_vars` and\n"
        msg *= "`eq_tends` for $(nameof(typeof(bl))) will\n"
        msg *= "enable printing a table of tendencies."
        @info msg
    end
    return nothing
end
