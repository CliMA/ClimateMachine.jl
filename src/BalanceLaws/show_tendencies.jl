##### Show tendencies

export show_tendencies

using PrettyTables

first_param(t::TendencyDef{TT, PV}) where {TT, PV} = PV

function format_tend(tend_type, include_params)
    t = "$(nameof(typeof(tend_type)))"
    return include_params ? t * "{$(first_param(tend_type))}" : t
end

format_tends(tend_types, include_params) =
    "(" * join(format_tend.(tend_types, include_params), ", ") * ")"

"""
    show_tendencies(bl; include_params = false)

Show a table of the tendencies for each
prognostic variable for the given balance law.

Using `include_params = true` will include the type
parameters for each of the tendency fluxes
and sources.

Requires definitions for
 - [`prognostic_vars`](@ref)
 - [`eq_tends`](@ref)
for the balance law.
"""
function show_tendencies(bl; include_params = false)
    ip = include_params
    prog_vars = prognostic_vars(bl)
    if !isempty(prog_vars)
        header = [
            "Equation" "Flux{FirstOrder}" "Flux{SecondOrder}" "Source"
            "(Y_i)" "(F_1)" "(F_2)" "(S)"
        ]
        eqs = collect(string.(typeof.(prog_vars)))
        fmt_tends(tt) = map(prog_vars) do pv
            format_tends(eq_tends(pv, bl, tt), ip)
        end |> collect
        F1 = fmt_tends(Flux{FirstOrder}())
        F2 = fmt_tends(Flux{SecondOrder}())
        S = fmt_tends(Source())
        data = hcat(eqs, F1, F2, S)
        @warn "This table is temporarily incomplete"
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
