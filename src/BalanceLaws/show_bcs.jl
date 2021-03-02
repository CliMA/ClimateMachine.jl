##### Show boundary conditions

export show_bcs

type_param(::BCDef{PV}) where {PV} = PV

function fmt_bc(t_in, include_params)
    t = "$(nameof(typeof(t_in)))"
    return include_params ? t * "{$(type_param(t_in))}" : t
end

fmt_bcs(bcs) = "($(join(bcs, ", ")))"

function show_bcs(
    bl::BalanceLaw;
    include_params = false,
    include_module = false,
    table_complete = false,
)

    prog_vars = prognostic_vars(bl)
    if !isempty(prog_vars)
        all_bcs = boundary_conditions(bl)
        header = hcat(
            ["Equation"; "(Y_i)"],
            hcat(map(i -> ["BC"; "$i"], 1:length(all_bcs))...),
        )
        if include_module
            eqs = collect(string.(typeof.(prog_vars)))
        else
            eqs = collect(last.(split.(string.(typeof.(prog_vars)), ".")))
        end
        bcs_default = dispatched_tuple(default_bcs(bl))
        bcs_entries = map(all_bcs) do bc
            map(prognostic_vars(bl)) do prog
                tup = used_bcs(bl, prog, bc, bcs_default)
                fmt_bcs(map(tup) do bc_pv
                    fmt_bc(bc_pv, include_params)
                end)
            end
        end
        table_complete || @warn "This BC table is temporarily incomplete"

        data = hcat(eqs, collect.(bcs_entries)...)
        pretty_table(
            data,
            header,
            header_crayon = crayon"yellow bold",
            subheader_crayon = crayon"green bold",
            crop = :none,
        )
    else
        msg = "Defining `prognostic_vars` and\n"
        msg *= "`default_bcs` for $(nameof(typeof(bl))) will\n"
        msg *= "enable printing a table of boundary conditions."
        @info msg
    end

end