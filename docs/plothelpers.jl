using Plots
using ClimateMachine.BalanceLaws: Prognostic, Auxiliary, GradientFlux

"""
    plot_friendly_name(ϕ)

Get plot-friendly string, since many Unicode
characters do not render in plot labels.
"""
function plot_friendly_name(ϕ)
    s = ϕ
    s = replace(s, "ρ" => "rho")
    s = replace(s, "α" => "alpha")
    s = replace(s, "∂" => "partial")
    s = replace(s, "∇" => "nabla")
    return s
end

"""
    export_plot(z, all_data, ϕ_all, filename;
                xlabel, ylabel, time_data, round_digits, horiz_layout)

Export plot of all variables, or all
available time-steps in `all_data`.
"""
function export_plot(
    z,
    all_data::Array,
    ϕ_all,
    filename;
    xlabel,
    ylabel,
    time_data,
    round_digits = 2,
    horiz_layout = false,
)
    ϕ_all isa Tuple || (ϕ_all = (ϕ_all,))
    single_var = ϕ_all[1] == xlabel && length(ϕ_all) == 1
    p = plot()
    for (t, data) in zip(time_data, all_data)
        for ϕ in ϕ_all
            ϕ_string = String(ϕ)
            ϕ_name = plot_friendly_name(ϕ_string)
            ϕ_data = data[ϕ_string][:]
            label = single_var ? "t=$(round(t, digits=round_digits))" :
                "$(ϕ_string), t=$(round(t, digits=round_digits))"
            if !horiz_layout
                plot!(
                    ϕ_data,
                    z;
                    xlabel = xlabel,
                    ylabel = ylabel,
                    label = label,
                )
            else
                plot!(
                    z,
                    ϕ_data;
                    xlabel = xlabel,
                    ylabel = ylabel,
                    label = label,
                )
            end
        end
    end
    savefig(filename)
end

function save_binned_surface_plots(
    x,
    y,
    z,
    title,
    filename,
    n_plots = (3, 3),
    z_label_prefix = "z",
    n_digits = 5,
)
    n_z_partitions = prod(n_plots)
    z_min_global = min(z...)
    z_max_global = max(z...)
    Δz = (z_max_global - z_min_global) / n_z_partitions
    z_min = ntuple(i -> z_min_global + (i - 1) * Δz, n_z_partitions)
    z_max = ntuple(i -> z_min_global + (i) * Δz, n_z_partitions)
    p = []
    for i in 1:n_z_partitions
        mask = z_min[i] .<= z .<= z_max[i]
        x_i = x[mask]
        y_i = y[mask]
        sz_min = string(z_min[i])[1:min(n_digits, length(string(z_min[i])))]
        sz_max = string(z_max[i])[1:min(n_digits, length(string(z_max[i])))]
        p_i = plot(
            x_i,
            y_i,
            title = "$(title), in ($sz_min, $sz_max)",
            seriestype = :scatter,
            markersize = 5,
        )
        push!(p, p_i)
    end
    plot(p..., layout = n_plots, legend = false)
    savefig(filename)
end;

state_prefix(::Prognostic) = "prog_"
state_prefix(::Auxiliary) = "aux_"
state_prefix(::GradientFlux) = "grad_flux_"

"""
    plot_results(solver_config, all_data, time_data, output_dir)

Exports plots of states given
 - `solver_config` a `SolverConfiguration`
 - `all_data` an array of dictionaries, returned from `dict_of_nodal_states`
 - `time_data` an array of time values
 - `output_dir` output directory
"""
function export_state_plots(
    solver_config,
    all_data,
    time_data,
    output_dir;
    state_types = (Prognostic(), Auxiliary()),
)
    FT = eltype(solver_config.Q)
    z = get_z(solver_config.dg.grid)
    mkpath(output_dir)
    for st in state_types
        vs = vars_state(solver_config.dg.balance_law, st, FT)
        for fn in flattenednames(vs)
            file_name = state_prefix(st) * replace(fn, "." => "_")
            export_plot(
                z,
                all_data,
                (fn,),
                joinpath(output_dir, "$(file_name).png");
                xlabel = fn,
                ylabel = "z [m]",
                time_data = time_data,
                round_digits = 5,
            )
        end
    end
end
