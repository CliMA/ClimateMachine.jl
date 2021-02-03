using ClimateMachine

if parse(Bool, get(ENV, "CLIMATEMACHINE_PLOT_EDMF_COMPARISON", "false"))
    using Plots
end

using OrderedCollections
using Test
using NCDatasets
using Dierckx
using PrettyTables
using Printf
using ClimateMachine.ArtifactWrappers

# Get PyCLES_output dataset folder:
#! format: off
PyCLES_output_dataset = ArtifactWrapper(
    @__DIR__,
    isempty(get(ENV, "CI", "")),
    "PyCLES_output",
    ArtifactFile[
    # ArtifactFile(url = "https://caltech.box.com/shared/static/johlutwhohvr66wn38cdo7a6rluvz708.nc", filename = "Rico.nc",),
    # ArtifactFile(url = "https://caltech.box.com/shared/static/zraeiftuzlgmykzhppqwrym2upqsiwyb.nc", filename = "Gabls.nc",),
    # ArtifactFile(url = "https://caltech.box.com/shared/static/toyvhbwmow3nz5bfa145m5fmcb2qbfuz.nc", filename = "DYCOMS_RF01.nc",),
    # ArtifactFile(url = "https://caltech.box.com/shared/static/ivo4751camlph6u3k68ftmb1dl4z7uox.nc", filename = "TRMM_LBA.nc",),
    # ArtifactFile(url = "https://caltech.box.com/shared/static/4osqp0jpt4cny8fq2ukimgfnyi787vsy.nc", filename = "ARM_SGP.nc",),
    ArtifactFile(url = "https://caltech.box.com/shared/static/jci8l11qetlioab4cxf5myr1r492prk6.nc", filename = "Bomex.nc",),
    # ArtifactFile(url = "https://caltech.box.com/shared/static/pzuu6ii99by2s356ij69v5cb615200jq.nc", filename = "Soares.nc",),
    # ArtifactFile(url = "https://caltech.box.com/shared/static/7upt639siyc2umon8gs6qsjiqavof5cq.nc", filename = "Nieuwstadt.nc",),
    ],
)
PyCLES_output_dataset_path = get_data_folder(PyCLES_output_dataset)
# data_files[:Rico] = Dataset(joinpath(PyCLES_output_dataset_path, "Rico.nc"), "r")
# data_files[:Gabls] = Dataset(joinpath(PyCLES_output_dataset_path, "Gabls.nc"), "r")
# data_files[:DYCOMS_RF01] = Dataset(joinpath(PyCLES_output_dataset_path, "DYCOMS_RF01.nc"), "r")
# data_files[:TRMM_LBA] = Dataset(joinpath(PyCLES_output_dataset_path, "TRMM_LBA.nc"), "r")
# data_files[:ARM_SGP] = Dataset(joinpath(PyCLES_output_dataset_path, "ARM_SGP.nc"), "r")
# data_files[:Soares] = Dataset(joinpath(PyCLES_output_dataset_path, "Soares.nc"), "r")
# data_files[:Nieuwstadt] = Dataset(joinpath(PyCLES_output_dataset_path, "Nieuwstadt.nc"), "r")
#! format: on

include("variable_map.jl")

function compute_mse(
    grid,
    bl,
    time_cm,
    dons_arr,
    ds,
    experiment,
    best_mse,
    plot_dir = nothing,
)
    mse = Dict()

    # Ensure domain matches:
    z_les = ds["z_half"][:]
    z_cm = get_z(grid; rm_dupes = true)
    @info "Z extent for LES vs CLIMA:"
    @show extrema(z_cm)
    @show extrema(z_les)

    time_les = ds["t"][:]

    # Find the nearest matching final time:
    t_cmp = min(time_cm[end], time_les[end])

    # Accidentally running a short simulation
    # could improve MSE. So, let's test that
    # we run for at least 400. We should increase
    # this as we can reach higher CFL.
    @test t_cmp >= 400

    # Ensure z_cm and dons_arr fields are consistent lengths:
    @test length(z_cm) == length(dons_arr[1][first(keys(dons_arr[1]))])

    data_cm = Dict()
    dons_cont = Dict()
    cm_variables = []
    computed_mse = []
    table_best_mse = []
    mse_reductions = []
    pycles_variables = []
    data_scales = []
    pycles_weight = []
    for (ftc) in keys(best_mse)
        # Only compare fields defined for var_map
        tup = var_map(ftc)
        tup == nothing && continue

        # Unpack the data
        LES_var = tup[1]
        facts = tup[2]
        push!(cm_variables, ftc)
        push!(pycles_variables, LES_var)
        data_ds = ds.group["profiles"]
        data_les = data_ds[LES_var][:]

        # Scale the data for comparison
        ρ = data_ds["rho"][:]
        a_up = data_ds["updraft_fraction"][:]
        a_en = 1 .- data_ds["updraft_fraction"][:]
        ρa_up = ρ .* a_up
        ρa_en = ρ .* a_en
        ρa = occursin("updraft", ftc) in ftc ? ρa_up : ρa_en
        if :a in facts && :ρ in facts
            data_les .*= ρa
            push!(pycles_weight, "ρa")
        elseif :ρ in facts
            data_les .*= ρ
            push!(pycles_weight, "ρ")
        else
            push!(pycles_weight, "1")
        end

        # Interpolate data
        steady_data = length(size(data_les)) == 1
        if steady_data
            data_les_cont = Spline1D(z_les, data_les)
        else # unsteady data
            data_les_cont = Spline2D(time_les, z_les, data_les')
        end
        data_cm_arr_ = [
            dons_arr[i][ftc][i_z]
            for i in 1:length(time_cm), i_z in 1:length(z_cm)
        ]
        data_cm_arr = reshape(data_cm_arr_, (length(time_cm), length(z_cm)))
        data_cm[ftc] = Spline2D(time_cm, z_cm, data_cm_arr)

        # Compute data scale
        data_scale = sum(abs.(data_les)) / length(data_les)
        push!(data_scales, data_scale)

        # Plot comparison
        if plot_dir ≠ nothing
            p = plot()
            if steady_data
                data_les_cont_mapped = map(z -> data_les_cont(z), z_cm)
            else
                data_les_cont_mapped = map(z -> data_les_cont(t_cmp, z), z_cm)
            end
            plot!(
                data_les_cont_mapped,
                z_cm ./ 10^3,
                xlabel = ftc,
                ylabel = "z [km]",
                label = "PyCLES",
            )
            plot!(
                map(z -> data_cm[ftc](t_cmp, z), z_cm),
                z_cm ./ 10^3,
                xlabel = ftc,
                ylabel = "z [km]",
                label = "CM",
            )
            mkpath(plot_dir)
            ftc_name = replace(ftc, "." => "_")
            savefig(joinpath(plot_dir, "$ftc_name.png"))
        end

        # Compute mean squared error (mse)
        if steady_data
            mse_single_var = sum(map(z_cm) do z
                (data_les_cont(z) - data_cm[ftc](t_cmp, z))^2
            end)
        else
            mse_single_var = sum(map(z_cm) do z
                (data_les_cont(t_cmp, z) - data_cm[ftc](t_cmp, z))^2
            end)
        end
        # Normalize by data scale
        mse[ftc] = mse_single_var / data_scale^2

        push!(mse_reductions, (best_mse[ftc] - mse[ftc]) / best_mse[ftc] * 100)
        push!(computed_mse, mse[ftc])
        push!(table_best_mse, best_mse[ftc])
    end

    # Tabulate output
    header = [
        "Variable" "Variable" "Weight" "Data scale" "MSE" "MSE" "MSE"
        "ClimateMachine (EDMF)" "PyCLES" "PyCLES" "" "Computed" "Best" "Reduction (%)"
    ]
    table_data = hcat(
        cm_variables,
        pycles_variables,
        pycles_weight,
        data_scales,
        computed_mse,
        table_best_mse,
        mse_reductions,
    )

    @info @sprintf(
        "Experiment comparison: %s at time t=%s\n",
        experiment,
        t_cmp
    )
    hl_worsened_mse = Highlighter(
        (data, i, j) -> !sufficient_mse(data[i, 5], data[i, 6]) && j == 5,
        crayon"red bold",
    )
    hl_worsened_mse_reduction = Highlighter(
        (data, i, j) -> !sufficient_mse(data[i, 5], data[i, 6]) && j == 7,
        crayon"red bold",
    )
    hl_improved_mse = Highlighter(
        (data, i, j) -> sufficient_mse(data[i, 5], data[i, 6]) && j == 7,
        crayon"green bold",
    )
    pretty_table(
        table_data,
        header,
        formatters = ft_printf("%.16e", 5:6),
        header_crayon = crayon"yellow bold",
        subheader_crayon = crayon"green bold",
        highlighters = (
            hl_worsened_mse,
            hl_improved_mse,
            hl_worsened_mse_reduction,
        ),
        crop = :none,
    )

    return mse
end

sufficient_mse(computed_mse, best_mse) = computed_mse <= best_mse + sqrt(eps())

function test_mse(computed_mse, best_mse, key)
    mse_not_regressed = sufficient_mse(computed_mse[key], best_mse[key])
    @test mse_not_regressed
    mse_not_regressed || @show key
end

function dons(diag_vs_z)
    return Dict(map(keys(first(diag_vs_z))) do k
        string(k) => [getproperty(ca, k) for ca in diag_vs_z]
    end)
end

get_dons_arr(diag_arr) = [dons(diag_vs_z) for diag_vs_z in diag_arr]

dons_arr = get_dons_arr(diag_arr)
