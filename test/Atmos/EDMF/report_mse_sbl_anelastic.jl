using ClimateMachine
const clima_dir = pkgdir(ClimateMachine);

if parse(Bool, get(ENV, "CLIMATEMACHINE_PLOT_EDMF_COMPARISON", "false"))
    plot_dir = joinpath(clima_dir, "output", "sbl_edmf", "pycles_comparison")
else
    plot_dir = nothing
end

include(joinpath(@__DIR__, "compute_mse.jl"))

data_file = Dataset(joinpath(PyCLES_output_dataset_path, "Gabls.nc"), "r")

#! format: off
best_mse = OrderedDict()
best_mse["prog_ρ"] = 9.3809207150296822e-03
best_mse["prog_ρu_1"] = 6.7269975116623837e+03
best_mse["prog_ρu_2"] = 6.8630628605220889e-01
#! format: on

computed_mse = compute_mse(
    solver_config.dg.grid,
    solver_config.dg.balance_law,
    time_data,
    dons_arr,
    data_file,
    "Gabls",
    best_mse,
    1800,
    plot_dir,
)

@testset "SBL Anelastic Solution Quality Assurance (QA) tests" begin
    #! format: off
    test_mse(computed_mse, best_mse, "prog_ρ")
    test_mse(computed_mse, best_mse, "prog_ρu_1")
    test_mse(computed_mse, best_mse, "prog_ρu_2")
    #! format: on
end
