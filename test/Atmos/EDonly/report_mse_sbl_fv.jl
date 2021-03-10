using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)));

if parse(Bool, get(ENV, "CLIMATEMACHINE_PLOT_EDMF_COMPARISON", "false"))
    plot_dir = joinpath(clima_dir, "output", "sbl_edmf", "pycles_comparison")
else
    plot_dir = nothing
end

include(joinpath(@__DIR__, "compute_mse.jl"))

data_file = Dataset(joinpath(PyCLES_output_dataset_path, "Gabls.nc"), "r")

#! format: off
best_mse = OrderedDict()
best_mse["prog_ρ"] = 6.9637575838887369e-03
best_mse["prog_ρu_1"] = 6.2604203232863292e+03
best_mse["prog_ρu_2"] = 1.3200376252941846e-04
best_mse["prog_turbconv_environment_ρatke"] = 2.3729827149654821e+02
best_mse["prog_turbconv_environment_ρaθ_liq_cv"] = 8.7727378291873919e+01
best_mse["prog_turbconv_updraft_1_ρa"] = 1.7947149151486141e+01
best_mse["prog_turbconv_updraft_1_ρaw"] = 1.7799688543526015e-01
best_mse["prog_turbconv_updraft_1_ρaθ_liq"] = 1.3315385576514325e+01
#! format: on

computed_mse = compute_mse(
    solver_config.dg.grid,
    solver_config.dg.balance_law,
    time_data,
    dons_arr,
    data_file,
    "Gabls",
    best_mse,
    60,
    plot_dir,
)

@testset "SBL EDMF Solution Quality Assurance (QA) tests" begin
    #! format: off
    test_mse(computed_mse, best_mse, "prog_ρ")
    test_mse(computed_mse, best_mse, "prog_ρu_1")
    test_mse(computed_mse, best_mse, "prog_ρu_2")
    test_mse(computed_mse, best_mse, "prog_turbconv_environment_ρatke")
    test_mse(computed_mse, best_mse, "prog_turbconv_environment_ρaθ_liq_cv")
    test_mse(computed_mse, best_mse, "prog_turbconv_updraft_1_ρa")
    test_mse(computed_mse, best_mse, "prog_turbconv_updraft_1_ρaw")
    test_mse(computed_mse, best_mse, "prog_turbconv_updraft_1_ρaθ_liq")
    #! format: on
end
