using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)));

if parse(Bool, get(ENV, "CLIMATEMACHINE_PLOT_EDMF_COMPARISON", "false"))
    plot_dir = joinpath(clima_dir, "output", "sbl_edmf", "pycles_comparison")
else
    plot_dir = nothing
end

include(joinpath(@__DIR__, "compute_mse.jl"))

data_file = Dataset(joinpath(PyCLES_output_dataset_path, "Gabls.nc"), "r")

N = polynomialorders(solver_config.dg.grid)
if !(N[2] == 0)
    #! format: off
    best_mse = OrderedDict()
    best_mse["prog_ρ"] = 8.4569296166338691e-03
    best_mse["prog_ρu_1"] = 6.3882515387705889e+03
    best_mse["prog_ρu_2"] = 1.9273137570200704e-04
    best_mse["prog_turbconv_environment_ρatke"] = 2.8547391074963491e+02
    best_mse["prog_turbconv_environment_ρaθ_liq_cv"] = 8.8855089232461694e+01
    best_mse["prog_turbconv_updraft_1_ρa"] = 2.4934236012539099e+01
    best_mse["prog_turbconv_updraft_1_ρaw"] = 3.5164073148198044e-01
    best_mse["prog_turbconv_updraft_1_ρaθ_liq"] = 2.0173652967933322e+01
    #! format: on
else
    #! format: off
    best_mse = OrderedDict()
    best_mse["prog_ρ"] = 8.3506568674116093e-03
    best_mse["prog_ρu_1"] = 6.2714070693454869e+03
    best_mse["prog_ρu_2"] = 1.2793997157719303e-04
    best_mse["prog_turbconv_environment_ρatke"] = 2.3701966316184607e+02
    best_mse["prog_turbconv_environment_ρaθ_liq_cv"] = 8.7727376552999246e+01
    best_mse["prog_turbconv_updraft_1_ρa"] = 1.7950627389897488e+01
    best_mse["prog_turbconv_updraft_1_ρaw"] = 1.7799954766804618e-01
    best_mse["prog_turbconv_updraft_1_ρaθ_liq"] = 1.3315847801000023e+01
    #! format: on
end

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
