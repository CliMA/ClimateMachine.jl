using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)));

if parse(Bool, get(ENV, "CLIMATEMACHINE_PLOT_EDMF_COMPARISON", "false"))
    plot_dir = joinpath(clima_dir, "output", "bomex_edmf", "pycles_comparison")
else
    plot_dir = nothing
end

include(joinpath(@__DIR__, "compute_mse.jl"))

#! format: off
best_mse = Dict()
best_mse[:Bomex] = Dict()
best_mse[:Bomex]["ρ"] = 3.4943021267397123e-02
best_mse[:Bomex]["ρu[1]"] = 3.0714039084256679e+03
best_mse[:Bomex]["ρu[2]"] = 1.3375796498101822e-03
best_mse[:Bomex]["moisture.ρq_tot"] = 4.8463531712319707e-02
best_mse[:Bomex]["turbconv.environment.ρatke"] = 6.7458994186482869e+02
best_mse[:Bomex]["turbconv.environment.ρaθ_liq_cv"] = 8.5667200477293633e+01
best_mse[:Bomex]["turbconv.environment.ρaq_tot_cv"] = 1.6455670738880309e+02
best_mse[:Bomex]["turbconv.updraft[1].ρa"] = 7.9547300082766554e+01
best_mse[:Bomex]["turbconv.updraft[1].ρaw"] = 8.4294492216319045e-02
best_mse[:Bomex]["turbconv.updraft[1].ρaθ_liq"] = 9.0095767705492662e+00
best_mse[:Bomex]["turbconv.updraft[1].ρaq_tot"] = 1.0767427989197111e+01
#! format: on

sufficient_mse(computed_mse, best_mse) = computed_mse <= best_mse + eps()

function test_mse(computed_mse, best_mse, key)
    mse_not_regressed = sufficient_mse(computed_mse[key], best_mse[key])
    @test mse_not_regressed
    mse_not_regressed || @show key
end

computed_mse = Dict(
    k => compute_mse(
        solver_config.dg.grid,
        solver_config.dg.balance_law,
        time_data,
        dons_arr,
        data_files[k],
        k,
        best_mse[k],
        plot_dir,
    ) for k in keys(data_files)
)

@testset "BOMEX EDMF Solution Quality Assurance (QA) tests" begin
    #! format: off
    test_mse(computed_mse[:Bomex], best_mse[:Bomex], "ρ")
    test_mse(computed_mse[:Bomex], best_mse[:Bomex], "ρu[1]")
    test_mse(computed_mse[:Bomex], best_mse[:Bomex], "moisture.ρq_tot")
    test_mse(computed_mse[:Bomex], best_mse[:Bomex], "turbconv.updraft[1].ρa")
    test_mse(computed_mse[:Bomex], best_mse[:Bomex], "turbconv.updraft[1].ρaw")
    test_mse(computed_mse[:Bomex], best_mse[:Bomex], "turbconv.updraft[1].ρaθ_liq")
    test_mse(computed_mse[:Bomex], best_mse[:Bomex], "turbconv.updraft[1].ρaq_tot")
    test_mse(computed_mse[:Bomex], best_mse[:Bomex], "turbconv.environment.ρatke")
    test_mse(computed_mse[:Bomex], best_mse[:Bomex], "turbconv.environment.ρaθ_liq_cv")
    test_mse(computed_mse[:Bomex], best_mse[:Bomex], "turbconv.environment.ρaq_tot_cv")
    #! format: on
end
