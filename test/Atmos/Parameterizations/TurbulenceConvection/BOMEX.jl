using Test

using CLIMA.TurbulenceConvection

const test_data_dir = joinpath(pwd(),"output","TestData")
mkpath(test_data_dir)

#### Accepting a new solution
# After a careful review of ALL of the solution results,
# set `accept_new_solution = true`, and run once. Then,
# reset to `false` before committing. This parameter should
# never be `true` for commits.
const accept_new_solution = false

tc = TurbulenceConvection
@testset "Integration test: EDMF equations (BOMEX)" begin
  grid, q, tmp = tc.EDMF.run(tc.EDMF.BOMEX())

  if accept_new_solution || !isfile(joinpath(test_data_dir, "q_expected.csv"))
    export_state(q, grid, test_data_dir, "q_expected.csv")
  end
  if accept_new_solution || !isfile(joinpath(test_data_dir, "tmp_expected.csv"))
    export_state(tmp, grid, test_data_dir, "tmp_expected.csv")
  end

  gm, en, ud, sd, al = allcombinations(q)
  DT = eltype(grid)
  q_expected = deepcopy(q)
  tmp_expected = deepcopy(tmp)
  assign!(q_expected, grid, DT(0))
  assign!(tmp_expected, grid, DT(0))

  import_state!(q_expected, grid, test_data_dir, "q_expected.csv")
  import_state!(tmp_expected, grid, test_data_dir, "tmp_expected.csv")

  D_q = compare(q, q_expected, grid, eps(Float32))
  D_tmp = compare(tmp, tmp_expected, grid, eps(Float32))

  @test all(D_q[:a])
  @test all(D_q[:w])
  @test all(D_q[:θ_liq])
  @test all(D_q[:q_tot])
  @test all(D_q[:tke])
  @test all(D_tmp[:q_liq])

end

