using Test
using CLIMA
using CLIMA.HydrostaticBoussinesq
using CLIMA.SimpleBox
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates
using CLIMA.Mesh.Grids: polynomialorder
import CLIMA.DGmethods: vars_state

function config_simple_box(FT, N, resolution, dimensions)
  prob = HomogeneousBox{FT}(dimensions...)

  cʰ = sqrt(grav * prob.H) # m/s
  model = HydrostaticBoussinesqModel{FT}(prob, cʰ = cʰ)

  config = CLIMA.Ocean_BoxGCM_Configuration("homogeneous_box",
                                            N, resolution, model)

  return config
end

function main(;imex::Bool = false)
  CLIMA.init()

  FT = Float64

  # DG polynomial order
  N = Int(4)

  # Domain resolution and size
  Nˣ = Int(20)
  Nʸ = Int(20)
  Nᶻ = Int(20)
  resolution = (Nˣ, Nʸ, Nᶻ)

  Lˣ = 4e6   # m
  Lʸ = 4e6   # m
  H  = 400   # m
  dimensions = (Lˣ, Lʸ, H)

  timestart = FT(0)    # s
  timeend   = FT(3600) # s

  if imex
    solver_type = CLIMA.IMEXSolverType(linear_model=LinearHBModel)
    Nᶻ = Int(400)
  else
    solver_type = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)
  end

  driver_config = config_simple_box(FT, N, resolution, dimensions)

  grid = driver_config.grid
  vert_filter = CutoffFilter(grid, polynomialorder(grid)-1)
  exp_filter  = ExponentialFilter(grid, 1, 8)
  modeldata = (vert_filter = vert_filter, exp_filter=exp_filter)

  solver_config = CLIMA.setup_solver(timestart, timeend, driver_config, init_on_cpu=true,
                                     ode_solver_type=solver_type, modeldata=modeldata)

  result = CLIMA.invoke!(solver_config)

  maxQ = Vars{vars_state(driver_config.bl, FT)}(maximum(solver_config.Q, dims=(1,3)))
  minQ = Vars{vars_state(driver_config.bl, FT)}(minimum(solver_config.Q, dims=(1,3)))

  @test maxQ.θ ≈ minQ.θ
end

main(imex=false)
main(imex=true)
