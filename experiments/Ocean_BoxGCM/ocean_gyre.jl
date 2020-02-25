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

function config_simple_box(FT, N, resolution, dimensions)
  prob = OceanGyre{FT}(dimensions...)

  cʰ = sqrt(grav * prob.H) # m/s
  model = HydrostaticBoussinesqModel{FT}(prob, cʰ = cʰ)

  config = CLIMA.Ocean_BoxGCM_Configuration("ocean_gyre",
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

  Lˣ = 4e6    # m
  Lʸ = 4e6    # m
  H  = 1000   # m
  dimensions = (Lˣ, Lʸ, H)

  timestart = FT(0)    # s
  timeend   = FT(30 * 86400) # s

  if imex
    solver_type = CLIMA.IMEXSolverType(linear_model=LinearHBModel)
  else
    solver_type = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)
  end
    
  driver_config = config_simple_box(FT, N, resolution, dimensions)

  grid = driver_config.grid
  vert_filter = CutoffFilter(grid, polynomialorder(grid)-1)
  exp_filter  = ExponentialFilter(grid, 1, 8)
  modeldata = (vert_filter = vert_filter, exp_filter=exp_filter)

  solver_config = CLIMA.setup_solver(timestart, timeend, driver_config, forcecpu=true,
                                     ode_solver_type=solver_type,  modeldata=modeldata)

  result = CLIMA.invoke!(solver_config)

end

main(imex=false)
