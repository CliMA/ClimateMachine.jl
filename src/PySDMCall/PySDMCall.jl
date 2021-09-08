

module PySDMCall

using PyCall

export PySDM, pysdm_init, PySDMConf, PySDMKernels, PySDMSpectra, bilinear_interpol, export_particles_to_vtk


mutable struct PySDM
    conf
    core
    rhod # <class 'numpy.ndarray'> czy ρ to to ??? - nie, ale bierzemy go
    field_values #
    exporter
end



mutable struct PySDMConf
    grid
    size
    dxdz
    simtime
    dt
    n_sd
    kappa # 1 hygroscopicity
    kernel # Geometric from PySDM
    spectrum_per_mass_of_dry_air #
end

function PySDMConf(
 # grid (75, 75)
    size,  # (1500, 1500)
    dxdz, # (dx, dz)
    simtime, # 1800
    dt,
    n_sd_per_gridbox,
    kappa, # 1
    kernel,
    spectrum_per_mass_of_dry_air #
)
    grid = (Int(size[1]/dxdz[1]), Int(size[2]/dxdz[2]))

    n_sd = grid[1] * grid[2] * n_sd_per_gridbox

    physics_constants = pyimport("PySDM.physics.constants")

    size_si = (py"$size[0] * $physics_constants.si.metres", py"$size[1] * $physics_constants.si.metres")

    simtime_si = py"$simtime * $physics_constants.si.second"

    dt_si = py"$dt * $physics_constants.si.second"
    dxdz_si = (py"$dxdz[0] * $physics_constants.si.metres", py"$dxdz[1] * $physics_constants.si.metres")


    PySDMConf(
        grid,
        size_si,
        dxdz_si,
        simtime_si,
        dt_si,
        n_sd,
        kappa,
        kernel,
        spectrum_per_mass_of_dry_air
    )
end


# CMStepper # coś co ma metodę wait & step # https://github.com/atmos-cloud-sim-uj/PySDM-examples/blob/main/PySDM_examples/Arabas_et_al_2015/mpdata.py



function __init__()
    # adds current directory in the Python search path.
    pushfirst!(PyVector(pyimport("sys")."path"), "")
    pushfirst!(PyVector(pyimport("sys")."path"), "./src/PySDMCall/")
end


function pysdm_init!(pysdm, varvals)
    pkg_formulae = pyimport("PySDM.physics.formulae")
    pkg_builder = pyimport("PySDM.builder")
    pkg_dynamics = pyimport("PySDM.dynamics")
    pkg_init = pyimport("PySDM.initialisation")
    pkg_backend = pyimport("PySDM.backends")
    pkg_clima = pyimport("clima_hydrodynamics")

    print("pysdm.conf.n_sd: ")
    println(pysdm.conf.n_sd)

    formulae = pkg_formulae.Formulae() # TODO: state_variable_triplet="TODO")
    builder = pkg_builder.Builder(n_sd=pysdm.conf.n_sd, backend=pkg_backend.CPU, formulae=formulae)


    pysdm.rhod = varvals["ρ"][:, 1, :] # pysdm rhod differs a little bit from CliMa rho
    u1 = varvals["ρu[1]"][:, 1, :] ./ pysdm.rhod
    u3 = varvals["ρu[3]"][:, 1, :] ./ pysdm.rhod

    @assert size(pysdm.rhod) == (76, 76)
    @assert size(u1) == (76, 76)
    @assert size(u3) == (76, 76)


    courant_coef_u1 = pysdm.conf.dxdz[1] / pysdm.conf.dt
    courant_coef_u3 = pysdm.conf.dxdz[2] / pysdm.conf.dt
    u1 = u1 ./ courant_coef_u1
    u3 = u3 ./ courant_coef_u3

    arkw_u1 = [ (u1[y, x-1] + u1[y, x]) / 2 for y in 1:size(u1)[1], x in 2:size(u1)[2]]
    arkw_u3 = [ (u3[y-1, x] + u3[y, x]) / 2 for y in 2:size(u3)[1], x in 1:size(u3)[2]]

    @assert size(arkw_u1) == (76, 75)
    @assert size(arkw_u3) == (75, 76)

    println("Arakawa grid: ")
    println(size(arkw_u1))
    println(size(arkw_u3))

    courant_field = (arkw_u1, arkw_u3)

    pysdm.rhod = bilinear_interpol(pysdm.rhod)

    pkg_env = pyimport("Kinematic2DMachine")

    environment = pkg_env.Kinematic2DMachine(dt=pysdm.conf.dt, grid=pysdm.conf.grid, size=pysdm.conf.size, clima_rhod=pysdm.rhod)

    builder.set_environment(environment)
    println(environment.mesh.__dict__)

    # builder.add_dynamic(pkg_dynamics.AmbientThermodynamics()) # override env.sync()   # sync in fields from CM  w tym miejscu pobieramy pola z CliMa

    builder.add_dynamic(pkg_dynamics.Condensation(kappa=pysdm.conf.kappa))


    pysdm_th = varvals["theta_dry"][:, 1, :]
    @assert size(pysdm_th) == (76, 76)
    pysdm_th = bilinear_interpol(pysdm_th)
    @assert size(pysdm_th) == (75, 75)

    pysdm_qv = varvals["q_vap"][:, 1, :]
    @assert size(pysdm_qv) == (76, 76)
    pysdm_qv = bilinear_interpol(pysdm_qv)
    @assert size(pysdm_qv) == (75, 75)


    builder.add_dynamic(pkg_clima.ClimateMachine(py"{'qv': $pysdm_qv, 'th': $pysdm_th}"))

    # has sense only for multithreading
    # builder.add_dynamic(pkg_dynamics.EulerianAdvection(solver = CMStepper())) # sync out field to CM and launch CM advection

    builder.add_dynamic(pkg_dynamics.Displacement(courant_field=courant_field,
                                                  # scheme="FTBS",
                                                  enable_sedimentation=false)) # enable_sedimentation=true

    builder.add_dynamic(pkg_dynamics.Coalescence(kernel=pysdm.conf.kernel))


    attributes = environment.init_attributes(spatial_discretisation=pkg_init.spatial_sampling.Pseudorandom(),
                                             spectral_discretisation=pkg_init.spectral_sampling.ConstantMultiplicity(
                                                    spectrum=pysdm.conf.spectrum_per_mass_of_dry_air
                                             ),
                                             kappa=pysdm.conf.kappa)


    pkg_PySDM_products = pyimport("PySDM.products")
    # pass in in meters
    liquid_water_mixing_ratio_product = pkg_PySDM_products.WaterMixingRatio(name="qc", description_prefix="liquid", radius_range=(0.0, Inf))
    relative_humidity_product = pkg_PySDM_products.RelativeHumidity()

    pysdm.core = builder.build(attributes, products=[liquid_water_mixing_ratio_product, relative_humidity_product])

    ####
    pkg_vtkexp = pyimport("PySDM.exporters.vtk_exporter")
    pysdm.exporter = pkg_vtkexp.VTKExporter(verbose=true)


    return nothing
end


function bilinear_interpol(A)
    array_size = size(A)

    A = [ (A[y, x-1] + A[y, x]) / 2 for y in 1:size(A)[1], x in 2:size(A)[2]]
    @assert size(A) == (array_size[1], array_size[2]-1)
    A = [ (A[y-1, x] + A[y, x]) / 2 for y in 2:size(A)[1], x in 1:size(A)[2]]
    @assert size(A) == (array_size[1]-1, array_size[2]-1)

    return A
end

function PySDMKernels()
    pyimport("PySDM.physics.coalescence_kernels")
end

function PySDMSpectra()
    pyimport("PySDM.physics.spectra")
end

function export_particles_to_vtk(pysdm)
    if !isnothing(pysdm.exporter)
        pysdm.exporter.export_particles(pysdm.core)
    end
end

end # module PySDMCall
