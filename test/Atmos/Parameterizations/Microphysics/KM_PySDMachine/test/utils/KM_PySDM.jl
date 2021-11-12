import Thermodynamics
const THDS = Thermodynamics
import CLIMAParameters
const CP = CLIMAParameters

using PyCall


function set_up_pysdm(
    solver_config,
    interpol,
    domain_extents_2d,
    domain_resolution_2d,
    init!,
    do_step!,
    fini!,
)

    krnl = PySDMKernels()

    spectra = PySDMSpectra()

    rho_spectrum = 1
    cm = 1e-2
    nm = 1e-9
    spectrum_per_mass_of_dry_air = spectra.Lognormal(
        norm_factor = 60 / cm^3 / rho_spectrum,
        m_mode = 40 * nm,
        s_geom = 1.4,
    )

    n_sd = 25

    pysdmconf = PySDMConfig(
        domain_extents_2d,
        domain_resolution_2d,
        solver_config.timeend,
        solver_config.dt,
        n_sd,
        1,
        krnl.Geometric(collection_efficiency = 1),
        spectrum_per_mass_of_dry_air,
    )

    pysdm_cw = PySDMCallWrapper(pysdmconf, init!, do_step!, fini!)

    pysdm_cb = GenericCallbacks.AtInit(PySDMCallback(
        "PySDMCallback",
        solver_config.dg,
        interpol,
        solver_config.mpicomm,
        pysdm_cw,
    ))

    pysdm_cb
end


function init!(pysdm, varvals)
    pkg_formulae = pyimport("PySDM.physics.formulae")
    pkg_builder = pyimport("PySDM.builder")
    pkg_dynamics = pyimport("PySDM.dynamics")
    pkg_init = pyimport("PySDM.initialisation")
    pkg_backend = pyimport("PySDM.backends")
    pkg_vtk_exp = pyimport("PySDM.exporters")

    formulae = pkg_formulae.Formulae()
    backend = pkg_backend.CPU(formulae)
    builder = pkg_builder.Builder(n_sd = pysdm.config.n_sd, backend = backend)

    pysdm.rhod = varvals["ρ"][:, 1, :]
    u1 = varvals["ρu[1]"][:, 1, :] ./ pysdm.rhod
    u3 = varvals["ρu[3]"][:, 1, :] ./ pysdm.rhod

    courant_coef_u1 = pysdm.config.dxdz[1] / pysdm.config.dt
    courant_coef_u3 = pysdm.config.dxdz[2] / pysdm.config.dt
    u1 = u1 ./ courant_coef_u1
    u3 = u3 ./ courant_coef_u3

    arkw_u1 = [
        (u1[y, x - 1] + u1[y, x]) / 2
        for y in 1:size(u1)[1], x in 2:size(u1)[2]
    ]
    arkw_u3 = [
        (u3[y - 1, x] + u3[y, x]) / 2
        for y in 2:size(u3)[1], x in 1:size(u3)[2]
    ]

    courant_field = (arkw_u1, arkw_u3)

    pysdm.rhod = cell_center_mean(pysdm.rhod)

    pkg_env = pyimport("Kinematic2DMachine")

    environment = pkg_env.Kinematic2DMachine(
        dt = pysdm.config.dt,
        grid = pysdm.config.grid,
        size = pysdm.config.size,
        clima_rhod = pysdm.rhod,
    )

    builder.set_environment(environment)

    builder.add_dynamic(pkg_dynamics.Condensation())


    pysdm_thd = varvals["theta_dry"][:, 1, :]
    pysdm_thd = cell_center_mean(pysdm_thd)

    pysdm_qv = varvals["q_vap"][:, 1, :]
    pysdm_qv = cell_center_mean(pysdm_qv)

    environment.set_thd(pysdm_thd)
    environment.set_qv(pysdm_qv)

    displacement = pkg_dynamics.Displacement(enable_sedimentation = false)

    builder.add_dynamic(displacement)

    builder.add_dynamic(pkg_dynamics.Coalescence(kernel = pysdm.config.kernel))


    attributes = environment.init_attributes(
        spatial_discretisation = pkg_init.spatial_sampling.Pseudorandom(),
        spectral_discretisation = pkg_init.spectral_sampling.ConstantMultiplicity(
            spectrum = pysdm.config.spectrum_per_mass_of_dry_air,
        ),
        kappa = pysdm.config.kappa,
    )


    pkg_PySDM_products = pyimport("PySDM.products")
    pysdm_products = []
    push!(
        pysdm_products,
        pkg_PySDM_products.WaterMixingRatio(
            name = "qc",
            description_prefix = "liquid",
            radius_range = (0.0, Inf),
        ),
    )

    push!(pysdm_products, pkg_PySDM_products.RelativeHumidity())
    push!(pysdm_products, pkg_PySDM_products.ParticleMeanRadius())
    push!(pysdm_products, pkg_PySDM_products.PeakSupersaturation())
    push!(pysdm_products, pkg_PySDM_products.ActivatingRate())
    push!(pysdm_products, pkg_PySDM_products.DeactivatingRate())
    push!(pysdm_products, pkg_PySDM_products.CondensationTimestepMin())
    push!(pysdm_products, pkg_PySDM_products.CondensationTimestepMax())
    push!(
        pysdm_products,
        pkg_PySDM_products.CloudDropletEffectiveRadius(
            radius_range = (0.0, Inf),
        ),
    )

    pysdm.particulator = builder.build(attributes, products = pysdm_products)

    displacement.upload_courant_field(courant_field)

    pysdm.exporter = pkg_vtk_exp.VTKExporter(verbose = true)

    return nothing
end


function do_step!(pysdm, varvals, t)

    dynamics = pysdm.particulator.dynamics

    dynamics["Displacement"]()

    update_pysdm_fields!(pysdm, varvals, t)

    pysdm.particulator.env.sync()

    dynamics["Condensation"]()

    pysdm.particulator._notify_observers()

    export_attributes_to_vtk(pysdm)
    export_products_to_vtk(pysdm)

    pysdm.particulator.n_steps += 1
end


function update_pysdm_fields!(pysdm, vals, t)

    liquid_water_mixing_ratio = pysdm.particulator.products["qc"].get() * 1e-3

    liquid_water_specific_humidity = liquid_water_mixing_ratio

    q_tot = vals["q_tot"][:, 1, :]
    q_tot = cell_center_mean(q_tot)

    q = THDS.PhasePartition.(q_tot, liquid_water_specific_humidity, 0.0)

    qv = THDS.vapor_specific_humidity.(q)

    e_int = vals["e_int"][:, 1, :]
    e_int = cell_center_mean(e_int)

    T = THDS.air_temperature.(param_set, e_int, q)

    ρ = pysdm.rhod
    thd = THDS.dry_pottemp.(param_set, T, ρ)

    pysdm.particulator.env.set_thd(thd)
    pysdm.particulator.env.set_qv(qv)

    return nothing
end

function fini!(pysdm, varvals, t)
    write_pvd(pysdm)
end

function cell_center_mean(A)
    A = [(A[y, x - 1] + A[y, x]) / 2 for y in 1:size(A)[1], x in 2:size(A)[2]]
    A = [(A[y - 1, x] + A[y, x]) / 2 for y in 2:size(A)[1], x in 1:size(A)[2]]
    return A
end

function PySDMKernels()
    pyimport("PySDM.physics.coalescence_kernels")
end

function PySDMSpectra()
    pyimport("PySDM.physics.spectra")
end

function export_attributes_to_vtk(pysdm)
    if !isnothing(pysdm.exporter)
        pysdm.exporter.export_attributes(pysdm.particulator)
    end
end

function export_products_to_vtk(pysdm)
    if !isnothing(pysdm.exporter)
        pysdm.exporter.export_products(pysdm.particulator)
    end
end

function write_pvd(pysdm)
    if !isnothing(pysdm.exporter)
        pysdm.exporter.write_pvd()
    end
end

function export_plt(var, title, t)
    py"""
    from matplotlib.pyplot import cm
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_vars(A, title=None):
        # Contour Plot
        plt.clf()
        X, Y = np.mgrid[0:A.shape[0], 0:A.shape[1]]
        Z = A
        cp = plt.contourf(X, Y, Z)
        cb = plt.colorbar(cp)
        if title:
            plt.title(title)

        plt.show()
        return plt
    """

    plt = py"plot_vars($var, title=$title)"
    plt.savefig(string(title, t, ".png"))
end
