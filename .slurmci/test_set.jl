cpu_tests = Set(((3, "examples/DGmethods_old/ex_001_periodic_advection.jl"),
                 (3, "examples/DGmethods_old/ex_002_solid_body_rotation.jl"),
                 (1, "test/Ocean/shallow_water/GyreDriver.jl")
                ))

cpu_gpu_tests = Set(((3, "examples/DGmethods_old/ex_001_periodic_advection.jl"),
                     (3, "examples/DGmethods_old/ex_002_solid_body_rotation.jl"),
                     (3, "examples/DGmethods_old/ex_003_acoustic_wave.jl"),
                     (3, "examples/DGmethods_old/ex_004_nonnegative.jl"),
                     (3, "examples/Microphysics/ex_1_saturation_adjustment.jl"),
                     (3, "examples/Microphysics/ex_2_Kessler.jl"),
                     (3, "test/DGmethods/advection_diffusion/pseudo1D_advection_diffusion.jl"),
                     (3, "test/DGmethods/Euler/isentropicvortex.jl"),
                     (3, "test/DGmethods/Euler/isentropicvortex-imex.jl"),
                     (3, "test/DGmethods/compressible_Navier_Stokes/mms_bc_atmos.jl"),
                     (3, "test/DGmethods/compressible_Navier_Stokes/mms_bc_dgmodel.jl"),
                     (3, "test/DGmethods/compressible_Navier_Stokes/rising_bubble-model.jl"),
                     (3, "test/DGmethods/compressible_Navier_Stokes/rising_bubble-model-imex.jl"),
                     (3, "test/DGmethods/compressible_Navier_Stokes/density_current-model.jl"),
                     (3, "test/DGmethods_old/Euler/RTB_IMEX.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone_IMEX.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone_aux.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone_bc.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone_integral.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone_source.jl"),
                     (3, "test/DGmethods_old/compressible_Navier_Stokes/mms_bc.jl"),
                     (3, "test/DGmethods_old/conservation/sphere.jl"),
                     (2, "test/DGmethods_old/sphere/advection_sphere_lsrk.jl"),
                     (2, "test/DGmethods_old/sphere/advection_sphere_ssp33.jl"),
                     (2, "test/DGmethods_old/sphere/advection_sphere_ssp34.jl"),
                     (2, "test/LinearSolvers/poisson.jl"),
                     (4, "examples/DGmethods/ex_001_dycoms.jl"),
                     (1, "test/Ocean/shallow_water/GyreDriver.jl")
                    ))

gpu_tests = Set(((3, "test/DGmethods/advection_diffusion/pseudo1D_advection_diffusion.jl true"),
                ))
