if preset_exp_type == "MoistBaroclinicWave"
    # Moist Baroclinic wave experiment
    # - follows http://www-personal.umich.edu/~cjablono/DCMIP-2016_TestCaseDocument_10June2016.pdf
    # - initial conditions: bc_wave_init_base_state, bc_wave_init_perturbation

    # initial conditions
    init_pert_name = "deterministic" # options: "deterministic" ("random" to be added)
    init_basestate_name = "bc_wave_initstate" # options: "bc_wave_state"
    init_moist_name = "moist_low_tropics" # options: "moist_low_tropics", "zero"

    # boundary conditions
    function get_bc(FT)
        bc_list = (AtmosBC(), AtmosBC(), )
        return bc_list
    end

    # sources
    function get_source()
        source_list = (Gravity(), Coriolis(), )
        return source_list
    end

elseif preset_exp_type == "DryBaroclinicWave"
    # Dry Moist Baroclinic wave
    # - same as above but with q_tot = 0

    init_pert_name = "deterministic"
    init_basestate_name = "bc_wave_initstate"
    init_moist_name = "zero"

    # boundary conditions
    function get_bc(FT)
        bc_list = (AtmosBC(), AtmosBC(), )
        return bc_list
    end

    # sources
    function get_source()
        source_list = (Gravity(), Coriolis(), )
        return source_list
    end

elseif preset_exp_type == "DryHeldSuarez"
    # Dry Moist Baroclinic wave
    # - same as above but with q_tot = 0

    init_pert_name = "none"
    init_basestate_name = "heldsuarez_initstate"
    init_moist_name = "zero"

    # boundary conditions
    function get_bc(FT)
        bc_list = (AtmosBC(), AtmosBC(), )
        return bc_list
    end

    # sources
    function get_source()
        source_list = (Gravity(), Coriolis(), held_suarez_forcing!)
        return source_list
    end
elseif preset_exp_type == "MoistHeldSuarez_no_sfcfluxes"
    # Dry Moist Baroclinic wave
    # - same as above but with q_tot = 0

    init_pert_name = "deterministic"
    init_basestate_name = "heldsuarez_initstate"
    init_moist_name = "moist_low_tropics"

    # boundary conditions
    function get_bc(FT)
        bc_list = (AtmosBC(), AtmosBC(), )
        return bc_list
    end

    # sources
    function get_source()
        source_list = (Gravity(), Coriolis(), held_suarez_forcing!)
        return source_list
    end

elseif preset_exp_type == "MoistHeldSuarez_bulk_sfcfluxes"
    # Dry Moist Baroclinic wave
    # - same as above but with q_tot = 0

    init_pert_name = "none" #"deterministic"
    init_basestate_name = "heldsuarez_initstate"
    init_moist_name = "zero"

    # boundary conditions
    function get_bc(FT)
        # Surface flux parameters
        C_drag = FT(0.0044)   # Bulk transfer coefficient
        T_sfc = FT(271)     # Surface temperature `[K]`

        bc_list = (
            AtmosBC(
                energy = BulkFormulaEnergySpatiallyVarying(
                    (state, aux, t, normPu_int) -> C_drag,
                    (state, aux, t) -> T_sfc,
                ),
                moisture = BulkFormulaMoistureSpatiallyVarying(
                    (state, aux, t, normPu_int) -> C_drag,
                    (state, aux, t) -> T_sfc,
                ),
            ),
            AtmosBC(),
        )
        return bc_list
    end

    # sources
    function get_source()
        source_list = (Gravity(), Coriolis(), held_suarez_forcing!)
        return source_list
    end
else
    error = define_your_init_cond_and_bc_cond_and_sources

end
