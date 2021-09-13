using Dierckx

include("KinematicModel.jl")

# speed up the relaxation timescales for cloud water and cloud ice
CLIMAParameters.Atmos.Microphysics.τ_cond_evap(::AbstractParameterSet) = 0.5
CLIMAParameters.Atmos.Microphysics.τ_sub_dep(::AbstractParameterSet) = 0.5

function vars_state(m::KinematicModel, ::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        ρq_tot::FT
        ρq_liq::FT
        ρq_ice::FT
        ρq_rai::FT
        ρq_sno::FT
    end
end

function vars_state(m::KinematicModel, ::Auxiliary, FT)
    @vars begin
        # defined in init_state_auxiliary
        p::FT
        z_coord::FT
        x_coord::FT
        # defined in update_aux
        u::FT
        w::FT
        q_tot::FT
        q_vap::FT
        q_liq::FT
        q_ice::FT
        q_rai::FT
        q_sno::FT
        e_tot::FT
        e_kin::FT
        e_pot::FT
        e_int::FT
        T::FT
        S_liq::FT
        S_ice::FT
        RH::FT
        rain_w::FT
        snow_w::FT
        # more diagnostics
        src_cloud_liq::FT
        src_cloud_ice::FT
        src_rain_acnv::FT
        src_snow_acnv::FT
        src_liq_rain_accr::FT
        src_liq_snow_accr::FT
        src_ice_snow_accr::FT
        src_ice_rain_accr::FT
        src_snow_rain_accr::FT
        src_rain_accr_sink::FT
        src_rain_evap::FT
        src_snow_subl::FT
        src_snow_melt::FT
        flag_cloud_liq::FT
        flag_cloud_ice::FT
        flag_rain::FT
        flag_snow::FT
        # helpers for bc
        ρe_init::FT
        ρq_tot_init::FT
    end
end

function init_kinematic_eddy!(eddy_model, state, aux, localgeo, t, spline_fun)

    FT = eltype(state)
    _grav::FT = grav(param_set)

    dc = eddy_model.data_config

    (x, y, z) = localgeo.coord
    (xc, yc, zc) = localgeo.center_coord

    @inbounds begin

        init_T, init_qt, init_p, init_ρ, init_dρ = spline_fun

        # density
        q_pt_0 = PhasePartition(init_qt(z))
        R_m, cp_m, cv_m, γ = gas_constants(param_set, q_pt_0)
        T::FT = init_T(z)
        ρ::FT = init_ρ(z)
        state.ρ = ρ
        aux.p = init_p(z)

        # moisture
        state.ρq_tot = ρ * init_qt(z)
        state.ρq_liq = ρ * q_pt_0.liq
        state.ρq_ice = ρ * q_pt_0.ice
        state.ρq_rai = ρ * FT(0)
        state.ρq_sno = ρ * FT(0)

        # [Grabowski1998](@cite)
        # velocity (derivative of streamfunction)
        # This is actually different than what comes out from taking a
        # derivative of Ψ from the paper. I have sin(π/2/X(x-xc)).
        # This setup makes more sense to me though.
        _Z::FT = FT(15000)
        _X::FT = FT(10000)
        _xc::FT = FT(30000)
        _A::FT = FT(4.8 * 1e4)
        _S::FT = FT(2.5 * 1e-2) * FT(0.01) #TODO
        _ρ_00::FT = FT(1)
        ρu::FT = FT(0)
        ρw::FT = FT(0)
        fact =
            _A / _ρ_00 * (
                init_ρ(z) * FT(π) / _Z * cos(FT(π) / _Z * z) +
                init_dρ(z) * sin(FT(π) / _Z * z)
            )

        if zc < _Z
            if x >= (_xc + _X)
                ρu = _S * z - fact
                ρw = FT(0)
            elseif x <= (_xc - _X)
                ρu = _S * z + fact
                ρw = FT(0)
            else
                ρu = _S * z - fact * sin(FT(π / 2.0) / _X * (x - _xc))
                ρw =
                    _A * init_ρ(z) / _ρ_00 * FT(π / 2.0) / _X *
                    sin(FT(π) / _Z * z) *
                    cos(FT(π / 2.0) / _X * (x - _xc))
            end
        else
            ρu = _S * z
            ρw = FT(0)
        end
        state.ρu = SVector(ρu, FT(0), ρw)
        u::FT = ρu / ρ
        w::FT = ρw / ρ

        # energy
        e_kin::FT = 1 // 2 * (u^2 + w^2)
        e_pot::FT = _grav * z
        e_int::FT = internal_energy(param_set, T, q_pt_0)
        e_tot::FT = e_kin + e_pot + e_int
        state.ρe = ρ * e_tot
    end
    return nothing
end

function nodal_update_auxiliary_state!(
    m::KinematicModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    _grav::FT = grav(param_set)
    _T_freeze::FT = T_freeze(param_set)

    @inbounds begin

        if t == FT(0)
            aux.ρe_init = state.ρe
            aux.ρq_tot_init = state.ρq_tot
        end

        # velocity
        aux.u = state.ρu[1] / state.ρ
        aux.w = state.ρu[3] / state.ρ
        # water
        aux.q_tot = state.ρq_tot / state.ρ
        aux.q_liq = state.ρq_liq / state.ρ
        aux.q_ice = state.ρq_ice / state.ρ
        aux.q_rai = state.ρq_rai / state.ρ
        aux.q_sno = state.ρq_sno / state.ρ
        aux.q_vap = aux.q_tot - aux.q_liq - aux.q_ice
        # energy
        aux.e_tot = state.ρe / state.ρ
        aux.e_kin = 1 // 2 * (aux.u^2 + aux.w^2)
        aux.e_pot = _grav * aux.z_coord
        aux.e_int = aux.e_tot - aux.e_kin - aux.e_pot
        # supersaturation
        q = PhasePartition(aux.q_tot, aux.q_liq, aux.q_ice)
        aux.T = air_temperature(param_set, aux.e_int, q)
        ts_neq = PhaseNonEquil_ρTq(param_set, state.ρ, aux.T, q)
        aux.S_liq = max(0, supersaturation(ts_neq, Liquid()))
        aux.S_ice = max(0, supersaturation(ts_neq, Ice()))
        aux.RH = relative_humidity(ts_neq) * FT(100)

        aux.rain_w =
            terminal_velocity(param_set, CM1M.RainType(), state.ρ, aux.q_rai)
        aux.snow_w =
            terminal_velocity(param_set, CM1M.SnowType(), state.ρ, aux.q_sno)

        # more diagnostics
        ts_eq = PhaseEquil_ρTq(param_set, state.ρ, aux.T, aux.q_tot)
        q_eq = PhasePartition(ts_eq)

        aux.src_cloud_liq =
            conv_q_vap_to_q_liq_ice(param_set, CM1M.LiquidType(), q_eq, q)
        aux.src_cloud_ice =
            conv_q_vap_to_q_liq_ice(param_set, CM1M.IceType(), q_eq, q)

        aux.src_rain_acnv = conv_q_liq_to_q_rai(param_set, aux.q_liq)
        aux.src_snow_acnv = conv_q_ice_to_q_sno(param_set, q, state.ρ, aux.T)

        aux.src_liq_rain_accr = accretion(
            param_set,
            CM1M.LiquidType(),
            CM1M.RainType(),
            aux.q_liq,
            aux.q_rai,
            state.ρ,
        )
        aux.src_liq_snow_accr = accretion(
            param_set,
            CM1M.LiquidType(),
            CM1M.SnowType(),
            aux.q_liq,
            aux.q_sno,
            state.ρ,
        )
        aux.src_ice_snow_accr = accretion(
            param_set,
            CM1M.IceType(),
            CM1M.SnowType(),
            aux.q_ice,
            aux.q_sno,
            state.ρ,
        )
        aux.src_ice_rain_accr = accretion(
            param_set,
            CM1M.IceType(),
            CM1M.RainType(),
            aux.q_ice,
            aux.q_rai,
            state.ρ,
        )

        aux.src_rain_accr_sink =
            accretion_rain_sink(param_set, aux.q_ice, aux.q_rai, state.ρ)

        if aux.T < _T_freeze
            aux.src_snow_rain_accr = accretion_snow_rain(
                param_set,
                CM1M.SnowType(),
                CM1M.RainType(),
                aux.q_sno,
                aux.q_rai,
                state.ρ,
            )
        else
            aux.src_snow_rain_accr = accretion_snow_rain(
                param_set,
                CM1M.RainType(),
                CM1M.SnowType(),
                aux.q_rai,
                aux.q_sno,
                state.ρ,
            )
        end

        aux.src_rain_evap = evaporation_sublimation(
            param_set,
            CM1M.RainType(),
            q,
            aux.q_rai,
            state.ρ,
            aux.T,
        )
        aux.src_snow_subl = evaporation_sublimation(
            param_set,
            CM1M.SnowType(),
            q,
            aux.q_sno,
            state.ρ,
            aux.T,
        )

        aux.src_snow_melt = snow_melt(param_set, aux.q_sno, state.ρ, aux.T)

        aux.flag_cloud_liq = FT(0)
        aux.flag_cloud_ice = FT(0)
        aux.flag_rain = FT(0)
        aux.flag_snow = FT(0)
        if (aux.q_liq >= FT(0))
            aux.flag_cloud_liq = FT(1)
        end
        if (aux.q_ice >= FT(0))
            aux.flag_cloud_ice = FT(1)
        end
        if (aux.q_rai >= FT(0))
            aux.flag_rain = FT(1)
        end
        if (aux.q_sno >= FT(0))
            aux.flag_snow = FT(1)
        end
    end
end

function boundary_state!(
    ::RusanovNumericalFlux,
    bctype,
    m::KinematicModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    t,
    args...,
)
    # 1 - left     (x = 0,   z = ...)
    # 2 - right    (x = -1,  z = ...)
    # 3,4 - y boundary (periodic)
    # 5 - bottom   (x = ..., z = 0)
    # 6 - top      (x = ..., z = -1)

    FT = eltype(state⁻)
    @inbounds state⁺.ρ = state⁻.ρ
    @inbounds state⁺.ρe = aux⁻.ρe_init
    @inbounds state⁺.ρq_tot = aux⁻.ρq_tot_init
    @inbounds state⁺.ρq_liq = FT(0) #state⁻.ρq_liq
    @inbounds state⁺.ρq_ice = FT(0) #state⁻.ρq_ice
    @inbounds state⁺.ρq_rai = FT(0)
    @inbounds state⁺.ρq_sno = FT(0)

    if bctype == 1
        @inbounds state⁺.ρu = SVector(state⁻.ρu[1], FT(0), FT(0))
    end
    if bctype == 2
        @inbounds state⁺.ρu = SVector(state⁻.ρu[1], FT(0), FT(0))
        @inbounds state⁺.ρe = state⁻.ρe
        @inbounds state⁺.ρq_tot = state⁻.ρq_tot
        @inbounds state⁺.ρq_liq = state⁻.ρq_liq
        @inbounds state⁺.ρq_ice = state⁻.ρq_ice
    end
    if bctype == 5
        @inbounds state⁺.ρu -= 2 * dot(state⁻.ρu, n) .* SVector(n)
    end
    if bctype == 6
        @inbounds state⁺.ρu = SVector(state⁻.ρu[1], FT(0), state⁻.ρu[3])
    end
end

@inline function wavespeed(
    m::KinematicModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    _...,
)
    FT = eltype(state)
    @inbounds begin
        u = state.ρu / state.ρ
        q_rai::FT = state.ρq_rai / state.ρ
        q_sno::FT = state.ρq_sno / state.ρ
        rain_w = terminal_velocity(param_set, CM1M.RainType(), state.ρ, q_rai)
        snow_w = terminal_velocity(param_set, CM1M.SnowType(), state.ρ, q_sno)

        nu =
            nM[1] * u[1] +
            nM[3] * max(u[3], rain_w, snow_w, u[3] - rain_w, u[3] - snow_w)
    end
    return abs(nu)
end

@inline function flux_first_order!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    _...,
)
    FT = eltype(state)
    @inbounds begin
        q_rai::FT = state.ρq_rai / state.ρ
        q_sno::FT = state.ρq_sno / state.ρ
        rain_w = terminal_velocity(param_set, CM1M.RainType(), state.ρ, q_rai)
        snow_w = terminal_velocity(param_set, CM1M.SnowType(), state.ρ, q_sno)

        # advect moisture ...
        flux.ρ = SVector(state.ρu[1], FT(0), state.ρu[3])
        flux.ρq_tot = SVector(
            state.ρu[1] * state.ρq_tot / state.ρ,
            FT(0),
            state.ρu[3] * state.ρq_tot / state.ρ,
        )
        flux.ρq_liq = SVector(
            state.ρu[1] * state.ρq_liq / state.ρ,
            FT(0),
            state.ρu[3] * state.ρq_liq / state.ρ,
        )
        flux.ρq_ice = SVector(
            state.ρu[1] * state.ρq_ice / state.ρ,
            FT(0),
            state.ρu[3] * state.ρq_ice / state.ρ,
        )
        flux.ρq_rai = SVector(
            state.ρu[1] * state.ρq_rai / state.ρ,
            FT(0),
            (state.ρu[3] / state.ρ - rain_w) * state.ρq_rai,
        )
        flux.ρq_sno = SVector(
            state.ρu[1] * state.ρq_sno / state.ρ,
            FT(0),
            (state.ρu[3] / state.ρ - snow_w) * state.ρq_sno,
        )
        # ... energy ...
        flux.ρe = SVector(
            state.ρu[1] / state.ρ * (state.ρe + aux.p),
            FT(0),
            state.ρu[3] / state.ρ * (state.ρe + aux.p),
        )
        # ... and don't advect momentum (kinematic setup)
    end
end

function source!(
    m::KinematicModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)

    _grav::FT = grav(param_set)

    _e_int_v0::FT = e_int_v0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)

    _cv_d::FT = cv_d(param_set)
    _cv_v::FT = cv_v(param_set)
    _cv_l::FT = cv_l(param_set)
    _cv_i::FT = cv_i(param_set)

    _T_0::FT = T_0(param_set)
    _T_freeze = T_freeze(param_set)

    @inbounds begin
        e_tot = state.ρe / state.ρ
        q_tot = state.ρq_tot / state.ρ
        q_liq = state.ρq_liq / state.ρ
        q_ice = state.ρq_ice / state.ρ
        q_rai = state.ρq_rai / state.ρ
        q_sno = state.ρq_sno / state.ρ
        u = state.ρu[1] / state.ρ
        w = state.ρu[3] / state.ρ
        ρ = state.ρ
        e_int = e_tot - 1 // 2 * (u^2 + w^2) - _grav * aux.z_coord

        q = PhasePartition(q_tot, q_liq, q_ice)
        T = air_temperature(param_set, e_int, q)
        _Lf = latent_heat_fusion(param_set, T)
        # equilibrium state at current T
        ts_eq = PhaseEquil_ρTq(param_set, state.ρ, T, q_tot)
        q_eq = PhasePartition(ts_eq)

        # zero out the source terms
        source.ρq_tot = FT(0)
        source.ρq_liq = FT(0)
        source.ρq_ice = FT(0)
        source.ρq_rai = FT(0)
        source.ρq_sno = FT(0)
        source.ρe = FT(0)

        # vapour -> cloud liquid water
        source.ρq_liq +=
            ρ * conv_q_vap_to_q_liq_ice(param_set, CM1M.LiquidType(), q_eq, q)
        # vapour -> cloud ice
        source.ρq_ice +=
            ρ * conv_q_vap_to_q_liq_ice(param_set, CM1M.IceType(), q_eq, q)

        ## cloud liquid water -> rain
        acnv = ρ * conv_q_liq_to_q_rai(param_set, q_liq)
        source.ρq_liq -= acnv
        source.ρq_tot -= acnv
        source.ρq_rai += acnv
        source.ρe -= acnv * (_cv_l - _cv_d) * (T - _T_0)

        ## cloud ice -> snow
        acnv = ρ * conv_q_ice_to_q_sno(param_set, q, state.ρ, T)
        source.ρq_ice -= acnv
        source.ρq_tot -= acnv
        source.ρq_sno += acnv
        source.ρe -= acnv * ((_cv_i - _cv_d) * (T - _T_0) - _e_int_i0)

        # cloud liquid water + rain -> rain
        accr =
            ρ * accretion(
                param_set,
                CM1M.LiquidType(),
                CM1M.RainType(),
                q_liq,
                q_rai,
                state.ρ,
            )
        source.ρq_liq -= accr
        source.ρq_tot -= accr
        source.ρq_rai += accr
        source.ρe -= accr * (_cv_l - _cv_d) * (T - _T_0)

        # cloud ice + snow -> snow
        accr =
            ρ * accretion(
                param_set,
                CM1M.IceType(),
                CM1M.SnowType(),
                q_ice,
                q_sno,
                state.ρ,
            )
        source.ρq_ice -= accr
        source.ρq_tot -= accr
        source.ρq_sno += accr
        source.ρe -= accr * ((_cv_i - _cv_d) * (T - _T_0) - _e_int_i0)

        # cloud liquid water + snow -> snow or rain
        accr =
            ρ * accretion(
                param_set,
                CM1M.LiquidType(),
                CM1M.SnowType(),
                q_liq,
                q_sno,
                state.ρ,
            )
        if T < _T_freeze
            source.ρq_liq -= accr
            source.ρq_tot -= accr
            source.ρq_sno += accr
            source.ρe -= accr * ((_cv_i - _cv_d) * (T - _T_0) - _e_int_i0)
        else
            source.ρq_liq -= accr
            source.ρq_tot -= accr
            source.ρq_sno -= accr * (_cv_l / _Lf * (T - _T_freeze))
            source.ρq_rai += accr * (FT(1) + _cv_l / _Lf * (T - _T_freeze))
            source.ρe +=
                -accr * ((_cv_l - _cv_d) * (T - _T_0) + _cv_l * (T - _T_freeze))
        end

        # cloud ice + rain -> snow
        accr =
            ρ * accretion(
                param_set,
                CM1M.IceType(),
                CM1M.RainType(),
                q_ice,
                q_rai,
                state.ρ,
            )
        accr_rain_sink =
            ρ * accretion_rain_sink(param_set, q_ice, q_rai, state.ρ)
        source.ρq_ice -= accr
        source.ρq_tot -= accr
        source.ρq_rai -= accr_rain_sink
        source.ρq_sno += accr + accr_rain_sink
        source.ρe +=
            accr_rain_sink * _Lf -
            accr * ((_cv_i - _cv_d) * (T - _T_0) - _e_int_i0)

        # rain + snow -> snow or rain
        if T < _T_freeze
            accr =
                ρ * accretion_snow_rain(
                    param_set,
                    CM1M.SnowType(),
                    CM1M.RainType(),
                    q_sno,
                    q_rai,
                    state.ρ,
                )
            source.ρq_sno += accr
            source.ρq_rai -= accr
            source.ρe += accr * _Lf
        else
            accr =
                ρ * accretion_snow_rain(
                    param_set,
                    CM1M.RainType(),
                    CM1M.SnowType(),
                    q_rai,
                    q_sno,
                    state.ρ,
                )
            source.ρq_rai += accr
            source.ρq_sno -= accr
            source.ρe -= accr * _Lf
        end

        # rain -> vapour
        evap =
            ρ * evaporation_sublimation(
                param_set,
                CM1M.RainType(),
                q,
                q_rai,
                state.ρ,
                T,
            )
        source.ρq_rai += evap
        source.ρq_tot -= evap
        source.ρe -= evap * (_cv_l - _cv_d) * (T - _T_0)

        # snow -> vapour
        subl =
            ρ * evaporation_sublimation(
                param_set,
                CM1M.SnowType(),
                q,
                q_sno,
                state.ρ,
                T,
            )
        source.ρq_sno += subl
        source.ρq_tot -= subl
        source.ρe -= subl * ((_cv_i - _cv_d) * (T - _T_0) - _e_int_i0)

        # snow -> rain
        melt = ρ * snow_melt(param_set, q_sno, state.ρ, T)

        source.ρq_sno -= melt
        source.ρq_rai += melt
        source.ρe -= melt * _Lf
    end
end

function main()
    # Working precision
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δx = FT(500)
    Δy = FT(1)
    Δz = FT(250)
    resolution = (Δx, Δy, Δz)
    # Domain extents
    xmax = 90000
    ymax = 10
    zmax = 16000
    # initial configuration
    wmax = FT(0.6)  # max velocity of the eddy  [m/s]
    θ_0 = FT(289) # init. theta value (const) [K]
    p_0 = FT(101500) # surface pressure [Pa]
    p_1000 = FT(100000) # reference pressure in theta definition [Pa]
    qt_0 = FT(7.5 * 1e-3) # init. total water specific humidity (const) [kg/kg]
    z_0 = FT(0) # surface height

    # time stepping
    t_ini = FT(0)
    t_end = FT(5 * 60) #FT(4 * 60 * 60) #TODO
    dt = FT(0.25)
    #CFL = FT(1.75)
    filter_freq = 1
    output_freq = 1200
    interval = "1200steps"

    # periodicity and boundary numbers
    periodicity_x = false
    periodicity_y = true
    periodicity_z = false
    idx_bc_left = 1
    idx_bc_right = 2
    idx_bc_front = 3
    idx_bc_back = 4
    idx_bc_bottom = 5
    idx_bc_top = 6


    #! format: off
    z_range = [
        0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500,
        2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
        5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300,
        7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700,
        9800, 9900, 10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900, 11000, 11100, 11200, 11300, 11400, 11500, 11600, 11700,
        11800, 11900, 12000, 12100, 12200, 12300, 12400, 12500, 12600, 12700, 12800, 12900, 13000, 13100, 13200, 13300, 13400, 13500, 13600, 13700,
        13800, 13900, 14000, 14100, 14200, 14300, 14400, 14500, 14600, 14700, 14800, 14900, 15000, 15100, 15200, 15300, 15400, 15500, 15600, 15700,
        15800, 15900, 16000, 16100, 16200, 16300, 16400, 16500, 16600, 16700, 16800, 16900, 17000]

    T_range = [
        299.184, 297.628892697526, 296.498576625316, 295.708541362848, 295.174276489603, 294.811271585061, 294.5350162287, 294.261,
        293.920421043582, 293.507311764631, 293.031413133474, 292.502466120435, 291.930211695841, 291.324390830018, 290.694744493293, 290.051013655991,
        289.402939288437, 288.760262346828, 288.130849318341, 287.515505258001, 286.913372692428, 286.323594148241, 285.74531215206, 285.177669230506,
        284.619807910198, 284.070870717755, 283.530000179798, 282.996338822946, 282.469029173819, 281.947213759037, 281.43003510522, 280.916635738988,
        280.406158186959, 279.897744975755, 279.390538631995, 278.883681682298, 278.376316653285, 277.867586071576, 277.356632463789, 276.842598356546,
        276.324685598782, 275.802667484647, 275.276636664165, 274.746689276312, 274.212921460063, 273.675429354393, 273.134309098278, 272.589656830692,
        272.041568690611, 271.49014081701, 270.935469348864, 270.377650425148, 269.816780184838, 269.252954766908, 268.686270310335, 268.116822954093,
        267.544708837157, 266.970024098502, 266.392864877104, 265.813327311938, 265.231507541979, 264.64747745881, 264.060960130126, 263.47141707779,
        262.878303486873, 262.281074542448, 261.679185429586, 261.072091333358, 260.459247438837, 259.840108931094, 259.214130995201, 258.580768816228,
        257.939477579249, 257.289712469334, 256.630928671556, 255.962581370986, 255.284125752695, 254.595017001755, 253.894710327766, 253.182887991267,
        252.46001426264, 251.726721646298, 250.983642646651, 250.231409768111, 249.470655515088, 248.702012391995, 247.926112903242, 247.14358955324,
        246.355074846402, 245.561201287137, 244.762601379858, 243.959907628976, 243.153728314245, 242.344349049499, 241.531825963971, 240.716210279609,
        239.897553218363, 239.075906002182, 238.251319853016, 237.423845992814, 236.593535643524, 235.760440027097, 234.924610365481, 234.086097880626,
        233.244953794481, 232.401229328996, 231.554975706119, 230.7062441478, 229.855085875989, 229.001552112634, 228.145694079684, 227.287574051742,
        226.42757986392, 225.566470305038, 224.705024343724, 223.84402094861, 222.984239088323, 222.126457731494, 221.271455846752, 220.420012402726,
        219.572906368047, 218.730916711342, 217.894822401242, 217.065402406377, 216.243435695375, 215.429701236865, 214.624977999479, 213.830040741192,
        213.045586818019, 212.272245655601, 211.51064437337, 210.761410090759, 210.0251699272, 209.302551002126, 208.594180434969, 207.90068534516,
        207.222682432974, 206.560708873131, 205.915265185628, 205.286851694754, 204.675968724799, 204.083116600052, 203.508795644802, 202.95350618334,
        202.417748539953, 201.902023577843, 201.407120974388, 200.934594512862, 200.486122895611, 200.063384824981, 199.668059003318, 199.301824132969,
        198.96635891628, 198.663342055596, 198.394452253265, 198.161279286222, 197.963547643009, 197.799211358657, 197.666154525954, 197.562261237686,
        197.485415586642, 197.433501665607, 197.40440356737, 197.396005384718, 197.406191210439, 197.432899562191, 197.47476703369, 197.530913717737,
        197.600469393593, 197.682563840515, 197.776326837764]

    q_range = [
        0.0162321692080669, 0.0167673003973785, 0.0169842024326598, 0.0169337519538153, 0.0166663497239751, 0.0162321692080669, 0.015681352854829,
        0.0150641570577189, 0.0144310460364482, 0.0138327343680399, 0.013320177602368, 0.012928720727755, 0.0126305522364067, 0.0123819207241334,
        0.0121389587958033, 0.0118577075098814, 0.0115066332638913, 0.0111042588941075, 0.010681699769686, 0.0102701709364724, 0.0099009900990099,
        0.00959624936114117, 0.00934081059469477, 0.00911016542683696, 0.00887975233624587, 0.0086249628234361, 0.00832855696312967, 0.0080029519599734,
        0.00766802230757524, 0.00734369421478042, 0.007049945387747, 0.00680137732291751, 0.00659092388806814, 0.00640607314371358, 0.00623428890344645,
        0.00606301560481065, 0.00588287130354119, 0.005697230175774, 0.00551266483334836, 0.00533575758863248, 0.00517309988062077, 0.00502869447640062,
        0.004896158939909, 0.00476650506495816, 0.00463073495294506, 0.00447984071677452, 0.00430803604057719, 0.00412246330674869, 0.0039335098478028,
        0.00375157835183152, 0.00358708648864089, 0.00344726315880705, 0.00332653452987796, 0.00321611598159287, 0.00310721394018452, 0.00299102691924227,
        0.00286116318208214, 0.00272089896736183, 0.00257593295768495, 0.00243197016394643, 0.00229472213908012, 0.00216860732664965, 0.00205285001385606,
        0.00194537329026098, 0.00184409796137261, 0.00174694285001248, 0.0016522111336286, 0.00155974942036123, 0.00146979043510184, 0.0013825670343689,
        0.00129831219414761, 0.00121701485317001, 0.001137687161242, 0.00105909638608903, 0.000980009009141718,  0.00089919072834449, 0.000815997019873583,
        0.000732145660692171, 0.00064994635423271, 0.000571710312185518, 0.000499750124937531, 0.000435858502312957, 0.000379744409038409,
        0.000330595173284745, 0.000287597487595375, 0.000249937515621095, 0.000216895300299521, 0.000188127832006747, 0.000163386230957203,
        0.000142421466578296, 0.000124984376952881, 0.000110689772327441, 9.86086827423503e-05, 8.7676088112503e-05, 7.68268601082915e-05,
        6.49957752746072e-05, 5.15108887771554e-05, 3.72737503739521e-05, 2.35794100864691e-05, 1.17230575137195e-05, 2.999991000027e-06,
        1.69619792857815e-06, 3.07897588648539e-06, 2.26364535967942e-06, 3.65542843474416e-07, 1.49999775000338e-06, 2.45283190408077e-06,
        2.55351934700394e-06, 2.09779045350945e-06, 1.38137398580681e-06, 6.99999510000343e-07, 2.86443287526343e-07, 1.21662921693633e-07,
        1.23660803334821e-07, 2.10439089554136e-07, 2.99999910000027e-07, 3.27788493770303e-07, 2.99022297121994e-07, 2.36361815382337e-07,
        1.62467544245733e-07, 9.9999990000001e-08, 6.64022211520834e-08, 5.82475139663064e-08, 6.68916919016523e-08, 8.36905765913686e-08,
        9.9999990000001e-08, 1.09002484657468e-07, 1.11187531664659e-07, 1.0887133139317e-07, 1.04370084083517e-07, 9.9999990000001e-08,
        9.75877787592422e-08, 9.70022971221656e-08, 9.76229211090621e-08, 9.88290267308115e-08, 9.9999990000001e-08, 1.00646340200832e-07,
        1.00803219684934e-07, 1.0063692406888e-07, 1.00313748968563e-07, 9.9999990000001e-08, 9.98268004299175e-08, 9.97847641264901e-08,
        9.9829322608123e-08, 9.99159173931719e-08, 9.9999990000001e-08, 1.00046398078965e-07, 1.00057663808278e-07, 1.0004572549811e-07,
        1.00022521458629e-07, 9.9999990000001e-08, 9.99875472541912e-08, 9.99845206403455e-08, 9.99877153994048e-08, 9.99939367723098e-08,
        9.9999990000001e-08, 1.00003352904274e-07, 1.00004193630342e-07, 1.00003352904274e-07, 1.00001671452137e-07, 9.9999990000001e-08,
        9.99989811287191e-08, 9.99986448382919e-08, 9.99988129835055e-08, 9.99993174191465e-08, 9.9999990000001e-08, 1.00000662580856e-07,
        1.00001167016497e-07, 1.0000133516171e-07, 1.00000998871283e-07, 9.9999990000001e-08]

    p_range = [
        101500, 100315.025749516, 99139.9660271538, 97974.7710180798, 96819.3909074598, 95673.7758804597, 94537.876176083, 93411.6422486837,
        92295.0246064536, 91187.9737719499, 90090.4404029628, 89002.3752464385, 87923.7291106931, 86854.4528968052, 85794.49764766, 84743.814421206,
        83702.3543252787, 82670.0686672607, 81646.9088044215, 80632.826109327, 79627.7720985403, 78631.6983820215, 77644.5566288927, 76666.2986022276,
        75696.8762101083, 74736.2413760207, 73784.3460742183, 72841.1424820254, 71906.5828275339, 70980.6193544154, 70063.204453007, 69154.2906087673,
        68253.8303673938, 67361.7763702768, 66478.0814065118, 65602.6982808849, 64735.579849904, 63876.6791769641, 63025.949377182, 62183.3435815507,
        61348.8150705175, 60522.317221463, 59703.8034731598, 58893.2273619179, 58090.542574603, 57295.7028140736, 56508.6618359204, 55729.3736066619,
        54957.7921455483, 54193.8714880201, 53437.5658219265, 52688.8294339685, 51947.6166734621, 51213.8819892159, 50487.5799836161, 49768.6652753625,
        49057.0925369589, 48352.8166561238, 47655.7925743794, 46965.9752497711, 46283.3197958903, 45607.7814272171, 44939.3154221458, 44277.8771606411,
        43623.4221794563, 42975.9060319998, 42335.2843266238, 41701.5128914544, 41074.5476095614, 40454.3443808926, 39840.8592642808, 39234.0484216161,
        38633.868080086, 38040.274570662, 37453.2243845259, 36872.6740298782, 36298.5800710783, 35730.8992971221, 35169.5885531642, 34614.6047016162,
        34065.9047673383, 33523.4458805626, 32987.1852382959, 32457.0801436939, 31933.0880637786, 31415.1664829788, 30903.2729431824, 30397.3652161132,
        29897.4011309542, 29403.3385345499, 28915.135440009, 28432.7499682896, 27956.1403087067, 27485.2647592591, 27020.0817857294, 26560.5498717234,
        26106.6275597007, 25658.2736275352, 25215.4469119545, 24778.106267783, 24346.2107202062, 23919.7194049202, 23498.5915276787, 23082.7864056438,
        22672.2635279751, 22266.9824021027, 21866.9025958119, 21471.9839183077, 21082.1862391501, 20697.469446465, 20317.7936031548, 19943.1188855014,
        19573.4055416778, 19208.6139342067, 18848.7046021555, 18493.6381033452, 18143.3750575744, 17797.8763325512, 17457.1028579612, 17121.0155825635,
        16789.575634671, 16462.7442590809, 16140.4827744667, 15822.752617039, 15509.5154044811, 15200.7327737537, 14896.3664255555, 14596.378315538,
        14300.7304630906, 14009.3849072278, 13722.3038717104, 13439.4497001572, 13160.7848122237, 12886.2717485712, 12615.8732366994, 12349.552023955,
        12087.2709233428, 11828.9930105004, 11574.6814267238, 11324.2993335354, 11077.8100828701, 10835.1771502091, 10596.3640894538, 10361.3345792892,
        10130.0524910676, 9902.48171660711, 9678.58621552163, 9458.33021860609, 9241.67802445076, 9028.59395243382, 8819.04251762682, 8612.98836336888,
        8410.3962187252, 8211.2309379189, 8015.45756519932, 7823.04116500174, 7633.94687886457, 7448.14015673894, 7265.58652567913, 7086.25151273945,
        6910.10064497419, 6910.10064497419, 6910.10064497419, 6910.10064497419, 6910.10064497419, 6910.10064497419, 6910.10064497419, 6910.10064497419,
        6910.10064497419, 6910.10064497419, 6910.10064497419 ]

    ρ_range = [
        1.17051362179725, 1.16251824590627, 1.15313016112624, 1.1426566568231, 1.13140764159872, 1.1196895485888, 1.10780103484676, 1.0960305075832,
        1.08459745274435, 1.07348344546654, 1.06261389733783, 1.05192695939713, 1.04140273218196, 1.03103180334705, 1.02080494033636, 1.01071303046314,
        1.00073949487493, 0.990838128123433, 0.980963129632192, 0.971094063252625, 0.961216822759604, 0.951323251200104, 0.941426814080701,
        0.931545806522527, 0.921697879536919, 0.911900034022987, 0.902164578162167, 0.892487313537494, 0.882860249793038, 0.873275726601859,
        0.863726411598162, 0.854208105591224, 0.844727922437057, 0.8352954667279, 0.825919969478444, 0.816610286976537, 0.807373343850795,
        0.798209593686815, 0.789117941453129, 0.780097337389561, 0.771146612530646, 0.762264270399455, 0.753452744759353, 0.74471551136303,
        0.736055874178972, 0.72747696742561, 0.718980350842218, 0.7105619137831, 0.70221632469523, 0.693938467344086, 0.685723438615901,
        0.677567861578885, 0.669473709625074, 0.661444143644113, 0.653482167320322, 0.645590627902435, 0.637771283999691, 0.630022091606339,
        0.622340165988717, 0.614722736979947, 0.607167148420651, 0.599671383580347, 0.592236161810936, 0.584863205035121, 0.577554164218497,
        0.570310604494631, 0.563133875176496, 0.556024730482117, 0.548983747911074, 0.542011464088805, 0.535108377645103, 0.528275027848979,
        0.521512222336075, 0.514820790002398, 0.508201503057759, 0.501655076786909, 0.495181994058148, 0.488781994213898, 0.482454636660537,
        0.476199055919507, 0.470012952419531, 0.463893943289264, 0.457840307961496, 0.451850519497349, 0.445923093407251, 0.440056584487831,
        0.434249561420135, 0.428500536139493, 0.422808039743611, 0.41717064648228, 0.411586974922854, 0.406055719826046, 0.40057578774956,
        0.395146674561446, 0.389768253950893, 0.384440397458089, 0.37916287643901, 0.373935102771167, 0.36875641578014, 0.363626174398787,
        0.358543758947717, 0.353508656205969, 0.348520707739897, 0.343579834253573, 0.338685947906162, 0.333838950271578, 0.329038686145689,
        0.324284810799633, 0.319576938026155, 0.314914687777366, 0.310297688380068, 0.305725572307512, 0.301197591585183, 0.296712545171569,
        0.292269251091959, 0.287866570849034, 0.283503406532332, 0.279178694598262, 0.27489141222907, 0.270640579366203, 0.266425260909222,
        0.262244566294465, 0.258097650982816, 0.25398371429053, 0.249901999425458, 0.245851790781137, 0.241832414286703, 0.237843244370408,
        0.233883781756254, 0.229953637104228, 0.226052458592054, 0.222179928272777, 0.218335761470863, 0.214519707224664, 0.210731549222293,
        0.206971102905131, 0.203238226271481, 0.199532888953038, 0.195855122091922, 0.192204980437394, 0.188582544697456, 0.184987920053912,
        0.181421234993557, 0.177882641628151, 0.174372316360016, 0.170890455921146, 0.167437037955477, 0.164011457709294, 0.160613093688105,
        0.157241413931045, 0.15389597846219, 0.150576437441233, 0.147282529608045, 0.144014082002788, 0.140771009961674, 0.137553374518383,
        0.134362580606833, 0.131201172349935, 0.128071588633207, 0.124976110750845, 0.121916861907264, 0.121948919193911, 0.121966894890357,
        0.121972083953105, 0.121965790399749, 0.121949291096701, 0.121923436104917, 0.121888780280552, 0.121845875238272, 0.121795274570523,
        0.121737533130168]

    dρ_range = [
        -3.59937517425616e-05, -8.74236524613322e-05, -9.98245979099523e-05, -0.00010912795442958, -0.000115340910679925, -0.000118522777615024,
        -0.000118766703148947, -0.000116181583122055, -0.00011260864631165, -0.000109795778726265, -0.000107714951035596, -0.000106039943861968,
        -0.000104460270820561, -0.000102973683474333, -0.000101578762228199, -0.000100274450938608, -9.92861102161202e-05, -9.88281212364185e-05,
        -9.86963271464388e-05, -9.87085122902761e-05, -9.88588230322522e-05, -9.8981065311496e-05, -9.89171829920543e-05, -9.86735488551799e-05,
        -9.82566574041453e-05, -9.76729921159812e-05, -9.70500064028184e-05, -9.65086017246105e-05, -9.60454400625446e-05, -9.56572439955241e-05,
        -9.5340728706457e-05, -9.5008763743794e-05, -9.45788919208772e-05, -9.40548370115828e-05, -9.3440348194004e-05, -9.27391611164088e-05,
        -9.20016033219379e-05, -9.12752248054719e-05, -9.05595697070144e-05, -8.98541831357713e-05, -8.91654815701201e-05, -8.84752762943482e-05,
        -8.77494447742739e-05, -8.69897168393693e-05, -8.61978039593136e-05, -8.53753836002353e-05, -8.45662005091732e-05, -8.38114256450538e-05,
        -8.31088828495735e-05, -8.24564348214017e-05, -8.1851961718636e-05, -8.12540449384666e-05, -8.06237270285041e-05, -7.99625864890717e-05,
        -7.92721952020196e-05, -7.85541037865398e-05, -7.78377768942751e-05, -7.71508783726086e-05, -7.64922509147597e-05, -7.58607550079355e-05,
        -7.5255257575603e-05, -7.46578263440909e-05, -7.40437160173945e-05, -7.34126676822411e-05, -7.27655403821961e-05, -7.21031866632626e-05,
        -7.14303672571544e-05, -7.07515642299482e-05, -7.00671933172211e-05, -6.93776513635009e-05, -6.86833098465315e-05, -6.79822059852417e-05,
        -6.72725239627913e-05, -6.65548356116998e-05, -6.58297114361597e-05, -6.50977145039132e-05, -6.43646771280071e-05, -6.36360540138981e-05,
        -6.29118551436968e-05, -6.22041292322341e-05, -6.15221570675246e-05, -6.08606412866085e-05, -6.02146109367996e-05, -5.95836319123706e-05,
        -5.89672983766289e-05, -5.8365225270403e-05, -5.7777757303502e-05, -5.72051962789751e-05, -5.66471081172719e-05, -5.61030583593601e-05,
        -5.55726043916327e-05, -5.50543285425413e-05, -5.45449966705289e-05, -5.40374636372204e-05, -5.35311627048121e-05, -5.30261985318617e-05,
        -5.25253559659224e-05, -5.20312184290027e-05, -5.15435901963297e-05, -5.10622691485893e-05, -5.05870395077161e-05, -5.01151309694529e-05,
        -4.96439716171912e-05, -4.91736441117515e-05, -4.87042469224382e-05, -4.82358875773372e-05, -4.77700486961607e-05, -4.73081018750182e-05,
        -4.68499874419861e-05, -4.63956358750055e-05, -4.59449597549036e-05, -4.54985344751462e-05, -4.50631284786667e-05, -4.46397691588802e-05,
        -4.42280098023119e-05, -4.38274214139499e-05, -4.34376409961601e-05, -4.30583023124663e-05, -4.26889789379916e-05, -4.23292356052957e-05,
        -4.19786196104028e-05, -4.16366783570754e-05, -4.13029635571471e-05, -4.09770310849877e-05, -4.06584588726149e-05, -4.03468387207095e-05,
        -4.00417403212974e-05, -3.9742477928175e-05, -3.94474194224351e-05, -3.91560587225531e-05, -3.88680420136295e-05, -3.85830382145498e-05,
        -3.83007173685112e-05, -3.80207275262849e-05, -3.77427401195186e-05, -3.74664417323129e-05, -3.71911016163793e-05, -3.69155924693975e-05,
        -3.66396540346307e-05, -3.6363043157771e-05, -3.60854967755032e-05, -3.58067813502568e-05, -3.55266676411338e-05, -3.52449049917705e-05,
        -3.49612716252254e-05, -3.46756639284376e-05, -3.43938806523058e-05, -3.41187600097204e-05, -3.38494075224641e-05, -3.35849193859056e-05,
        -3.33243734832797e-05, -3.3066881627664e-05, -3.28115645135523e-05, -3.25575278845173e-05, -3.23039138342865e-05, -3.20463462400501e-05,
        -3.17652064416084e-05, -3.14588936623928e-05, -3.11289784299643e-05, -3.07770405929711e-05, -1.50046969840055e-05, -2.48015587883801e-07,
        -1.13657079374249e-07, -7.70202792434148e-09, -1.15984516793898e-07, -2.12713494118585e-07, -3.03471417808635e-07, -3.88725590955189e-07,
        -4.68452477358432e-07, -5.42635999340642e-07, -6.1126695981267e-07]
    #! format: on

    init_T = Spline1D(z_range, T_range)
    init_qt = Spline1D(z_range, q_range)
    init_p = Spline1D(z_range, p_range)
    init_ρ = Spline1D(z_range, ρ_range)
    init_dρ = Spline1D(z_range, dρ_range)

    driver_config, ode_solver_type = config_kinematic_eddy(
        FT,
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        wmax,
        θ_0,
        p_0,
        p_1000,
        qt_0,
        z_0,
        periodicity_x,
        periodicity_y,
        periodicity_z,
        idx_bc_left,
        idx_bc_right,
        idx_bc_front,
        idx_bc_back,
        idx_bc_bottom,
        idx_bc_top,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t_ini,
        t_end,
        driver_config,
        (init_T, init_qt, init_p, init_ρ, init_dρ),
        ode_solver_type = ode_solver_type,
        ode_dt = dt,
        init_on_cpu = true,
        #Courant_number = CFL,
    )

    model = driver_config.bl

    mpicomm = MPI.COMM_WORLD

    # get state variables indices for filtering
    ρq_liq_ind = varsindex(vars_state(model, Prognostic(), FT), :ρq_liq)
    ρq_ice_ind = varsindex(vars_state(model, Prognostic(), FT), :ρq_ice)
    ρq_rai_ind = varsindex(vars_state(model, Prognostic(), FT), :ρq_rai)
    ρq_sno_ind = varsindex(vars_state(model, Prognostic(), FT), :ρq_sno)
    # get aux variables indices for testing
    q_tot_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_tot)
    q_vap_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_vap)
    q_liq_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_liq)
    q_ice_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_ice)
    q_rai_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_rai)
    q_sno_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_sno)
    S_liq_ind = varsindex(vars_state(model, Auxiliary(), FT), :S_liq)
    S_ice_ind = varsindex(vars_state(model, Auxiliary(), FT), :S_ice)
    rain_w_ind = varsindex(vars_state(model, Auxiliary(), FT), :rain_w)
    snow_w_ind = varsindex(vars_state(model, Auxiliary(), FT), :snow_w)

    # filter out negative values
    cb_tmar_filter =
        GenericCallbacks.EveryXSimulationSteps(filter_freq) do (init = false)
            Filters.apply!(
                solver_config.Q,
                (:ρq_tot, :ρq_liq, :ρq_ice, :ρq_rai, :ρq_sno),
                solver_config.dg.grid,
                TMARFilter(),
            )
            nothing
        end
    cb_boyd_filter =
        GenericCallbacks.EveryXSimulationSteps(filter_freq) do (init = false)
            Filters.apply!(
                solver_config.Q,
                (:ρq_tot, :ρq_liq, :ρq_ice, :ρq_rai, :ρq_sno, :ρe, :ρ),
                solver_config.dg.grid,
                BoydVandevenFilter(solver_config.dg.grid, 1, 8),
            )
        end

    # output for paraview

    # initialize base output prefix directory from rank 0
    vtkdir = abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk"))
    if MPI.Comm_rank(mpicomm) == 0
        mkpath(vtkdir)
    end
    MPI.Barrier(mpicomm)

    vtkstep = [0]
    cb_vtk =
        GenericCallbacks.EveryXSimulationSteps(output_freq) do (init = false)
            out_dirname = @sprintf(
                "microphysics_test_4_mpirank%04d_step%04d",
                MPI.Comm_rank(mpicomm),
                vtkstep[1]
            )
            out_path_prefix = joinpath(vtkdir, out_dirname)
            @info "doing VTK output" out_path_prefix
            writevtk(
                out_path_prefix,
                solver_config.Q,
                solver_config.dg,
                flattenednames(vars_state(model, Prognostic(), FT)),
                solver_config.dg.state_auxiliary,
                flattenednames(vars_state(model, Auxiliary(), FT)),
            )
            vtkstep[1] += 1
            nothing
        end

    # output for netcdf
    boundaries = [
        FT(0) FT(0) FT(0)
        xmax ymax zmax
    ]
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )
    dgngrps = [
        setup_dump_state_diagnostics(
            AtmosLESConfigType(),
            interval,
            driver_config.name,
            interpol = interpol,
        ),
        setup_dump_aux_diagnostics(
            AtmosLESConfigType(),
            interval,
            driver_config.name,
            interpol = interpol,
        ),
    ]
    dgn_config = ClimateMachine.DiagnosticsConfiguration(dgngrps)

    # call solve! function for time-integrator
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cb_boyd_filter, cb_tmar_filter),
        check_euclidean_distance = true,
    )

    max_q_tot = maximum(abs.(solver_config.dg.state_auxiliary[:, q_tot_ind, :]))
    @test !isnan(max_q_tot)

    max_q_vap = maximum(abs.(solver_config.dg.state_auxiliary[:, q_vap_ind, :]))
    @test !isnan(max_q_vap)

    max_q_liq = maximum(abs.(solver_config.dg.state_auxiliary[:, q_liq_ind, :]))
    @test !isnan(max_q_liq)

    max_q_ice = maximum(abs.(solver_config.dg.state_auxiliary[:, q_ice_ind, :]))
    @test !isnan(max_q_ice)

    max_q_rai = maximum(abs.(solver_config.dg.state_auxiliary[:, q_rai_ind, :]))
    @test !isnan(max_q_rai)

    max_q_sno = maximum(abs.(solver_config.dg.state_auxiliary[:, q_sno_ind, :]))
    @test !isnan(max_q_sno)

end

main()
