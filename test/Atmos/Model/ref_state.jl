using CLIMAParameters
using CLIMAParameters.Planet: R_d, grav, MSLP
using StaticArrays

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Atmos: AtmosModel, DryModel, HydrostaticState
using ClimateMachine.Atmos: atmos_init_aux!, atmos_init_ref_state_pressure!
using ClimateMachine.Orientations: orientation_nodal_init_aux!
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
using ClimateMachine.BalanceLaws:
    BalanceLaw, vars_state, number_states, Auxiliary
using ClimateMachine.DGMethods: LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.VariableTemplates

using Test

@testset "Hydrostatic reference states" begin
    # We should provide an interface to call all physics
    # kernels in some way similar to this:
    function compute_ref_state(z::FT, atmos) where {FT}
        local_geom = LocalGeometry(5, SVector{3, FT}(0, 0, z), @SMatrix zeros(
            FT,
            3,
            3,
        ))

        st = vars_state(atmos, Auxiliary(), FT)
        nst = number_states(atmos, Auxiliary())
        arr = MArray{Tuple{nst}, FT}(undef)
        fill!(arr, 0)
        aux = Vars{st}(arr)

        # Hack: need coord in sync with incoming z, so that
        # altitude returns correct value.
        aux.coord = @SArray FT[0, 0, z]
        # Need orientation defined, so that z
        orientation_nodal_init_aux!(
            atmos.orientation,
            atmos.param_set,
            aux,
            local_geom,
        )
        aux.orientation.∇Φ = @SArray FT[0, 0, grav(param_set)]

        atmos_init_ref_state_pressure!(atmos.ref_state, atmos, aux, local_geom)

        # Mega-hack: define ∇p so that we actually compute ρ based
        # on the ideal gas law
        T_virt, p = atmos.ref_state.virtual_temperature_profile(param_set, z)
        hack_value = FT(-grav(param_set) * p / (R_d(param_set) * T_virt))
        tmp = Vars{@vars(∇p::SVector{3, FT})}(SVector{3, FT}(0, 0, hack_value))

        atmos_init_aux!(atmos.ref_state, atmos, aux, tmp, local_geom)
        return aux
    end

    FT = Float64
    RH = FT(0.5)
    profile = DecayingTemperatureProfile{FT}(param_set)
    m = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        init_state_prognostic = x -> x,
        ref_state = HydrostaticState(profile, RH),
        moisture = DryModel(),
    )

    z = collect(range(FT(0), stop = FT(25e3), length = 100))
    phase_type = PhaseEquil

    aux_arr = compute_ref_state.(z, Ref(m))
    T = map(x -> x.ref_state.T, aux_arr)
    p = map(x -> x.ref_state.p, aux_arr)
    ρ = map(x -> x.ref_state.ρ, aux_arr)
    q_tot = map(x -> x.ref_state.ρq_tot, aux_arr) ./ ρ
    q_pt = PhasePartition.(q_tot)

    # TODO: test that ρ and p are in discrete hydrostatic balance

    # Test state for thermodynamic consistency (with ideal gas law)
    @test all(
        T .≈ air_temperature_from_ideal_gas_law.(Ref(param_set), p, ρ, q_pt),
    )

    # Test that relative humidity in reference state is approximately
    # input relative humidity
    RH_ref = relative_humidity.(Ref(param_set), T, p, Ref(phase_type), q_pt)
    @test all(isapprox.(RH, RH_ref, atol = 0.05))

end
