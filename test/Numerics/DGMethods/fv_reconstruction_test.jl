using Test
using StaticArrays
using ClimateMachine.Atmos:
    AtmosProblem,
    NoReferenceState,
    AtmosModel,
    DryModel,
    ConstantDynamicViscosity,
    AtmosLESConfigType,
    HBFVReconstruction
import ClimateMachine.DGMethods.FVReconstructions: FVConstant, FVLinear, width
using ClimateMachine.Orientations
import StaticArrays: SUnitRange
import ClimateMachine.BalanceLaws:
    Primitive, Prognostic, vars_state, number_states
using ClimateMachine.VariableTemplates: Vars
using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()



# lin_func, quad_func, third_func, fourth_func
#
# ```
# num_state_primitive::Int64
# pointwise values::Array{FT,   num_state_primitive by length(ξ)}
# integration values::Array{FT, num_state_primitive by length(ξ)}
# ```

function lin_func(ξ)
    return 1, [(2 * ξ .+ 1)';], [(ξ .^ 2 .+ ξ)';]
end

function quad_func(ξ)
    return 2,
    [(3 * ξ .^ 2 .+ 1)'; (2 * ξ .+ 1)'],
    [(ξ .^ 3 .+ ξ)'; (ξ .^ 2 .+ ξ)']
end

function third_func(ξ)
    return 1, [(4 * ξ .^ 3 .+ 1)';], [(ξ .^ 4 .+ ξ)';]
end

function fourth_func(ξ)
    return 2,
    [(5 * ξ .^ 4 .+ 1)'; (3 * ξ .^ 2 .+ 1)'],
    [(ξ .^ 5 .+ ξ)'; (ξ .^ 3 .+ ξ)']
end


@testset "Hydrostatic balanced linear reconstruction test" begin

    function initialcondition!(problem, bl, state, aux, coords, t, args...) end

    fv_recon! = FVLinear()
    stencil_width = width(fv_recon!)
    stencil_center = stencil_width + 1
    stencil_diameter = 2stencil_width + 1
    @test stencil_width == 1
    func = lin_func

    for FT in (Float64,)
        model = AtmosModel{FT}(
            AtmosLESConfigType,
            param_set;
            problem = AtmosProblem(init_state_prognostic = initialcondition!),
            orientation = FlatOrientation(),
            ref_state = NoReferenceState(),
            turbulence = ConstantDynamicViscosity(FT(0)),
            moisture = DryModel(),
        )

        vars_prim = Vars{vars_state(model, Primitive(), FT)}

        hb_recon! = HBFVReconstruction(model, fv_recon!)
        _grav = FT(grav(hb_recon!._atmo.param_set))

        @test width(hb_recon!) == 1

        num_state_prognostic = number_states(model, Prognostic())
        num_state_primitive = number_states(model, Primitive())
        local_state_face_primitive = ntuple(Val(2)) do _
            MArray{Tuple{num_state_primitive}, FT}(undef)
        end



        # interior point reconstruction test
        grid = FT[0; 1; 3; 6]
        grid_c = (grid[2:end] + grid[1:(end - 1)]) / 2
        local_cell_weights =
            MArray{Tuple{stencil_diameter}, FT}(grid[2:end] - grid[1:(end - 1)])

        # linear profile for all variables expect pressure
        local_state_primitive = SVector(ntuple(Val(stencil_diameter)) do _
            MArray{Tuple{num_state_primitive}, FT}(undef)
        end...)

        # values at the cell centers       1     2      3
        _, uc, _ = func(grid_c)
        # values at the cell faces     0.5   1.5*   2.5*     3.5
        _, uf, _ = func(grid)
        for i_d in 1:stencil_diameter
            for i_p in 1:num_state_prognostic
                local_state_primitive[i_d][i_p] = uc[i_d]
            end
        end

        # pressure profile is updated to satisfy the discrete hydrostatic balance 
        p_surf = FT(100) # at the bottom wall
        p_ref = p_surf
        for i_d in 1:stencil_diameter
            p_ref -=
                vars_prim(local_state_primitive[i_d]).ρ *
                _grav *
                local_cell_weights[i_d] / 2
            vars_prim(local_state_primitive[i_d]).p = p_ref
            p_ref -=
                vars_prim(local_state_primitive[i_d]).ρ *
                _grav *
                local_cell_weights[i_d] / 2
        end

        local_state_primitive_hb = copy(local_state_primitive)

        # interior point reconstruction
        hb_recon!(
            local_state_face_primitive[1],
            local_state_face_primitive[2],
            local_state_primitive,
            local_cell_weights,
        )
        # bottom face
        @test vars_prim(local_state_face_primitive[1]).ρ ≈ uf[2]
        @test vars_prim(local_state_face_primitive[1]).p ≈
              p_surf -
              vars_prim(local_state_primitive[1]).ρ *
              _grav *
              local_cell_weights[1]
        # top face
        @test vars_prim(local_state_face_primitive[2]).ρ ≈ uf[3]
        @test vars_prim(local_state_face_primitive[2]).p ≈
              p_surf -
              vars_prim(local_state_primitive[1]).ρ *
              _grav *
              local_cell_weights[1] -
              vars_prim(local_state_primitive[2]).ρ *
              _grav *
              local_cell_weights[2]

        # make sure the 
        for i_d in 1:stencil_diameter
            @test all(
                local_state_primitive[i_d] ≈ local_state_primitive_hb[i_d],
            )
        end

        @info "Start boundary test"


        # boundary point reconstruction  
        rng = SUnitRange(stencil_center, stencil_center)

        hb_recon!(
            local_state_face_primitive[1],
            local_state_face_primitive[2],
            local_state_primitive[rng],
            local_cell_weights[rng],
        )

        # bottom face
        @test vars_prim(local_state_face_primitive[1]).ρ ≈ uc[stencil_center]
        @test vars_prim(local_state_face_primitive[1]).p ≈
              vars_prim(local_state_primitive[stencil_center]).p +
              uc[stencil_center] * _grav * local_cell_weights[stencil_center] /
              2

        # top face
        @test vars_prim(local_state_face_primitive[2]).ρ ≈ uc[stencil_center]
        @test vars_prim(local_state_face_primitive[2]).p ≈
              vars_prim(local_state_primitive[stencil_center]).p -
              uc[stencil_center] * _grav * local_cell_weights[stencil_center] /
              2

    end


end
