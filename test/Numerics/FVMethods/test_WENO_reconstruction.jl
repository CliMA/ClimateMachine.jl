using Test
using StaticArrays
using ClimateMachine.VariableTemplates
using ClimateMachine.FVMethods:
    const_reconstruction!,
    weno_reconstruction!,
    fv_reconstruction!

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

@testset "First order (1 stenciels) reconstruction test" begin
    FT = Float64

    grid = FT[0; 1] * FT(0.1)
    h = grid[2:end] - grid[1:(end - 1)]
    grid_c = (grid[2:end] + grid[1:(end - 1)]) / 2
    func = lin_func
    # values at the cell centers       1
    num_state_primitive, uc, uc_I = func(grid_c)
    # values at the cell faces     0.5*   1.5*
    num_state_primitive, uf, uf_I = func(grid)
    u = similar(uc)
    for i in 1:length(h)
        u[:, i] = (uf_I[:, i + 1] - uf_I[:, i]) / h[i]
    end
    cell_states_primitive = (u[:, 1],)
    state_primitive_top = similar(cell_states_primitive[1])
    state_primitive_bottom = similar(cell_states_primitive[1])
    cell_weights = SVector(h[1])
    const_reconstruction!(
        state_primitive_top,
        state_primitive_bottom,
        cell_states_primitive,
        cell_weights,
    )
    @test all(uc[:, 1] .≈ state_primitive_bottom)
    @test all(uc[:, 1] .≈ state_primitive_top)
end

@testset "Second order (3 stenciels) reconstruction test" begin
    FT = Float64

    ## weno3 test pass, set IS .= 1.0, leads to p2 recovery
    grid = FT[0; 1; 3; 6] * FT(0.1)
    h = grid[2:end] - grid[1:(end - 1)]
    grid_c = (grid[2:end] + grid[1:(end - 1)]) / 2
    func = quad_func
    # values at the cell centers       1     2      3
    num_state_primitive, uc, uc_I = func(grid_c)
    # values at the cell faces     0.5   1.5*   2.5*     3.5
    num_state_primitive, uf, uf_I = func(grid)
    u = similar(uc)
    for i in 1:length(h)
        u[:, i] = (uf_I[:, i + 1] - uf_I[:, i]) / h[i]
    end
    cell_states_primitive = (
        u[:, 1],
        u[:, 2],
        u[:, 3],
    )
    state_primitive_top = similar(cell_states_primitive[1])
    state_primitive_bottom = similar(cell_states_primitive[1])
    cell_weights = SVector(h[1], h[2], h[3])
    weno_reconstruction!(
        state_primitive_top,
        state_primitive_bottom,
        cell_states_primitive,
        cell_weights,
    )
    @test isapprox(uf[:, 2][1], state_primitive_bottom[1]; rtol=0.05)
    @test uf[:, 2][2] ≈ state_primitive_bottom[2]

    @test isapprox(uf[:, 3][1], state_primitive_top[1]; rtol=0.05)
    @test uf[:, 3][2] ≈ state_primitive_top[2]

    ## fv test pass, set limiter .= 1, leads to p2 recovery  ???
    fv_reconstruction!(
        state_primitive_top,
        state_primitive_bottom,
        cell_states_primitive,
        cell_weights,
    )

    @test isapprox(uf[:, 2][1], state_primitive_bottom[1]; rtol=0.05)
    @test uf[:, 2][2] ≈ state_primitive_bottom[2]

    @test isapprox(uf[:, 3][1], state_primitive_top[1]; rtol=0.05)
    @test uf[:, 3][2] ≈ state_primitive_top[2]
end

@testset "Fifth order (5 stenciels) reconstruction test" begin
    FT = Float64

    ## weno5 test, set IS .= 1.0, leads to p4 recovery
    grid = FT[0; 1; 3; 6; 7; 9] * FT(0.1)
    h = grid[2:end] - grid[1:(end - 1)]
    grid_c = (grid[2:end] + grid[1:(end - 1)]) / 2
    func = fourth_func
    # values at the cell centers        1     2      3      4      5
    num_state_primitive, uc, uc_I = func(grid_c)
    # values at the cell faces     0.5    1.5    2.5*    3.5*    4.5    5.5
    num_state_primitive, uf, uf_I = func(grid)
    u = similar(uc)
    for i in 1:length(h)
        u[:, i] = (uf_I[:, i + 1] - uf_I[:, i]) / h[i]
    end

    cell_states_primitive = (
        u[:, 1],
        u[:, 2],
        u[:, 3],
        u[:, 4],
        u[:, 5],
    )
    state_primitive_top = similar(cell_states_primitive[1])
    state_primitive_bottom = similar(cell_states_primitive[1])
    cell_weights = SVector(h[1], h[2], h[3], h[4], h[5])
    weno_reconstruction!(
        state_primitive_top,
        state_primitive_bottom,
        cell_states_primitive,
        cell_weights,
    )

    @test isapprox(uf[:, 3][1], state_primitive_bottom[1]; rtol=0.05)
    @test uf[:, 3][2] ≈ state_primitive_bottom[2]

    @test isapprox(uf[:, 4][1], state_primitive_top[1]; rtol=0.05)
    @test uf[:, 4][2] ≈ state_primitive_top[2]

end
