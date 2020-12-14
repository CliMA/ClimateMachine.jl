using ClimateMachine

ClimateMachine.init()

using ClimateMachine.Ocean.Fields: RectangularElement, assemble

@testset "$(@__FILE__)" begin

    Nx = 3
    Ny = 4
    Nz = 5

    data = rand(Nx, Ny, Nz)
    x = repeat(range(0.0, 1.0, length = Nx), 1, Ny, Nz)
    y = repeat(range(0.0, 1.1, length = Ny), Nx, 1, Nz)
    z = repeat(range(0.0, 1.2, length = Nz), Nx, Ny, 1)

    element = RectangularElement(data, x, y, z)

    @test size(element) == (Nx, Ny, Nz)
    @test maximum(element) == maximum(data)
    @test minimum(element) == minimum(data)
    @test maximum(abs, element) == maximum(abs, data)
    @test minimum(abs, element) == minimum(abs, data)
    @test element[1, 1, 1] == data[1, 1, 1]

    east_element = RectangularElement(2 .* data, x .+ x[end, 1, 1], y, z)
    north_element = RectangularElement(3 .* data, x, y .+ y[1, end, 1], z)
    top_element = RectangularElement(4 .* data, x, y, z .+ z[1, 1, end])

    west_east = assemble(Val(1), element, east_element)
    south_north = assemble(Val(2), element, north_element)
    bottom_top = assemble(Val(3), element, top_element)

    @test west_east.x[1, 1, 1] == x[1, 1, 1]
    @test west_east.x[end, 1, 1] == 2 * x[end, 1, 1]

    @test south_north.y[1, 1, 1] == y[1, 1, 1]
    @test south_north.y[1, end, 1] == 2 * y[1, end, 1]

    @test bottom_top.z[1, 1, 1] == z[1, 1, 1]
    @test bottom_top.z[1, 1, end] == 2 * z[1, 1, end]

    northeast_element =
        RectangularElement(5 .* data, x .+ x[end, 1, 1], y .+ y[1, end, 1], z)

    # Remember that matrix literals are transposed, so "northeast"
    # is the bottom right corner (for example).
    four_elements = [
        element north_element
        east_element northeast_element
    ]

    four_elements = reshape(four_elements, 2, 2, 1)

    four_way = assemble(four_elements)

    @test four_way.x[end, 1, 1] == west_east.x[end, 1, 1]
    @test four_way.y[1, end, 1] == south_north.y[1, end, 1]
    @test four_way.z[1, 1, end] == z[1, 1, end]
end
