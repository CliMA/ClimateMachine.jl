using ClimateMachine

ClimateMachine.init()

using ClimateMachine.Ocean.Fields:
    RectangularElement, x_assemble, y_assemble, z_assemble, assemble

@testset "$(@__FILE__)" begin

    Nx = 3
    Ny = 4
    Nz = 5

    data = rand(Nx, Ny, Nz)
    x = range(0.0, 1.0, length = Nx)
    y = range(0.0, 1.1, length = Ny)
    z = range(0.0, 1.2, length = Nz)

    element = RectangularElement(data, x, y, z)

    @test size(element) == (Nx, Ny, Nz)
    @test maximum(element) == maximum(data)
    @test minimum(element) == minimum(data)
    @test maximum(abs, element) == maximum(abs, data)
    @test minimum(abs, element) == minimum(abs, data)
    @test element[1, 1, 1] == data[1, 1, 1]

    east_element = RectangularElement(2 .* data, x .+ x[end], y, z)
    north_element = RectangularElement(3 .* data, x, y .+ y[end], z)
    top_element = RectangularElement(4 .* data, x, y, z .+ z[end])

    west_east = x_assemble(element, east_element)
    south_north = y_assemble(element, north_element)
    bottom_top = z_assemble(element, top_element)

    @test west_east.x[1] == x[1]
    @test west_east.x[end] == 2 * x[end]

    @test south_north.y[1] == y[1]
    @test south_north.y[end] == 2 * y[end]

    @test bottom_top.z[1] == z[1]
    @test bottom_top.z[end] == 2 * z[end]

    northeast_element =
        RectangularElement(5 .* data, x .+ x[end], y .+ y[end], z)

    # Transpoed...
    four_elements = [
        element north_element
        east_element northeast_element
    ]

    four_elements = reshape(four_elements, 2, 2, 1)

    four_way = assemble(four_elements)

    @test four_way.x[end] == west_east.x[end]
    @test four_way.y[end] == south_north.y[end]
    @test four_way.z[end] == z[end]
end
