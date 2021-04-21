include("../interface/grid/domains.jl")

@testset "domain tests" begin
    """
        IntervalDomain
    """
    interval = IntervalDomain(min = -1, max = 2)
    @test interval.min == -1 && interval.max == 2 && interval.periodic == false
    @test ndims(interval) == 1
    @test length(interval) == 3

    interval = IntervalDomain(min = -1, max = 2, periodic = true)
    @test interval.min == -1 && interval.max == 2 && interval.periodic == true

    """
        ProductDomain
    """
    interval  = IntervalDomain(min = -1, max = 2)
    interval2 = IntervalDomain(min = 2, max = 3, periodic=true)
    product   = ProductDomain(domains = (interval, interval2))
    @test product.domains[1] == interval && product.domains[2] == interval2
    @test product.periodicity == (false, true)
    @test ndims(product) == 2
    @test length(product) == (3, 1)

    product2 = ProductDomain(domains = (interval, interval2))
    product3 = ProductDomain(domains = (product, product2))
    @test product3.periodicity == (false, true, false, true)
    @test ndims(product3) == 4
    @test length(product3) == (3, 1, 3, 1)

    """
        SphericalShell
    """
    sphere = SphericalShell(radius = 1, height = 2)
    @test sphere.radius == 1 && sphere.height == 2
    @test ndims(sphere) == 3
    @test length(sphere) == (1, 1, 2)
end