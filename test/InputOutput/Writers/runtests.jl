using JLD2
using NCDatasets
using OrderedCollections
using Test
using ClimateMachine.Writers

@testset "Writers" begin
    odims = OrderedDict(
        "x" => collect(1:5),
        "y" => collect(1:5),
        "z" => collect(1010:10:1050),
    )
    ovars = OrderedDict(
        "v1" => (("x", "y", "z"), rand(5, 5, 5)),
        "v2" => (("x", "y", "z"), rand(5, 5, 5)),
    )
    jfn, _ = mktemp()
    jfull = full_name(JLD2Writer(), jfn)
    write_data(JLD2Writer(), jfn, odims, ovars, 0.5)
    nfn, _ = mktemp()
    nfull = full_name(NetCDFWriter(), nfn)
    write_data(NetCDFWriter(), nfn, odims, ovars, 0.5)

    jldopen(jfull, "r") do jds
        @test get(jds, "dim_1", "foo") == "x"
        @test get(jds, "dim_2", "foo") == "y"
        @test get(jds, "dim_3", "foo") == "z"
        @test get(jds, "dim_4", "foo") == "foo"
        @test jds["x"] == odims["x"]
        @test jds["y"] == odims["y"]
        @test jds["z"] == odims["z"]
        @test jds["t"] == [1]
        @test get(jds, "v1", []) == ovars["v1"][2]
        @test get(jds, "v2", []) == ovars["v2"][2]
        @test get(jds, "simtime", [1.0]) == [0.5]
    end

    NCDataset(nfull, "r") do nds
        @test nds["x"] == odims["x"]
        @test nds["y"] == odims["y"]
        @test nds["z"] == odims["z"]
        @test try
            nds["a"] == ones(5)
            false
        catch e
            true
        end
        @test nds["t"] == [1]
        @test nds["v1"] == ovars["v1"][2]
        @test nds["v2"] == ovars["v2"][2]
        @test nds["simtime"] == [0.5]
    end
end
