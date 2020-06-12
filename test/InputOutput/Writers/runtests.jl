using JLD2
using NCDatasets
using OrderedCollections
using Test
using ClimateMachine.Writers

@testset "Writers" begin
    odims = OrderedDict(
        "x" => (collect(1:5), Dict()),
        "y" => (collect(1:5), Dict()),
        "z" => (collect(1010:10:1050), Dict()),
    )
    ovars = OrderedDict(
        "v1" => (("x", "y", "z"), rand(5, 5, 5), Dict()),
        "v2" => (("x", "y", "z"), rand(5, 5, 5), Dict()),
    )
    jfn, _ = mktemp()
    jfull = full_name(JLD2Writer(), jfn)
    write_data(JLD2Writer(), jfn, odims, ovars, 0.5)
    nfn, _ = mktemp()
    nfull = full_name(NetCDFWriter(), nfn)
    write_data(NetCDFWriter(), nfn, odims, ovars, 0.5)

    jldopen(jfull, "r") do jds
        @test get(jds, "dim_1", "foo") == "time"
        @test get(jds, "dim_2", "foo") == "x"
        @test get(jds, "dim_3", "foo") == "y"
        @test get(jds, "dim_4", "foo") == "z"
        @test get(jds, "dim_5", "foo") == "foo"
        @test jds["x"] == odims["x"][1]
        @test jds["y"] == odims["y"][1]
        @test jds["z"] == odims["z"][1]
        @test length(jds["time"]) == 1
        @test get(jds, "time", [1.0]) == [0.5]
        @test get(jds, "v1", []) == ovars["v1"][2]
        @test get(jds, "v2", []) == ovars["v2"][2]
    end

    NCDataset(nfull, "r") do nds
        @test nds["x"] == odims["x"][1]
        @test nds["y"] == odims["y"][1]
        @test nds["z"] == odims["z"][1]
        @test try
            nds["a"] == ones(5)
            false
        catch e
            true
        end
        @test length(nds["time"]) == 1
        @test nds["time"] == [0.5]
        @test nds["v1"] == ovars["v1"][2]
        @test nds["v2"] == ovars["v2"][2]
    end
end
