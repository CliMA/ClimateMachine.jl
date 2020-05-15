using Test
using ClimateMachine.TicToc

function foo()
    @tic foo
    sleep(0.25)
    @toc foo
end

function bar()
    @tic bar
    sleep(1)
    @toc bar
end

@testset "TicToc" begin
    @test tictoc() >= 2
    foo_i = findfirst(s -> s == :tictoc__foo, TicToc.timing_info_names)
    bar_i = findfirst(s -> s == :tictoc__bar, TicToc.timing_info_names)
    @test foo_i != nothing
    @test bar_i != nothing
    foo()
    foo()
    @test TicToc.timing_infos[foo_i].ncalls == 2
    @test TicToc.timing_infos[bar_i].ncalls == 0
    @test TicToc.timing_infos[foo_i].time >= 5e8
    bar()
    @test TicToc.timing_infos[bar_i].ncalls == 1
    @test TicToc.timing_infos[bar_i].time >= 1e9
    buf = IOBuffer()
    old_stdout = stdout
    try
        rd, = redirect_stdout()
        TicToc.print_timing_info()
        Libc.flush_cstdio()
        flush(stdout)
        write(buf, readavailable(rd))
    finally
        redirect_stdout(old_stdout)
    end
    str = String(take!(buf))
    @test findnext("tictoc__foo", str, 1) != nothing
    @test findnext("tictoc__bar", str, 1) != nothing
end
