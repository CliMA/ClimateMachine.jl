using Test
using CLIMA.TurbulenceConvection.StateVecs

@testset "DomainSubSet, single domain" begin
  dd,dss = DomainDecomp(gm=1),DomainSubSet(gm=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)
  @test i_gm==1
  @test i_en==0
  @test i_ud==(0,)
  @test idx_ss==idx

  dd,dss = DomainDecomp(en=1),DomainSubSet(en=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)
  @test i_en==1
  @test i_gm==0
  @test i_ud==(0,)
  @test idx_ss==idx

  dd,dss = DomainDecomp(ud=1),DomainSubSet(ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)
  @test i_en==0
  @test i_gm==0
  @test i_ud==(1,)
  @test idx_ss==idx

  dd,dss = DomainDecomp(ud=2),DomainSubSet(ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)
  @test i_en==0
  @test i_gm==0
  @test i_ud==(1,2)
  @test idx_ss==idx
end

@testset "DomainSubSet, two domains" begin
  dd,dss = DomainDecomp(gm=1,en=1),DomainSubSet(gm=true,en=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)
  @test i_gm==2
  @test i_en==1
  @test i_ud==(0,)
  @test idx_ss==idx

  dd,dss = DomainDecomp(gm=1,ud=3),DomainSubSet(gm=true,ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)
  @test i_gm==4
  @test i_en==0
  @test i_ud==(1,2,3)
  @test idx_ss==idx

  dd,dss = DomainDecomp(en=1,ud=3),DomainSubSet(en=true,ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)
  @test i_gm==0
  @test i_en==4
  @test i_ud==(1,2,3)
  @test idx_ss==idx

  dd,dss = DomainDecomp(gm=1,ud=1),DomainSubSet(gm=true,ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)
  @test i_gm==2
  @test i_en==0
  @test i_ud==(1,)
  @test idx_ss==idx

  dd,dss = DomainDecomp(en=1,ud=1),DomainSubSet(en=true,ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)
  @test i_gm==0
  @test i_en==2
  @test i_ud==(1,)
  @test idx_ss==idx
end

@testset "DomainIdx, all domains" begin
  dd, dss = DomainDecomp(gm=1,en=1,ud=4), DomainSubSet(gm=true,en=true,ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)
  @test i_gm == 6
  @test i_en == 5
  @test i_ud == (1,2,3,4)
  @test idx_ss==idx
end

@testset "DomainIdx, utilizing DomainSubSet" begin
  dd = DomainDecomp(gm=1,en=1,ud=4)
  idx = DomainIdx(dd)

  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)
  @test i_gm == 6
  @test i_en == 5
  @test i_ud == (1,2,3,4)

  dss = DomainSubSet(gm=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)

  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx_ss)
  @test i_gm == 1
  @test i_en == 0
  @test i_ud == (0,)

  dss = DomainSubSet(en=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)

  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx_ss)
  @test i_gm == 0
  @test i_en == 1
  @test i_ud == (0,)

  dss = DomainSubSet(ud=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)

  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx_ss)
  @test i_gm == 0
  @test i_en == 0
  @test i_ud == (1,2,3,4)
end

@testset "Test DomainSubSet indexing, multiple domains" begin
  dd, dss = DomainDecomp(gm=1,en=1,ud=4), DomainSubSet(gm=true)

  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)

  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  @test updraft(idx_ss) == (0,)
  @test environment(idx_ss) == 0
  @test gridmean(idx_ss) == 1

  dss = DomainSubSet(en=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  @test updraft(idx_ss) == (0,)
  @test environment(idx_ss) == 1
  @test gridmean(idx_ss) == 0

  dss = DomainSubSet(ud=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  @test updraft(idx_ss) == (1,2,3,4)
  @test environment(idx_ss) == 0
  @test gridmean(idx_ss) == 0

  dss = DomainSubSet(gm=true,en=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  @test updraft(idx_ss) == (0,)
  @test environment(idx_ss) == 1
  @test gridmean(idx_ss) == 2

  dss = DomainSubSet(gm=true,ud=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  @test updraft(idx_ss) == (1,2,3,4)
  @test environment(idx_ss) == 0
  @test gridmean(idx_ss) == 5

  dss = DomainSubSet(en=true,ud=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  @test updraft(idx_ss) == (1,2,3,4)
  @test environment(idx_ss) == 5
  @test gridmean(idx_ss) == 0

  dss = DomainSubSet(gm=true,en=true,ud=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = get_sv_a_map(idx, idx_ss)
  @test updraft(idx_ss) == (1,2,3,4)
  @test environment(idx_ss) == 5
  @test gridmean(idx_ss) == 6
end

@testset "Test global indexing, many updrafts" begin

  vars = ((:ρ_0,   DomainSubSet(gm=true)),
          (:a,     DomainSubSet(gm=true,en=true,ud=true)),
          (:tke,   DomainSubSet(en=true,ud=true)),
          (:K_h,   DomainSubSet(ud=true)),
          )

  D = Dict([x => y for (x,y) in vars])

  dd = DomainDecomp(gm=1,en=1,ud=4)
  idx = DomainIdx(dd)

  sd_unmapped = get_sd_unmapped(vars, idx, dd)
  sd_mapped = get_sd_mapped(vars, idx, dd)

  @test sd_unmapped[:a] == Int[6, 5, 1, 2, 3, 4]
  @test sd_unmapped[:tke] == Int[5, 1, 2, 3, 4]
  @test sd_unmapped[:K_h] == Int[1, 2, 3, 4]
  @test sd_unmapped[:ρ_0] == Int[6]

  @test sd_mapped[:a] == Int[6, 5, 1, 2, 3, 4]
  @test sd_mapped[:tke] == Int[0, 5, 1, 2, 3, 4]
  @test sd_mapped[:K_h] == Int[0, 0, 1, 2, 3, 4]
  @test sd_mapped[:ρ_0] == Int[1, 0, 0, 0, 0]


  vm, dss_per_var, var_names = get_var_mapper(vars, dd)

  idx_ss_per_var = Dict([name => DomainIdx(dd, dss_per_var[name]) for name in var_names]...)
  a_map = Dict([name => get_sv_a_map(idx, idx_ss_per_var[name]) for name in var_names]...)
  i_gm,i_en,i_ud = eachdomain(idx)

  @test get_i_state_vec(vm, a_map[:ρ_0], :ρ_0, i_gm) == 1
  @test get_i_state_vec(vm, a_map[:a]  , :a, i_ud[1]) == 2
  @test get_i_state_vec(vm, a_map[:a]  , :a, i_ud[2]) == 3
  @test get_i_state_vec(vm, a_map[:a]  , :a, i_ud[3]) == 4
  @test get_i_state_vec(vm, a_map[:a]  , :a, i_ud[4]) == 5
  @test get_i_state_vec(vm, a_map[:a]  , :a, i_en)    == 6
  @test get_i_state_vec(vm, a_map[:a]  , :a, i_gm)    == 7

  @test get_i_state_vec(vm, a_map[:tke], :tke, i_ud[1]) == 8
  @test get_i_state_vec(vm, a_map[:tke], :tke, i_ud[2]) == 9
  @test get_i_state_vec(vm, a_map[:tke], :tke, i_ud[3]) == 10
  @test get_i_state_vec(vm, a_map[:tke], :tke, i_ud[4]) == 11
  @test get_i_state_vec(vm, a_map[:tke], :tke, i_en)    == 12

  @test get_i_state_vec(vm, a_map[:K_h], :K_h, i_ud[1]) == 13
  @test get_i_state_vec(vm, a_map[:K_h], :K_h, i_ud[2]) == 14
  @test get_i_state_vec(vm, a_map[:K_h], :K_h, i_ud[3]) == 15
  @test get_i_state_vec(vm, a_map[:K_h], :K_h, i_ud[4]) == 16

end
