using Test
using CLIMA.TurbulenceConvection.StateVecs
SV = StateVecs

@testset "DomainSubSet, unit tests" begin

  @test SV.get_param(SV.GridMean{1}) == 1
  @test SV.get_param(SV.Environment{1}) == 1
  @test SV.get_param(SV.Updraft{5}) == 5

  @test SV.gridmean(DomainSubSet(gm=true)) == true
  @test SV.environment(DomainSubSet(gm=true)) == false
  @test SV.updraft(DomainSubSet(gm=true)) == false

  @test SV.gridmean(DomainSubSet(en=true)) == false
  @test SV.environment(DomainSubSet(en=true)) == true
  @test SV.updraft(DomainSubSet(en=true)) == false

  @test SV.gridmean(DomainSubSet(ud=true)) == false
  @test SV.environment(DomainSubSet(ud=true)) == false
  @test SV.updraft(DomainSubSet(ud=true)) == true

  dd = DomainDecomp(gm=1, en=1, ud=4)
  @test SV.get_param(dd, DomainSubSet(gm=true)) == (1,0,0)
  @test SV.get_param(dd, DomainSubSet(en=true)) == (0,1,0)
  @test SV.get_param(dd, DomainSubSet(ud=true)) == (0,0,4)

  @test SV.gridmean(dd, DomainSubSet(gm=true)) == 1
  @test SV.environment(dd, DomainSubSet(en=true)) == 1
  @test SV.updraft(dd, DomainSubSet(ud=true)) == 4

  @test SV.gridmean(dd, DomainSubSet(en=true)) == 0
  @test SV.environment(dd, DomainSubSet(gm=true)) == 0
  @test SV.updraft(dd, DomainSubSet(gm=true)) == 0

  @test SV.sum(dd, DomainSubSet(gm=true)) == 1
  @test SV.sum(dd, DomainSubSet(en=true)) == 1
  @test SV.sum(dd, DomainSubSet(ud=true)) == 4
end

@testset "DomainSubSet, single domain" begin
  dd,dss = DomainDecomp(gm=1),DomainSubSet(gm=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  gm, en, ud, sd, al = allcombinations(idx)
  @test gm==1
  @test en==0
  @test ud==(0,)
  @test idx_ss==idx

  dd,dss = DomainDecomp(en=1),DomainSubSet(en=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  gm, en, ud, sd, al = allcombinations(idx)
  @test en==1
  @test gm==0
  @test ud==(0,)
  @test idx_ss==idx

  dd,dss = DomainDecomp(ud=1),DomainSubSet(ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  gm, en, ud, sd, al = allcombinations(idx)
  @test en==0
  @test gm==0
  @test ud==(1,)
  @test idx_ss==idx

  dd,dss = DomainDecomp(ud=2),DomainSubSet(ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  gm, en, ud, sd, al = allcombinations(idx)
  @test en==0
  @test gm==0
  @test ud==(1,2)
  @test idx_ss==idx
end

@testset "DomainSubSet, two domains" begin
  dd,dss = DomainDecomp(gm=1,en=1),DomainSubSet(gm=true,en=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  gm, en, ud, sd, al = allcombinations(idx)
  @test gm==2
  @test en==1
  @test ud==(0,)
  @test idx_ss==idx

  dd,dss = DomainDecomp(gm=1,ud=3),DomainSubSet(gm=true,ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  gm, en, ud, sd, al = allcombinations(idx)
  @test gm==4
  @test en==0
  @test ud==(1,2,3)
  @test idx_ss==idx

  dd,dss = DomainDecomp(en=1,ud=3),DomainSubSet(en=true,ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  gm, en, ud, sd, al = allcombinations(idx)
  @test gm==0
  @test en==4
  @test ud==(1,2,3)
  @test idx_ss==idx

  dd,dss = DomainDecomp(gm=1,ud=1),DomainSubSet(gm=true,ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  gm, en, ud, sd, al = allcombinations(idx)
  @test gm==2
  @test en==0
  @test ud==(1,)
  @test idx_ss==idx

  dd,dss = DomainDecomp(en=1,ud=1),DomainSubSet(en=true,ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  gm, en, ud, sd, al = allcombinations(idx)
  @test gm==0
  @test en==2
  @test ud==(1,)
  @test idx_ss==idx
end

@testset "DomainIdx, all domains" begin
  dd, dss = DomainDecomp(gm=1,en=1,ud=4), DomainSubSet(gm=true,en=true,ud=true)
  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  gm, en, ud, sd, al = allcombinations(idx)
  @test gm == 6
  @test en == 5
  @test ud == (1,2,3,4)
  @test idx_ss==idx
end

@testset "DomainIdx, utilizing DomainSubSet" begin
  dd = DomainDecomp(gm=1,en=1,ud=4)
  idx = DomainIdx(dd)

  gm, en, ud, sd, al = allcombinations(idx)
  @test gm == 6
  @test en == 5
  @test ud == (1,2,3,4)

  dss = DomainSubSet(gm=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)

  gm, en, ud, sd, al = allcombinations(idx_ss)
  @test gm == 1
  @test en == 0
  @test ud == (0,)

  dss = DomainSubSet(en=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)

  gm, en, ud, sd, al = allcombinations(idx_ss)
  @test gm == 0
  @test en == 1
  @test ud == (0,)

  dss = DomainSubSet(ud=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)

  gm, en, ud, sd, al = allcombinations(idx_ss)
  @test gm == 0
  @test en == 0
  @test ud == (1,2,3,4)
end

@testset "Test DomainSubSet indexing, multiple domains" begin
  dd, dss = DomainDecomp(gm=1,en=1,ud=4), DomainSubSet(gm=true)

  idx = DomainIdx(dd)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  gm, en, ud, sd, al = allcombinations(idx)

  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  @test SV.updraft(idx_ss) == (0,)
  @test SV.environment(idx_ss) == 0
  @test SV.gridmean(idx_ss) == 1

  dss = DomainSubSet(en=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  @test SV.updraft(idx_ss) == (0,)
  @test SV.environment(idx_ss) == 1
  @test SV.gridmean(idx_ss) == 0

  dss = DomainSubSet(ud=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  @test SV.updraft(idx_ss) == (1,2,3,4)
  @test SV.environment(idx_ss) == 0
  @test SV.gridmean(idx_ss) == 0

  dss = DomainSubSet(gm=true,en=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  @test SV.updraft(idx_ss) == (0,)
  @test SV.environment(idx_ss) == 1
  @test SV.gridmean(idx_ss) == 2

  dss = DomainSubSet(gm=true,ud=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  @test SV.updraft(idx_ss) == (1,2,3,4)
  @test SV.environment(idx_ss) == 0
  @test SV.gridmean(idx_ss) == 5

  dss = DomainSubSet(en=true,ud=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  @test SV.updraft(idx_ss) == (1,2,3,4)
  @test SV.environment(idx_ss) == 5
  @test SV.gridmean(idx_ss) == 0

  dss = DomainSubSet(gm=true,en=true,ud=true)
  idx_ss = DomainIdx(dd, dss)
  a_map = SV.get_sv_a_map(idx, idx_ss)
  @test SV.updraft(idx_ss) == (1,2,3,4)
  @test SV.environment(idx_ss) == 5
  @test SV.gridmean(idx_ss) == 6
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
  @test SV.get_param(dd) == (1,1,4)
  @test SV.has_gridmean(idx) == true
  @test SV.has_environment(idx) == true
  @test SV.has_updraft(idx) == true

  sd = subdomains(idx)
  al = alldomains(idx)
  gm, en, ud = eachdomain(idx)
  gm, en, ud, sd, al = allcombinations(idx)

  @test sum(dd) == 6

  sd_unmapped = SV.get_sd_unmapped(vars, idx, dd)
  sd_mapped = SV.get_sd_mapped(vars, idx, dd)

  @test sd_unmapped[:a] == Int[6, 5, 1, 2, 3, 4]
  @test sd_unmapped[:tke] == Int[5, 1, 2, 3, 4]
  @test sd_unmapped[:K_h] == Int[1, 2, 3, 4]
  @test sd_unmapped[:ρ_0] == Int[6]

  @test sd_mapped[:a] == Int[6, 5, 1, 2, 3, 4]
  @test sd_mapped[:tke] == Int[0, 5, 1, 2, 3, 4]
  @test sd_mapped[:K_h] == Int[0, 0, 1, 2, 3, 4]
  @test sd_mapped[:ρ_0] == Int[1, 0, 0, 0, 0]

  vm, dss_per_var, var_names = SV.get_var_mapper(vars, dd)

  idx_ss_per_var = Dict([name => DomainIdx(dd, dss_per_var[name]) for name in var_names]...)
  a_map = Dict([name => SV.get_sv_a_map(idx, idx_ss_per_var[name]) for name in var_names]...)
  gm, en, ud, sd, al = allcombinations(idx)

  @test SV.var_string(vm, idx, idx_ss_per_var[:a], :a, gm) == "a_gm"
  @test SV.var_string(vm, idx, idx_ss_per_var[:a], :a, en) == "a_en"
  @test SV.var_string(vm, idx, idx_ss_per_var[:a], :a, ud[1]) == "a_ud_1"
  @test SV.var_string(vm, idx, idx_ss_per_var[:a], :a, ud[2]) == "a_ud_2"
  @test SV.var_string(vm, idx, idx_ss_per_var[:a], :a, ud[3]) == "a_ud_3"
  @test SV.var_string(vm, idx, idx_ss_per_var[:a], :a, ud[4]) == "a_ud_4"
  @test SV.var_suffix(vm, idx, idx_ss_per_var[:a], :a, gm) == "_gm"
  @test SV.var_suffix(vm, idx, idx_ss_per_var[:a], :a, en) == "_en"
  @test SV.var_suffix(vm, idx, idx_ss_per_var[:a], :a, ud[1]) == "_ud_1"
  @test SV.var_suffix(vm, idx, idx_ss_per_var[:a], :a, ud[2]) == "_ud_2"
  @test SV.var_suffix(vm, idx, idx_ss_per_var[:a], :a, ud[3]) == "_ud_3"
  @test SV.var_suffix(vm, idx, idx_ss_per_var[:a], :a, ud[4]) == "_ud_4"
  @test_throws BoundsError SV.var_suffix(vm, idx, idx_ss_per_var[:a], :a, 1000)

  @test SV.get_i_var(a_map[:ρ_0], gm) == 1
  @test SV.get_i_var(a_map[:a]  , ud[1]) == 1
  @test SV.get_i_var(a_map[:a]  , ud[2]) == 2
  @test SV.get_i_var(a_map[:a]  , ud[3]) == 3
  @test SV.get_i_var(a_map[:a]  , ud[4]) == 4
  @test SV.get_i_var(a_map[:a]  , en) == 5
  @test SV.get_i_var(a_map[:a]  , gm) == 6

  @test SV.get_i_state_vec(vm, a_map[:ρ_0], :ρ_0, gm) == 1
  @test SV.get_i_state_vec(vm, a_map[:a]  , :a, ud[1]) == 2
  @test SV.get_i_state_vec(vm, a_map[:a]  , :a, ud[2]) == 3
  @test SV.get_i_state_vec(vm, a_map[:a]  , :a, ud[3]) == 4
  @test SV.get_i_state_vec(vm, a_map[:a]  , :a, ud[4]) == 5
  @test SV.get_i_state_vec(vm, a_map[:a]  , :a, en)    == 6
  @test SV.get_i_state_vec(vm, a_map[:a]  , :a, gm)    == 7

  @test SV.get_i_state_vec(vm, a_map[:tke], :tke, ud[1]) == 8
  @test SV.get_i_state_vec(vm, a_map[:tke], :tke, ud[2]) == 9
  @test SV.get_i_state_vec(vm, a_map[:tke], :tke, ud[3]) == 10
  @test SV.get_i_state_vec(vm, a_map[:tke], :tke, ud[4]) == 11
  @test SV.get_i_state_vec(vm, a_map[:tke], :tke, en)    == 12

  @test SV.get_i_state_vec(vm, a_map[:K_h], :K_h, ud[1]) == 13
  @test SV.get_i_state_vec(vm, a_map[:K_h], :K_h, ud[2]) == 14
  @test SV.get_i_state_vec(vm, a_map[:K_h], :K_h, ud[3]) == 15
  @test SV.get_i_state_vec(vm, a_map[:K_h], :K_h, ud[4]) == 16

end
