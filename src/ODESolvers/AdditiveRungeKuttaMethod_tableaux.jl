function ARK2GiraldoKellyConstantinescu_tableau(RT, paperversion)
  a32 = RT(paperversion ? (3 + 2sqrt(2)) / 6 : 1 // 2)
  RKA_explicit = [RT(0)           RT(0)   RT(0);
                  RT(2 - sqrt(2)) RT(0)   RT(0);
                  RT(1 - a32)     RT(a32) RT(0)]

  RKA_implicit = [RT(0)               RT(0)               RT(0);
                  RT(1 - 1 / sqrt(2)) RT(1 - 1 / sqrt(2)) RT(0);
                  RT(1 / (2sqrt(2)))  RT(1 / (2sqrt(2)))  RT(1 - 1 / sqrt(2))]

  RKB = [RT(1 / (2sqrt(2))), RT(1 / (2sqrt(2))), RT(1 - 1 / sqrt(2))]
  RKC = [RT(0), RT(2 - sqrt(2)), RT(1)]

  RKA_explicit, RKA_implicit, RKB, RKC
end

function ARK548L2SA2KennedyCarpenter_tableau(RT)
  Nstages = 8
  gamma = RT(2 // 9)

  RKA_explicit = zeros(RT, Nstages, Nstages)
  RKA_implicit = zeros(RT, Nstages, Nstages)
  RKB = zeros(RT, Nstages)
  RKC = zeros(RT, Nstages)

  # the main diagonal
  for is = 2:Nstages
    RKA_implicit[is, is] = gamma
  end

  RKA_implicit[3, 2] = RT(2366667076620 //  8822750406821)
  RKA_implicit[4, 2] = RT(-257962897183 // 4451812247028)
  RKA_implicit[4, 3] = RT(128530224461 // 14379561246022)
  RKA_implicit[5, 2] = RT(-486229321650 // 11227943450093)
  RKA_implicit[5, 3] = RT(-225633144460 // 6633558740617)
  RKA_implicit[5, 4] = RT(1741320951451 // 6824444397158)
  RKA_implicit[6, 2] = RT(621307788657 // 4714163060173)
  RKA_implicit[6, 3] = RT(-125196015625 // 3866852212004)
  RKA_implicit[6, 4] = RT(940440206406 // 7593089888465)
  RKA_implicit[6, 5] = RT(961109811699 // 6734810228204)
  RKA_implicit[7, 2] = RT(2036305566805 // 6583108094622)
  RKA_implicit[7, 3] = RT(-3039402635899 // 4450598839912)
  RKA_implicit[7, 4] = RT(-1829510709469 // 31102090912115)
  RKA_implicit[7, 5] = RT(-286320471013 // 6931253422520)
  RKA_implicit[7, 6] = RT(8651533662697 // 9642993110008)

  RKA_explicit[3, 1] = RT(1 // 9)
  RKA_explicit[3, 2] = RT(1183333538310 // 1827251437969)
  RKA_explicit[4, 1] = RT(895379019517 // 9750411845327)
  RKA_explicit[4, 2] = RT(477606656805 // 13473228687314)
  RKA_explicit[4, 3] = RT(-112564739183 // 9373365219272)
  RKA_explicit[5, 1] = RT(-4458043123994 // 13015289567637)
  RKA_explicit[5, 2] = RT(-2500665203865 // 9342069639922)
  RKA_explicit[5, 3] = RT(983347055801 // 8893519644487)
  RKA_explicit[5, 4] = RT(2185051477207 // 2551468980502)
  RKA_explicit[6, 1] = RT(-167316361917 // 17121522574472)
  RKA_explicit[6, 2] = RT(1605541814917 // 7619724128744)
  RKA_explicit[6, 3] = RT(991021770328 // 13052792161721)
  RKA_explicit[6, 4] = RT(2342280609577 // 11279663441611)
  RKA_explicit[6, 5] = RT(3012424348531 // 12792462456678)
  RKA_explicit[7, 1] = RT(6680998715867 // 14310383562358)
  RKA_explicit[7, 2] = RT(5029118570809 // 3897454228471)
  RKA_explicit[7, 3] = RT(2415062538259 // 6382199904604)
  RKA_explicit[7, 4] = RT(-3924368632305 // 6964820224454)
  RKA_explicit[7, 5] = RT(-4331110370267 // 15021686902756)
  RKA_explicit[7, 6] = RT(-3944303808049 // 11994238218192)
  RKA_explicit[8, 1] = RT(2193717860234 // 3570523412979)
  RKA_explicit[8, 2] = RKA_explicit[8, 1]
  RKA_explicit[8, 3] = RT(5952760925747 // 18750164281544)
  RKA_explicit[8, 4] = RT(-4412967128996 // 6196664114337)
  RKA_explicit[8, 5] = RT(4151782504231 // 36106512998704)
  RKA_explicit[8, 6] = RT(572599549169 // 6265429158920)
  RKA_explicit[8, 7] = RT(-457874356192 // 11306498036315)
  
  RKB[2] = 0
  RKB[3] = RT(3517720773327 // 20256071687669)
  RKB[4] = RT(4569610470461 // 17934693873752)
  RKB[5] = RT(2819471173109 // 11655438449929)
  RKB[6] = RT(3296210113763 // 10722700128969)
  RKB[7] = RT(-1142099968913 // 5710983926999)

  RKC[2] = RT(4 // 9)
  RKC[3] = RT(6456083330201 // 8509243623797)
  RKC[4] = RT(1632083962415 // 14158861528103)
  RKC[5] = RT(6365430648612 // 17842476412687)
  RKC[6] = RT(18 // 25)
  RKC[7] = RT(191 // 200)
  
  for is = 2:Nstages
    RKA_implicit[is, 1] = RKA_implicit[is, 2]
  end
 
  for is = 1:Nstages-1
    RKA_implicit[Nstages, is] = RKB[is]
  end

  RKB[1] = RKB[2]
  RKB[8] = gamma

  RKA_explicit[2, 1] = RKC[2]
  RKA_explicit[Nstages, 1] = RKA_explicit[Nstages, 2]

  RKC[1] = 0
  RKC[Nstages] = 1
  
  RKA_explicit, RKA_implicit, RKB, RKC
end

function ARK437L2SA1KennedyCarpenter_tableau(RT)
  Nstages = 7
  gamma = RT(1235 // 10000)

  RKA_explicit = zeros(RT, Nstages, Nstages)
  RKA_implicit = zeros(RT, Nstages, Nstages)
  RKB = zeros(RT, Nstages)
  RKC = zeros(RT, Nstages)

  # the main diagonal
  for is = 2:Nstages
    RKA_implicit[is, is] = gamma
  end

  RKA_implicit[3, 2] = RT(624185399699 // 4186980696204)
  RKA_implicit[4, 2] = RT(1258591069120 // 10082082980243)
  RKA_implicit[4, 3] = RT(-322722984531 // 8455138723562)
  RKA_implicit[5, 2] = RT(-436103496990 // 5971407786587)
  RKA_implicit[5, 3] = RT(-2689175662187 // 11046760208243)
  RKA_implicit[5, 4] = RT(4431412449334 // 12995360898505)
  RKA_implicit[6, 2] = RT(-2207373168298 // 14430576638973)
  RKA_implicit[6, 3] = RT(242511121179 // 3358618340039)
  RKA_implicit[6, 4] = RT(3145666661981 // 7780404714551)
  RKA_implicit[6, 5] = RT(5882073923981 // 14490790706663)
  RKA_implicit[7, 2] = 0
  RKA_implicit[7, 3] = RT(9164257142617 // 17756377923965)
  RKA_implicit[7, 4] = RT(-10812980402763 // 74029279521829)
  RKA_implicit[7, 5] = RT(1335994250573 // 5691609445217)
  RKA_implicit[7, 6] = RT(2273837961795 // 8368240463276)

  RKA_explicit[3, 1] = RT(247 // 4000)
  RKA_explicit[3, 2] = RT(2694949928731 // 7487940209513)
  RKA_explicit[4, 1] = RT(464650059369 // 8764239774964)
  RKA_explicit[4, 2] = RT(878889893998 // 2444806327765)
  RKA_explicit[4, 3] = RT(-952945855348 // 12294611323341)
  RKA_explicit[5, 1] = RT(476636172619 // 8159180917465)
  RKA_explicit[5, 2] = RT(-1271469283451 // 7793814740893)
  RKA_explicit[5, 3] = RT(-859560642026 // 4356155882851)
  RKA_explicit[5, 4] = RT(1723805262919 // 4571918432560)
  RKA_explicit[6, 1] = RT(6338158500785 // 11769362343261)
  RKA_explicit[6, 2] = RT(-4970555480458 // 10924838743837)
  RKA_explicit[6, 3] = RT(3326578051521 // 2647936831840)
  RKA_explicit[6, 4] = RT(-880713585975 // 1841400956686)
  RKA_explicit[6, 5] = RT(-1428733748635 // 8843423958496)
  RKA_explicit[7, 2] = RT(760814592956 // 3276306540349)
  RKA_explicit[7, 3] = RT(-47223648122716 // 6934462133451)
  RKA_explicit[7, 4] = RT(71187472546993 // 9669769126921)
  RKA_explicit[7, 5] = RT(-13330509492149 // 9695768672337)
  RKA_explicit[7, 6] = RT(11565764226357 // 8513123442827)

  RKB[2] = 0
  RKB[3] = RT(9164257142617 // 17756377923965)
  RKB[4] = RT(-10812980402763 // 74029279521829)
  RKB[5] = RT(1335994250573 // 5691609445217)
  RKB[6] = RT(2273837961795 // 8368240463276)
  RKB[7] = RT(247 // 2000)

  RKC[2] = RT(247 // 2000)
  RKC[3] = RT(4276536705230 // 10142255878289)
  RKC[4] = RT(67 // 200)
  RKC[5] = RT(3 // 40)
  RKC[6] = RT(7 // 10)

  for is = 2:Nstages
    RKA_implicit[is, 1] = RKA_implicit[is, 2]
  end

  for is = 1:Nstages-1
    RKA_implicit[Nstages, is] = RKB[is]
  end

  RKB[1] = RKB[2]

  RKA_explicit[2, 1] = RKC[2]
  RKA_explicit[Nstages, 1] = RKA_explicit[Nstages, 2]

  RKC[1] = 0
  RKC[Nstages] = 1
  
  RKA_explicit, RKA_implicit, RKB, RKC
end
