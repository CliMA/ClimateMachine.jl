# Testing reference values and precisions
# Each test block of varr and parr should be followed by an append to refVals, refPrecs arrays.
# e.g.
#   refVals=[]
#   refPrecs=[]
#
#   varr = ..........
#   par  = ..........
#
#   append!(refVals ,[ varr ] )
#   append!(refPrecs,[ parr ] )
#
#   varr = ..........
#   par  = ..........
#
#   append!(refVals ,[ varr ] )
#   append!(refPrecs,[ parr ] )
#
#   varr = ..........
#   par  = ..........
#
#   append!(refVals ,[ varr ] )
#   append!(refPrecs,[ parr ] )
#
#   etc.....
#
#   Now for real!
#

refVals = []
refPrecs = []
# SC ========== Test number 1 reference values and precision match template. =======
# SC ========== /Users/chrishill/projects/clima/cm/test/Ocean/HydrostaticBoussinesq/test_ocean_gyre.jl test reference values ======================================
# BEGIN SCPRINT
# varr - reference values (from reference run)
# parr - digits match precision (hand edit as needed)
#
# [
#  [ MPIStateArray Name, Field Name, Maximum, Minimum, Mean, Standard Deviation ],
#  [         :                :          :        :      :          :           ],
# ]
varr = [
    [
        "Q",
        "u[1]",
        -2.18427865357219835873e-02,
        4.54905817704320050709e-02,
        2.91799084468619303323e-03,
        9.86739717788472255056e-03,
    ],
    [
        "Q",
        "u[2]",
        -6.47928098321387119229e-02,
        7.44690631237251432495e-02,
        -1.82439830964300966215e-03,
        1.02482752029024841434e-02,
    ],
    [
        "Q",
        :η,
        -6.35241759974374819997e-01,
        6.25677877693153861038e-01,
        -8.60068718305024041901e-04,
        2.24583755068219675932e-01,
    ],
    [
        "Q",
        :θ,
        9.03726335555428482264e-05,
        9.03968461560725344839e+00,
        2.49953258705220227043e+00,
        2.19711762918947606238e+00,
    ],
    [
        "s_aux",
        :y,
        0.00000000000000000000e+00,
        4.00000000000000046566e+06,
        2.00000000000000000000e+06,
        1.18025270281469495967e+06,
    ],
    [
        "s_aux",
        :w,
        -4.50036075461459223737e-05,
        4.09915220804811021062e-05,
        3.06722602595587297903e-07,
        8.22999489798410955438e-06,
    ],
    [
        "s_aux",
        :pkin,
        -9.00231747934270076783e-01,
        0.00000000000000000000e+00,
        -3.32080874523538072118e-01,
        2.56218992025050162908e-01,
    ],
    [
        "s_aux",
        :wz0,
        -2.22688579102610652283e-05,
        1.52895210358102938127e-05,
        -2.78589672590168561413e-08,
        7.72520757253795920232e-06,
    ],
]
parr = [
    ["Q", "u[1]", 12, 12, 12, 12],
    ["Q", "u[2]", 12, 12, 12, 12],
    ["Q", :η, 12, 12, 12, 12],
    ["Q", :θ, 12, 12, 12, 12],
    ["s_aux", :y, 12, 12, 12, 12],
    ["s_aux", :w, 12, 12, 12, 12],
    ["s_aux", :pkin, 12, 12, 12, 12],
    ["s_aux", :wz0, 12, 12, 12, 12],
]
# END SCPRINT

append!(refVals, [varr])
append!(refPrecs, [parr])

# SC ========== Test number 2 reference values and precision match template. =======
# SC ========== /Users/chrishill/projects/clima/cm/test/Ocean/HydrostaticBoussinesq/test_ocean_gyre.jl test reference values ======================================
# BEGIN SCPRINT
# varr - reference values (from reference run)
# parr - digits match precision (hand edit as needed)
#
# [
#  [ MPIStateArray Name, Field Name, Maximum, Minimum, Mean, Standard Deviation ],
#  [         :                :          :        :      :          :           ],
# ]
varr = [
    [
        "Q",
        "u[1]",
        -2.14403161060213384714e-02,
        4.51092912722281316751e-02,
        2.91883577365915263327e-03,
        9.85543151135863208789e-03,
    ],
    [
        "Q",
        "u[2]",
        -6.47849908268208068973e-02,
        7.44377079062340796245e-02,
        -1.82528856707940708402e-03,
        1.02465626974943338490e-02,
    ],
    [
        "Q",
        :η,
        -6.35241717401733962944e-01,
        6.25675643032199912952e-01,
        -8.59997171611655590248e-04,
        2.24578663698960928619e-01,
    ],
    [
        "Q",
        :θ,
        1.21624650983828366162e-04,
        9.05348870526675320036e+00,
        2.49953463844110279624e+00,
        2.19723475068906504148e+00,
    ],
    [
        "s_aux",
        :y,
        0.00000000000000000000e+00,
        4.00000000000000046566e+06,
        2.00000000000000000000e+06,
        1.18025270281469495967e+06,
    ],
    [
        "s_aux",
        :w,
        -4.48750663401728297594e-05,
        4.07425796079212936126e-05,
        3.05580091823959372610e-07,
        8.20754136359875149074e-06,
    ],
    [
        "s_aux",
        :pkin,
        -9.00231018468760080253e-01,
        0.00000000000000000000e+00,
        -3.32080814803801249724e-01,
        2.56218661345592568779e-01,
    ],
    [
        "s_aux",
        :wz0,
        -2.24168211854045644460e-05,
        1.54464254869175437296e-05,
        -2.71999565492576589218e-08,
        7.76164707621753630128e-06,
    ],
]
parr = [
    ["Q", "u[1]", 12, 12, 12, 12],
    ["Q", "u[2]", 12, 12, 12, 12],
    ["Q", :η, 12, 12, 12, 12],
    ["Q", :θ, 12, 12, 12, 12],
    ["s_aux", :y, 12, 12, 12, 12],
    ["s_aux", :w, 12, 12, 12, 12],
    ["s_aux", :pkin, 12, 12, 12, 12],
    ["s_aux", :wz0, 12, 12, 12, 12],
]

append!(refVals, [varr])
append!(refPrecs, [parr])
