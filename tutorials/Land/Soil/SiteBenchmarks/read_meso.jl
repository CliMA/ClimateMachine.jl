using DelimitedFiles

file = "/Users/katherinedeck/Desktop/ok_data/MesoSoilv1_3/Sheet1-Table 1.csv"

data = readdlm(file, ',')
stations = ["MEF","BLAC","BREC","REDR"]
# @s = 11, 63, 14, 79
depths =[5,10,25,45,60,75]
di = 1
st_flag = [data[k,1] ∈  stations for k in 1:753]
st_data = data[st_flag,:]
for d in depths
    mine = median(st_data[st_data[:,2] .== d,9:end],dims = 1)
    mine[6] = mine[6] ./100 ./3600 ./24*1e7
    mine[3] = mine[3] *9.8 #(*10^3/rho)
    println(d, mine)
end

    
