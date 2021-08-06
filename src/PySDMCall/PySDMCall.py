from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt
import sys
from multiprocessing import Process

sys.path.insert(1, '/home/alexios/Documents/projects/PySDM/PySDM-examples/PySDM_examples/Arabas_et_al_2015/')

import example
import threading


class Flag:
    def __init__(self):
        self.flg = 1

    def flagOff(self):
        self.flg = 0

    def flagOn(self):
        self.flg = 1

    def flagVal(self):
        return self.flg    


def test_call(varvals, dt, dx, simtime):
    print('test from Python: ')
    
    

    
    rho = np.array(varvals['ρ'])[:, 0, :]
    #print(rho.shape)
    #rhon = rho[:, 0, :]
    #print(rhon.shape)

    rhou1 = np.array(varvals['ρu[1]'])[:, 0, :]
    #print(rhou1.shape)

    rhou3 = np.array(varvals['ρu[3]'])[:, 0, :]
    #print(rhou3.shape)

    coef = dt/dx

    print(rho[0,0])
    u1 = rhou1 / rho * coef
    print(u1[0,0])
    u3 = rhou3 / rho * coef
    print(np.amin(u1))

    print('dt')
    print(dt)
    print('dx')
    print(dx)

    #print("u1 positive than negative") #vertical
    #print(u1[0]) 
    #print(u1[1])
    #print(u1[75])

    #print("u1 positive than negative than positive") #horiz
    #print(u1[:, 0])
    #print(u1[:, 1])


    #print("u3 all positive") #vertical
    #print(u3[0])
    #print(u3[1]) 
    #print(u3[10])
    #print("u3 all negative")
    #print(u3[-1])
    #print(u3[-2])
    #print(u3[-10])

    #print(u3[0, :])
    #print(u3[1, :])

    arkw_u1 = np.array([[ (u1[x, y-1] + u1[x, y]) / 2 for y in range(1, u1.shape[1])] for x in range(u1.shape[0])])
    arkw_u3 = np.array([[ (u3[x-1, y] + u3[x, y]) / 2 for y in range(u3.shape[1])] for x in range(1, u3.shape[0])])

    #print(arkw_u1[0])
    #print(arkw_u3[0])
   # print("pysdmcall py ")
   # print(flag)

    #p = Process(
    #    target=example.main,
    #    args=(arkw_u1, arkw_u3, dt, simtime, flag )
    #)    
    #p.start()
    #threading.Thread(
    #    target=example.main,
    #    args=(arkw_u1, arkw_u3, dt, simtime, flag, flagOff, )
    #).start()

    return example.main(arkw_u1, arkw_u3, dt, simtime)


    

    