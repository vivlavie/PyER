#Heat_SCE
#To read heat on and around SCE's
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import math
import numpy as np
# import dill
import unicodedata
import re
import csv
import os
import time
from kfxtools import * #fit for Python 2.

Ns = {}
with open('nodecords.csv', 'r') as file:
# with open('n.csv', 'r') as file:
    reader = csv.reader(file)
    for n in reader:
        Ns[n[0]] = [float(x) for x in n[1:]]
        # print(Ns[n[0]])
Cs = []
with open('connectivity.csv', 'r') as file:
# with open('c.csv', 'r') as file:
    reader = csv.reader(file)
    for c in reader:
        Cs.append([c[0],c[1],True])
        

Vh = 1.3 #average
Vv = 0.49 #downward slowest person
# r_th = [37500, 12500, 4700]
# t_th = [1, 40, 120]
impairment = ['Imd fatality','12.5kW/mw','4.7kW/m2']
r_th = [6300]
t_th = [1]
impairment = ['6.3kW/m2']

fieldnum = 0


basefolder = ".//R3D"
Js = {}
Js['J01'] = 'P04_A_AP'
Js['J02'] = 'P04_A_FP'
Js['J03'] = 'P04_A_AS'
Js['J04'] = 'P04_A_FS'
Js['J05'] = 'P04_B_AS'
Js['J06'] = 'P04_B_FS'
Js['J07'] = 'S05_A_AS'
Js['J08'] = 'S05_A_AP'
Js['J09'] = 'S05_A_F'
Js['J10'] = 'S05_A_F_10'
Js['J11'] = 'S05_B_AP'
Js['J12'] = 'S05_B_F'
Js['J13'] = 'S04_A_A'
Js['J14'] = 'S04_A_FP'
Js['J15'] = 'S04_A_FS'
Js['J16'] = 'S04_B_AP'
Js['J17'] = 'S04_B_F'
Js['J18'] = 'S03_A_P'
Js['J19'] = 'S03_B_A'
Js['J20'] = 'S03_B_F'
Js['J21'] = 'P02_B_FS'
Js['J22'] = 'P02_B_FP'
Js['J23'] = 'P05_A_P'
Js['J24'] = 'P05_A_S'
Js['J25'] = 'P05_B_A'
Js['J26'] = 'P05_B_F'
Js['J27'] = 'P03_B_P'
Js['J28'] = 'P03_B_FS'
Js['J29'] = 'KOD_B'


for j in Js.keys():
# for j in ['J12']:
    colid = int(j[-2:])+10
    fdr = Js[j][:3]    
    fn = basefolder + "//" + j+"_rad_exit.r3d"    
    fnv = basefolder +"//" + j+"_vis_exit.r3d"    
    
    if (os.path.exists(fn) == False):
        print(fn + " does not exist")
    elif (os.path.exists(fnv) == False):        
        print(fnv +" does not exist")
    else:    
        #Rad radiation
        T = readr3d(fn)            
        fieldname = T.names[fieldnum]
        print(fieldname)
        print(Js[j],fn,fieldname)

        #Read Visibility
        Tv = readr3d(fnv)    
        fieldnamev = Tv.names[fieldnum]
        er_imp_kfx = open(j+"_er_imp.kfx","w")
        #For each connection
        for c in Cs:
            x1,y1,z1 = Ns[c[0]]
            x2,y2,z2 = Ns[c[1]]
            z1 += 1.5        
            z2 += 1.5        
            dx,dy,dz = x2-x1,y2-y1,z2-z1
            x1m,y1m,z1m = x1+0.25*dx,y1+0.25*dy,z1+0.25*dz
            xm,ym,zm = x1+0.5*dx,y1+0.5*dy,z1+0.5*dz
            xm2,ym2,zm2 = x1+0.75*dx,y1+0.75*dy,z1+0.75*dz                       
            
            #Impairment assessment w.r.t. Radiation
            r1 = T.point_value(x1,y1,z1,fieldnum)
            r1m = T.point_value(x1m,y1m,z1m,fieldnum)
            rm = T.point_value(xm,ym,zm,fieldnum)
            rm2 = T.point_value(xm2,ym2,zm2,fieldnum)
            r2 = T.point_value(x2,y2,z2,fieldnum)
            #Personnel moving speed, horizontal or vertical
            if abs(dz) < 0.01:
                V = Vh
            else:
                V = Vv
            dt = sqrt(dx*dx+dy*dy+dz*dz)/V
            ravg = (r1+r1m+rm+rm2+r2)/5.
            rmax = max([r1,r1m,rm,rm2,r2])
            color = "0 1 0"
            #For each radiation impairment criteria
            for ck in range(0,len(r_th)):
                # if (ravg > r_th[ck]) and (dt > t_th[ck]):
                if (rmax > r_th[ck]) and (dt > t_th[ck]):
                    print ( Ns[c[0]], "->",Ns[c[1]], "{:8.1f} kW/m2 {:6.1f} sec {:s}".format(ravg/1000, dt, impairment[ck]))
                    c[2] = False #Set that the connnection is 'impaired'
                    color = "1 0 0"
                    #Python 2.x
                    # er_imp_kfx.write("COLOR: 1 0 0\n PART: "+c[0]+" "+c[1]+"\n")
                    #Python 3.x
                    # print("COLOR: 1 0 0", file = er_imp_kfx)                                   
            
            
            #Impairment assessment w.r.t. Visibility
            r1v = Tv.point_value(x1,y1,z1,fieldnum)
            r1mv = Tv.point_value(x1m,y1m,z1m,fieldnum)
            rmv = Tv.point_value(xm,ym,zm,fieldnum)
            rm2v = Tv.point_value(xm2,ym2,zm2,fieldnum)
            r2v = Tv.point_value(x2,y2,z2,fieldnum)
            # print(r1v, r1mv, rmv, rm2v, r2v)
            # Visibility = (r1v+r1mv+rmv+rm2v+r2v)/5.
            Visibility = max([min([r1v,r1mv,rmv,rm2v,r2v]),1])
            if (ravg > 4700) and (Visibility < 5.):
                    print ( Ns[c[0]], "->",Ns[c[1]], "Min visibility {:8.1f}".format(Visibility))                    
                    print([r1v,r1,r1mv,r1m,rmv,rm,rm2v,rm2,r2v,r2])
                    if c[2] == False:
                        #impaired by both heat and visibility
                        color = "0 1 1"           
                    else:
                        #impaired only by visibility
                        color = "0 0 1"           
                    c[2] = False #Set that the connnection is 'impaired'

            er_imp_kfx.write("COLOR: "+color+"\nPART: "+c[0]+"_"+c[1]+"\n")                                
            er_imp_kfx.write("SBOX: {:10.1f} {:10.1f} {:10.1f} {:10.1f} {:10.1f} {:10.1f} 400 400\n".format(x1*1000,y1*1000,z1*1000,x2*1000,y2*1000,z2*1000))
        er_imp_kfx.close()
