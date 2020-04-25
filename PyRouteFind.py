

#PyRouteFind.py 


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
from copy import deepcopy
from kfxtools import * #fit for Python 2.

import heapq

# 탐색할 그래프와 시작 정점을 인수로 전달받습니다.
def dijkstra(graph, start, end):
    # 시작 정점에서 각 정점까지의 거리를 저장할 딕셔너리를 생성하고, 무한대(inf)로 초기화합니다.
    distances = {vertex: [float('inf'), start] for vertex in graph}
    # 그래프의 시작 정점의 거리는 0으로 초기화 해줌
    distances[start] = [0, start]
    # 모든 정점이 저장될 큐를 생성합니다.
    queue = []
    # 그래프의 시작 정점과 시작 정점의 거리(0)을 최소힙에 넣어줌
    heapq.heappush(queue, [distances[start][0], start])
    while queue:        
        # 큐에서 정점을 하나씩 꺼내 인접한 정점들의 가중치를 모두 확인하여 업데이트합니다.
        current_distance, current_vertex = heapq.heappop(queue)
        # 더 짧은 경로가 있다면 무시한다.
        if distances[current_vertex][0] < current_distance:
            continue            
        for adjacent, weight in graph[current_vertex].items():
            distance = current_distance + weight
            # 만약 시작 정점에서 인접 정점으로 바로 가는 것보다 현재 정점을 통해 가는 것이 더 가까울 경우에는
            if distance < distances[adjacent][0]:
                # 거리를 업데이트합니다.
                distances[adjacent] = [distance, current_vertex]
                heapq.heappush(queue, [distance, adjacent])    
    path = end
    path_output = end + '->'
    
    while distances[path][1] != start:        
        path_output += distances[path][1] + '->'
        path = distances[path][1]
    path_output += start        
    # print (path_output)
    # distance_cumulative = distances[end][0]    
    return path_output, distances[end][0], distances 
    


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

R = {}    
for c in Cs:
    x1,y1,z1 = Ns[c[0]]
    x2,y2,z2 = Ns[c[1]]
    if (z1 == z2):
        v = Vh
    elif (x1 == x2) and (y1 == y2) and (not (z1 == z2)):
        v = Vv
    else:
        print('something wrong in applying the velocity')
       
    d = sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    s = d / v
    #no the distance but the time along the route is being used
    if not (c[0] in R.keys()):
        R[c[0]] = {c[1]:s}
    else:
        R[c[0]][c[1]] = s

    if not (c[1] in R.keys()):
        R[c[1]] = {c[0]:s} 
    else:
        R[c[1]][c[0]] = s

NodesForeward = ['71','80','91','79','109','87']
SafeDoors = ['13','24'] #13 STBD, 24 PORT

for start in NodesForeward:
    for end in SafeDoors:
        er_path,time_taken,er_graph = dijkstra(R,start,end)
        print("{:80s}{:10.1f}".format(er_path,time_taken))
        c = np.random.uniform(size=(1,3))[0]
        # print("COLOR: {:8.1f} {:8.1f} {:8.1f}".format(c[0],c[1],c[2]))
        # print("PART: "+start+"_"+end)                                
        ns = er_path.split('->')
        for ni in range(1,len(ns)):
            x1,y1,z1 = Ns[ns[ni]]
            x2,y2,z2 = Ns[ns[ni-1]]
            # print("SBOX: {:10.1f} {:10.1f} {:10.1f} {:10.1f} {:10.1f} {:10.1f} 400 400".format(x1*1000,y1*1000,z1*1000,x2*1000,y2*1000,z2*1000))



# 
mygraph = {
    'A' 'C': 1, 'D': 2},
    'B': {},
    'C': {'B': 5, 'D': 2},
    'D': {'E': 3, 'F': 5},
    'E': {'F': 1},
    'F': {'A': 5}
}



#Check all nodes are connected
AllNodes = R.keys()
for start in AllNodes:
    for end in AllNodes:
        er_path,time_taken,er_graph = dijkstra(R,start,end)
        if (time_taken < 0.1) and (start != end) and not (start in R[end].keys()) and not (end in R[start].keys()):
            print("No route between ", start, " and ", end, time_taken)
        # print("{:80s}{:10.1f}".format(er_path,time_taken))



start = '41'
end = '0'
er_path,time_taken,er_graph = dijkstra(R,start,end)
#er_path from 'end' to 'start
nodes_taken = er_path.split('->')
s=0.
t = 0.
n = nodes_taken[0]
while not (n == start):
    tt=er_graph[n][0]
    nn = er_graph[n][1]
    dt = R[n][nn]
    t += dt
    print(n,'->(',tt,')->',dt)
    s += tt
    n = nn
print(n)
er_path_imp,time_taken_imp,er_graph_imp = dijkstra(R,start,end)
    


# P04_A_AP        : /projects/300341/Reliance/Fire/P04/J01
# P04_B_FS        : /projects/300341/Reliance/Fire/P04/J06
# S05_A_F           : /projects/300341/Reliance/Fire/S05/J09
# S05_B_F           : /projects/300341/Reliance/Fire/S05/J12
# S04_A_A          : /projects/300341/Reliance/Fire/S04/J13
# S03_A_P          : /projects/300341/Reliance/Fire/S03/J18
# S03_B_A          : /projects/300341/Reliance/Fire/P04/J19
NumRowsJetInfo = 5
basefolder = "./Rev.B/"
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

start = '97'
end = '13'
er_path,time_taken,er_graph = dijkstra(R,start,end)


r_th = [37500, 12500, 4700]
t_th = [1, 40, 120]
impairment = ['Imd fatality','12.5kW/mw','4.7kW/m2']
fieldnum = 0
# for j in Js.keys():
for j in ['J09']:
    Rimp = deepcopy(R)
    # colid = int(j[-2:])+10
    # fdr = Js[j][:3]
    fn =  "./R3D/" + j+"_rad_exit.r3d"    
    fnv = "./R3D/" + j+"_vis_exit.r3d"    
    # fn = basefolder + "/" + j+"_rad_exit.r3d"    
    # fnv = basefolder +"/" + j+"_vis_exit.r3d"    
    
    if (os.path.exists(fn) == False):
        print(fn + " does not exist")
    elif (os.path.exists(fnv) == False):        
        print(fnv +" does not exist")
    else:    
        print(fn)
        T = readr3d(fn)    
        Tv = readr3d(fnv)    
        #Define field name to read in
        fieldname = T.names[fieldnum]
        fieldnamev = Tv.names[fieldnum]
        print(fieldname)
        # print(Js[j],fn,fieldname)
        # er_imp_kfx = open(j+"_er_imp.kfx","w")

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
            color = "0 1 0"
            #For each radiation impairment criteria
            for ck in range(0,len(r_th)):
                if (ravg > r_th[ck]) and (dt > t_th[ck]):
                    print ( Ns[c[0]], "->",Ns[c[1]], "{:8.1f} kW/m2 {:6.1f} sec {:s}".format(ravg/1000, dt, impairment[ck]))
                    #take out the impaired route from the connected route graph
                    try:
                        Rimp[c[1]].pop(c[0])
                    # else:
                        Rimp[c[0]].pop(c[1])
                    except:
                        print('Conection ',c[1],' to ',c[0], ' has alreadby been taken out.')
                    #Set that the connnection is 'impaired'
                    c[2] = False 
                    color = "1 0 0"

                    #P
                    # nodes_taken = er_path.split('->')ython 2.x
                    # er_imp_kfx.write("COLOR: nodes_taken:
                    # er_graph[n]RT: path]
                    # +" "+c[1]+"\n")
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
                    #take out the impaired route from the connected route graph
                    try:
                        Rimp[c[1]].pop(c[0])
                    # else:
                        Rimp[c[0]].pop(c[1])
                    except:
                        print('Conection ',c[1],' to ',c[0], ' has alreadby been taken out.')
                    c[2] = False #Set that the connnection is 'impaired'

            # er_imp_kfx.write("COLOR: "+color+"\nPART: "+c[0]+"_"+c[1]+"\n")                                
            # er_imp_kfx.write("SBOX: {:10.1f} {:10.1f} {:10.1f} {:10.1f} {:10.1f} {:10.1f} 400 400\n".format(x1*1000,y1*1000,z1*1000,x2*1000,y2*1000,z2*1000))
        # er_imp_kfx.close()
    er_path_imp,time_taken_imp,er_graph_imp = dijkstra(Rimp,start,end)

                      
er_path
er_path_imp
time_taken
time_taken_imp

# iHeat.save('SCE_Heat_'+time.strftime("%Y%m%d-%H%M%S")+'.xlsx')


""" for j in Js.keys():
    #Check impaired ER and update the Graph
    er_path_imp,time_taken_imp,er_graph_imp = dijkstra(Rimp,start,end)
    #For each combination of 'start' and 'end', check effects of fire
    for start in NodesForeward:
        for end in SafeDoors:
            er_path,time_taken,er_graph = dijkstra(Rimp,start,end) """



ns = er_path.split('->')
Ps = np.zeros((len(ns),3))
for ni in range(0,len(ns)):    
    Ps[ni,:] = Ns[ns[ni]]

ns_imp = er_path_imp.split('->')
Ps_imp = np.zeros((len(ns),3))
for ni in range(0,len(ns)):    
    Ps_imp[ni,:] = Ns[ns_imp[ni]]
    

x = Ps[:,0]
y = Ps[:,1]
z = Ps[:,2]
x_imp = Ps_imp[:,0]
y_imp = Ps_imp[:,1]
z_imp = Ps_imp[:,2]

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.pbaspect = [1.,1./3.,1./6.]
# ax.pbaspect = [6,2,1]
# ax.pbaspect = [1,1,1]
# ax.set_xlim3d(20,200)
# ax.set_ylim3d(-90,90)
# ax.set_zlim3d(0,180)
# ax.pbaspect = [1,1,0.1]

ax.xaxis.set_major_locator(plt.FixedLocator([120, 141, 168, 193]))
ax.xaxis.set_major_formatter(plt.FixedFormatter(['2/3','3/4','4/5','S05']))
ax.yaxis.set_major_locator(plt.FixedLocator([-27, -3, 3, 27]))
ax.yaxis.set_major_formatter(plt.FixedFormatter(['ER_S','Tray_S','Tray_P','ER_P']))
ax.zaxis.set_major_locator(plt.FixedLocator([35, 44, 52]))
ax.zaxis.set_major_formatter(plt.FixedFormatter(['A','B','C']))
# ax.plot(x, y, z, label="'Optimal Escape Path from {:s} to {:s}".format(start,end))
ax.plot(x, y, z,'g')
ax.plot(x_imp, y_imp, z_imp,'r--')
ax.auto_scale_xyz([40,220], [-30,30], [26,56])

# ax.legend()
plt.show()


