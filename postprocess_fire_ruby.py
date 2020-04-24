#from __future__ import print_function
# -*- coding: utf-8 -*-
#!/usr/bin/env python2
import sys
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.interpolate as intp
import csv
import re
from matplotlib_venn import venn2, venn3

""" 
Functions for postprocessing of fire simulations. The following functions have been
modified for the Gudrun project where only three leak categories are used:
-leak_category 
-summarise_leak_categories

loe_leak_category is no longer used, and has not been modified
"""

#** Create scenario matrix *****************************************************************************
#**********************************************************************************************

def setWind(row, winddata):
    """ Assign wind probabilities to each scenario.  Winddata must contain 
    velocities and directions which are found in the scenario matrix. 
    The syntax of winddata.csv is as follows:
    velocity,NORTH,SOUTH,EAST,WEST
    6,0.1358,0.1155,0.1061,0.1124
    12,0.1328,0.1826,0.0846,0.1302
    """
    velocity=row['Wind speed']
    winddir=row['Wind direction']
    Pwind=winddata.loc[winddata['velocity']==velocity][winddir]
    if len(Pwind) > 0:
        return Pwind.values[0]
    else:
        return np.nan

def setPleakDir(row, sceninfo):
    """Probabilities for leak directions are set directly from fire_scenarios.csv
    For instance, if n_leakdir=6, a probability of 1/6=0.167 is set for each scenario.
    If different weightings are to be applied (e.g. downward leaks are more likely,
    this will have to be changed manually (e.g. in pandas or Excel)
    Syntax of fire_scenarios.csv:
    area,leakrate,n_leakdir,n_leakloc
    Mezz_JET,5,6,1
    Mezz_JET,15,6,1
    Mezz_JET,30,6,1
    Mezz_JET,100,1,1
    Lower_JET,5,6,1
    ...
    """
    area=row['Area']
    leakrate=row['Leak rate']
    n_leakdir=sceninfo.loc[(sceninfo['area']==area) & (sceninfo['leakrate'] == leakrate)]['n_leakdir']
    if len(n_leakdir) > 0:
        p_leakdir = 1./n_leakdir.values[0]
    else:
        p_leakdir = np.nan
    return p_leakdir

def setPleakLoc(row, sceninfo):
    """Probabilities for leak locations are set directly from fire_scenarios.csv
    For instance, if n_leakloc=2, a probability of 1/2=0.5 is set for each scenario.
    If different weightings are to be applied (e.g. one location is more likely,
    this will have to be changed manually (e.g. in pandas or Excel)
    """
    area=row['Area']
    leakrate=row['Leak rate']
    n_leakloc=sceninfo.loc[(sceninfo['area']==area) & (sceninfo['leakrate'] == leakrate)]['n_leakloc']
    if len(n_leakloc) > 0:
        p_leakloc = 1./n_leakloc.values[0]
    else:
        p_leakloc = np.nan
    return p_leakloc

def create_scenario_matrix(scenlist='min_vis.csv', sceninfo='fire_scenarios.csv',winddata='winddata.csv'):
    """ Note that min_vis.csv (or max_rad.csv) has to be created before running this function
       (TODO: Make a new function which does not require all -pa commands to be run first)
        Input: min_vis.csv/max_rad.csv, fire_scenarios.csv, winddata.csv
        Output: scenario_matrix (all scenarios with weighted frequencies of occurrence)
                This dataframe may be exported to csv and xlsx
    """
    df=pd.read_csv(scenlist,usecols=['Filename'])
    winddata=pd.read_csv(winddata)
    sceninfo=pd.read_csv(sceninfo)
    df = df.rename(columns={'Filename': 'scenario_ID'})
    df['Area'] = df['scenario_ID'].apply(lambda x: '_'.join(x.split('_')[:-4]))
    df['leakdir'] = df['scenario_ID'].apply(lambda x: int(x.split('_')[-4]))
    df['Leak rate'] = df['scenario_ID'].apply(lambda x: float(re.sub('[^0-9\.]','',x.split('_')[-3])))
    df['Wind direction'] = df['scenario_ID'].apply(lambda x: x.split('_')[-2])
    df['Wind speed'] = df['scenario_ID'].apply(lambda x: int(x.split('_')[-1]))
    df.loc[:,'P_wind']=df.apply(lambda row: setWind(row, winddata), axis=1)
    df.loc[:,'P_leakdir']=df.apply(lambda row: setPleakDir(row, sceninfo), axis=1)
    df.loc[:,'P_leakloc']=df.apply(lambda row: setPleakLoc(row, sceninfo), axis=1)
    df['f_scen'] = df['P_wind'] * df['P_leakdir'] * df['P_leakloc']
    return df
 
#** Network tools *****************************************************************************
#**********************************************************************************************

def read_nodes(start_nodes='start_nodes.csv', end_nodes='end_nodes.csv'):
    """Read the start nodes and end nodes (usually only one end node) into
    two lists
    """
    if type(end_nodes)==str: #If read from file (use try/except instead?)
        with open(end_nodes, 'r') as f:
            reader=csv.reader(f)
            for row in reader:
                endNodes=[int(i) for i in list(row)]
    else:
        if type(end_nodes)==list:
            endNodes=end_nodes #If not use the end nodes as is
        else:
            endNodes=[end_nodes]
    all_startnodes=pd.read_csv(start_nodes)
    list_of_all_startnodes=list(all_startnodes['node'].values)
    return list_of_all_startnodes, endNodes

def read_safe_edges(safe_edges='safe_edges.csv'):
    """Read safe edges from file into a list. If the file contains
    no safe edges, [] is returned
    """
    list_of_safe_edges=[]
    try:
        with open(safe_edges, 'r') as f:
            reader=csv.reader(f)
            for row in reader:
                list_of_safe_edges.append([int(i) for i in list(row)])
        return list_of_safe_edges
    except IOError:
        return []

def initialise_graph(edges='connectivity.csv', nodes='nodecords.csv', endnode=None):
    """Initialise the graph from an escape route network
    TO DO: Generalise the endnode-issues, may vary from project to project"""

    edgelist = pd.read_csv(edges, names=['node1', 'node2'])
    nodelist = pd.read_csv(nodes, names=['id', 'x', 'y', 'z'])
    G = nx.Graph()
    # Add edges
    for i, elrow in edgelist.iterrows():
        G.add_edge(elrow[0], elrow[1])
    # Add node attributes
    for i, nlrow in nodelist.iterrows():
        G.node[nlrow['id']].update(nlrow[1:].to_dict())

    for edge in G.edges():
        n1 = edge[0]
        n2 = edge[1]
        dx = abs(G.node[n1]['x'] - G.node[n2]['x'])
        dy = abs(G.node[n1]['y'] - G.node[n2]['y'])
        dz = abs(G.node[n1]['z'] - G.node[n2]['z'])
        G[n1][n2]['dx'] = dx
        G[n1][n2]['dy'] = dy
        G[n1][n2]['dz'] = dz
        G[n1][n2]['distance'] = max([dx, dy, dz])

    if endnode:
        G.remove_node(endnode)  # Remove endnode and adjacent edges from graph
    else:
        print("Warning: No endnode removed from graph")
    return G

def add_common_endnode(G,end_nodes):
    """If personell may escape to several locations that are safe, it is convenient
    to add an endnode that connects all these locations
    Input: A graph G and a list of endnodes, e.g. [232,240,288]
    Output: A new graph G with the new common endnode inserted.
    If there e.g. are 136 nodes in the network, the new endnode will be no. 137
    TODO: Add a distance of zero?
    """
    common_endnode = max(list(G.nodes))+1
    G_new = G.copy()
    for i in range(0,len(end_nodes)):
        G_new.add_edge(common_endnode,end_nodes[i])
    return G_new

def imp_matrix_from_file(imp_type, imp_criteria, rad_file='max_rad.csv', vis_file='min_vis.csv'):
    """Create an impairment matrix with True/False for all scenarios for all edges"""
    if imp_type == 'rad':
        impairment_matrix = pd.read_csv(rad_file, index_col='Filename')
        # True if radiation > imp_criteria
        impairment_matrix = impairment_matrix > imp_criteria
    elif imp_type == 'vis':
        impairment_matrix = pd.read_csv(vis_file, index_col='Filename')
        # True if visibility < imp_criteria
        impairment_matrix = impairment_matrix < imp_criteria
    else:
        # Use sys.exit(1) with message?
        print("Error: imp_type should be either 'rad' or 'vis'")
    return impairment_matrix

def graph_impaired(G, scenario_edges, list_of_safe_edges):
    """
    Input: A graph G and a series of edges for a specific scenario with True/False 
           depending on the edge is impaired or not 
    Output: A new graph G_impaired where impaired edges are removed 
    Edges included in list_of_safe_edges will not be removed
    """
    graph_safe_edges = safe_edges(list_of_safe_edges)

    G_impaired = G.copy()
    for i in range(len(scenario_edges)):
        if scenario_edges[i] == True:
            n1, n2 = nodesFromText(scenario_edges.index[i])
            # Remove edge if it is not included in list_of_safe_edges:
            if graph_safe_edges.has_edge(n1, n2) == False:
                G_impaired.remove_edge(n1, n2)
    return G_impaired

def safe_edges(safe_edges='safe_edges.csv'):
    """ Return a graph with safe edges, e.g. escape tunnel, shielded stair tower etc.
    """
    list_of_safe_edges=read_safe_edges(safe_edges)
    if list_of_safe_edges == []:
        return nx.Graph()  # Return empty graph
    else:
        graph_safe_edges = nx.Graph()
        for row in list_of_safe_edges:
            graph_safe_edges.add_edge(row[0], row[1])
        return graph_safe_edges

def remove_edges():
    """Not needed, just use G.remove_edge(n1,n2)"""
    return ''

def add_edges():
    """Not needed, just use G.add_edge(n1,n2)"""
    return ''

def nodesFromText(text):
    """ Return numbers from strings like "Escape_12_35" """
    firstNode = int(text.split('_')[1])
    secondNode = int(text.split('_')[2])
    return firstNode, secondNode

def lossOfEscape(G, startNode, endNode):
    """ Returns 1 if loss of escape, 0 if not"""
    loe = 0
    if nx.has_path(G, startNode, endNode) == False:
        loe = 1
    return loe

def get_impaired_edges(G, scenario_imp_matrix):
    """For a given scenario, return impaired edges as graph_impairedEdges
    Example:
    get_impaired_edges(G_escapeNetwork,imp_matrix.iloc[i])
    get_impaired_edges(G_escapeNetwork,imp_matrix.loc['Lower_JET_1_5kg_NORTH_6'])
    """
    graph_impairedEdges=nx.Graph()
    impaired_edges=scenario_imp_matrix.index[np.nonzero(scenario_imp_matrix)]
    for i in range(0,len(impaired_edges)):
        n1, n2 = nodesFromText(impaired_edges[i])
        graph_impairedEdges.add_edge(n1, n2)

    return graph_impairedEdges

def calculate_loe(G, imp_matrix, start_nodes, end_nodes, list_of_safe_edges):
#def calculate_loe(G, imp_matrix, start_nodes='start_nodes.csv',
#         end_nodes='end_nodes.csv', list_of_safe_edges='safe_edges.csv'):
    """
    For a given escape network G, return as a dataframe all scenarios in imp_matrix
    with information on whether it is possible to escape from startNodes to endNodes
    TODO: Speed up the code (use Fortran?)
    """
    startNodes, endNodes = read_nodes(start_nodes, end_nodes)
    header = ['sn_' + str(sn) + '_en_' + str(en)
              for en in endNodes for sn in startNodes]
    header.insert(0, 'scenario_ID')
    temp_file = []
    for i in range(len(imp_matrix)):
        filename = imp_matrix.iloc[i].name
        temp_array = [filename]
        for en in endNodes:
            for sn in startNodes:
                G_scenario = graph_impaired(G, imp_matrix.iloc[i], list_of_safe_edges)
                loe = lossOfEscape(G_scenario, sn, en)
                temp_array.append(loe)
        temp_file.append(temp_array)
    df_loe = pd.DataFrame(temp_file, columns=header)
    return df_loe

def loe_from_areas(df_loe,start_nodes='start_nodes.csv'):
    """Add columns to df_loe on whether escape is lost for the areas defined in start_nodes.csv
    """
    def check_column(col,startnodes):
        """Help function to check whether a string is included in a column"""
        included=False
        for i in startnodes:
            if col.startswith(i):
                included=True
        return included
    StartNodes=pd.read_csv(start_nodes)
    for area in StartNodes['area_name'].unique():
        area_nodes=StartNodes[StartNodes['area_name']==area]['node']
        filter_col = [col for col in df_loe if 
            check_column(col,['sn_'+str(i) for i in area_nodes.values])]
        df_loe[area]=df_loe[filter_col].sum(axis=1) > 0
    return df_loe*1 #Convert True/False to 1/0

#---------- Fatality calculations ------------------------------------------------------------------
#From Andrea

#---------- Plotting ------------------------------------------------------------------

def plotVenn2(df, column1, column2, ax, labels=('Set 1', 'Set 2')):
    A, B = df[column1].values, df[column2].values
    Ab = np.sum((A == 1) & (B != 1))
    aB = np.sum((A != 1) & (B == 1))
    AB = np.sum((A == 1) & (B == 1))
    ab = np.sum((A != 1) & (B != 1))
    return venn2(subsets=(Ab, aB, AB, ab), set_labels=labels,ax=ax)

def plotVenn3(df, column1, column2, column3, ax, labels=('Set 1', 'Set 2', 'Set 3')):
    """column1='loe_146', for instance, labels=('aa','bb','cc')"""
    A, B, C = df[column1].values, df[column2].values, df[column3].values
    Abc = np.sum((A == 1) & (B != 1) & (C != 1))
    aBc = np.sum((A != 1) & (B == 1) & (C != 1))
    ABc = np.sum((A == 1) & (B == 1) & (C != 1))
    abC = np.sum((A != 1) & (B != 1) & (C == 1))
    AbC = np.sum((A == 1) & (B != 1) & (C == 1))
    aBC = np.sum((A != 1) & (B == 1) & (C == 1))
    ABC = np.sum((A == 1) & (B == 1) & (C == 1))
    return venn3(subsets=(Abc, aBc, ABc, abC, AbC, aBC, ABC), set_labels=labels, ax=ax)

def kfxbox(x, y, z, dx, dy, dz, color, por=0, partname=''):
    """Create a box for kfx-file
       BOX: X0 Y0 Z0 DX DY DZ [P XPH YPH ZPH VPH]
    """
    x, y, z, dx, dy, dz = [
        1000 * x for x in (x, y, z, dx, dy, dz)]  # Millimeter to meter
    if not color:
        color = (1, 0, 0)
    text = ''
    if partname:
        text += 'COLOR: {} {} {}\n'.format(*color)
        text += 'PART: {}\n'.format(partname)
    text += 'BOX: {} {} {} {} {} {} P {} {} {} {}\n'.format(
        x, y, z, dx, dy, dz, por, por, por, por)
    return text

def kfxsbox(x0, y0, z0, x1, y1, z1, color=(1, 0, 0), partname='', h=0.1, w=0.1):
    """Create a sbox for kfx-file
       SBOX: X0 Y0 Z0 X1 Y1 Z1 HEIGHT WIDTH
    """
    x0, y0, z0, x1, y1, z1, h, w = [
        1000 * x for x in (x0, y0, z0, x1, y1, z1, h, w)]  # Millimeter to meter
    x0, y0, z0, x1, y1, z1 = [
        x + 150 for x in (x0, y0, z0, x1, y1, z1)]  # Better placement
    text = ''
    if partname:
        text += 'COLOR: {} {} {}\n'.format(*color)
        text += 'PART: {}\n'.format(partname)
    text += 'SBOX: {} {} {} {} {} {} {} {}\n'.format(
        x0, y0, z0, x1, y1, z1, h, w)
    return text

def kfxlabel(x, y, z, label, r=0, g=0, b=0, partname='text'):
    """Create label for kfx-file"""
    text = ''
    if partname:
        text += 'COLOR: {} {} {}\n'.format(r, g, b)
        text += 'PART: {}\n'.format(partname)
    text += 'TEXT: {} {} {} {}\n'.format(x + 0.3, y, z + 0.3, label)
    return text

def visualize_network(G, outfile, startNodes, endNodes, list_of_safe_edges, labels=True):
    """Create a kfx file visualizing the graph. Colors are hardcoded.
    Use [] as argument if e.g. endNodes are not to be visualized
    """
    #kfxoutfile = '{}.kfx'.format(out_file)
    print("Writing file {}\n".format(outfile))
    with open(outfile, 'w') as f:
        for n in G.nodes(data=True):
            x = n[1]['x']
            y = n[1]['y']
            z = n[1]['z']
            dx = 0.3
            dy = 0.3
            dz = 0.3
            color = (1, 0, 0)
            if n[0] in startNodes:
                color = (0, 0.5, 0)
            if n[0] in endNodes:
                color = (0, 0, 0.5)
            f.write(kfxbox(x, y, z, dx, dy, dz, color, por=0,
                           partname='Node {}'.format(n[0])))
            if labels:
                f.write(kfxlabel(x, y, z, '{}'.format(n[0]), 0, 0, 1,
                                 partname='Node label {}'.format(n[0])))
        for e in G.edges(data=True):
            n1 = e[0]
            n2 = e[1]
            x0 = G.node[n1]['x']
            y0 = G.node[n1]['y']
            z0 = G.node[n1]['z']
            x1 = G.node[n2]['x']
            y1 = G.node[n2]['y']
            z1 = G.node[n2]['z']
            color = (1, 0, 0)
            if [n1, n2] in list_of_safe_edges or [n2, n1] in list_of_safe_edges:
                color = (0, 1, 0)
            f.write(kfxsbox(x0, y0, z0, x1, y1, z1, color,
                            partname='Edge {n1} to {n2}'.format(n1=n1, n2=n2), h=0.1, w=0.1))

#** Calculation tools *****************************************************************************
#**********************************************************************************************

def leak_category(leak_rate): #Modified for Gudrun (Ma has been removed)
    if (leak_rate > 0.1) & (leak_rate <= 1):
        category = 'S'
    elif (leak_rate > 1) & (leak_rate <= 10):
        category = 'Me'
    elif leak_rate > 10:
        category = 'L'
    else:
        category = 'Not found'
    return category
    
def read_leakfz(area, leakfz_file='leakfz'):
    """Read leakfz into numpy array. Return None if the file is
    not found or the area cannot be found in the file. Syntax of leakfz:
    Cellar_JET
    0.000001,0.1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,...
    3.17E-02,2.75E-02,4.03E-03,1.79E-03,1.08E-03,7.47E-04,5.56E-04,...
    4.25E-02,3.92E-02,5.65E-03,2.56E-03,1.55E-03,1.07E-03,7.94E-04,...
    RC330_360_JET
    0.000001,0.1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,...
    2.48E-02,2.08E-02,3.29E-03,1.54E-03,9.53E-04,6.68E-04,5.04E-04,...
    0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,...
    TODO: Change file format of leakfz
    """
    area_found=False
    try:
        with open(leakfz_file, 'r') as f:
            data=f.readlines()
            for index, ll in enumerate(data):
                ll=ll.strip() # Remove '\n'
                line=ll.split(',')
                if line[0] == area:
                    area_found=True
                    rates=data[index+1].strip().split(',')
                    gasfz=data[index+2].strip().split(',')
                    liqfz=data[index+3].strip().split(',')
                    if area.endswith('JET'):
                        freq=gasfz
                    elif area.endswith('POOL'):
                        freq=liqfz
                    else:
                        print("Error: In leakfz_file, area must end with either 'JET' or 'POOL'")
    except IOError:
        return None
    if area_found:
        rates=[float(i) for i in rates] 
        freq=[float(i) for i in freq] 
        return np.array([rates,freq])
    else:
        return None # Or return something else?!

def interpolate_values(x, y):
    """Interpolate rates and impairment probabilites / leak frequencies
    A rate of 0 kg/s with 0 impairment probability /leak frequence is added
    """
    xnew=range(0,int(max(x))+1)
    x = np.insert(x,0,0)
    y = np.insert(y,0,0)
    f = intp.interp1d(x, y, kind='linear')
    ynew = f(xnew)
    return ynew

def leak_categorize(y,category,intp_leakfz=None):
    """For a set of discretized interpolated impairment probabilites y,
    return the impariment probability for a given leak category (e.g. 'Me'). 
    Do frequency weighting if intp_leakfz is available
    """
    if intp_leakfz is not None:
        y=y[:len(intp_leakfz)]
        y_category=y[[x for x in range(0,len(y)) if (leak_category(x)==category)]] # Impprob for the category
        f_category=intp_leakfz[[x for x in range(0,len(y)) if (leak_category(x)==category)]] # Frequencies for the category
        norm_f_category=[float(x)/sum(f_category) for x in f_category]
        freq_weighted_cons=sum(y_category*norm_f_category)
    else:
        y_category=y[[x for x in range(0,len(y)) if (leak_category(x)==category)]]
        freq_weighted_cons=np.mean(y_category)
    return freq_weighted_cons

def escalate(y, rate):
    """ Input: A discretized array of consequences from 0 to max(y), as well
               as an "escalation rate" (normally set to 50 kg/s)
        Output: A new array where each element in y is assigned a new
               consequence y[i]=y[i+rate]
        Note: If rate is set to 50 kg/s, the last 50 elements are set
              to y[-1] (since the input array is not large enough to do y[i+50])
    """
    y_new=y.copy()
    for i in range(0,len(y)-rate):
        y_new[i] = y[i+rate]
    y_new[-rate:]=y[-1]
    return y_new


def extrapolate(x,y,area):
    """This function is project specific. It is only used if extrapolation is set to True
    in the function loe_leak_category. The extrapolation may be variable-specific,
    i.e. different for sight and radiation. Also, it may be different for
    loss of escape, PLL, exposure of targets etc.
    REWRITE THIS FUNCTION! Instead take a file as input! <---------------------------------------
    """
    if area == 'Cellar_JET' or area == 'Lower_JET' or area == 'Mezz_JET':
        x = np.append(x, 500)
        y = np.append(y, 0.5)
    return x, y

def loe_leak_rate(df,loe_area='utility'):
    """Calculate loss of escape per leak rate for one of the loss of escape
    areas defined in start_nodes.csv
    Input: Dataframe df with all results
    Output: Dataframe p_imp (used further in loe_leak_category)
    """
    p_imp=df.pivot_table(index=['Area','Leak rate'],columns=loe_area,
    values='f_scen',aggfunc='sum',fill_value=0)
    p_imp.reset_index(inplace=True)
    p_imp=p_imp.rename(columns={0.0:'no_imp',1.0:'Impprob'})
    p_imp.drop('no_imp',axis=1,inplace=True)
    if not 'Impprob' in p_imp:
        p_imp['Impprob']=0 #Needed since the pivot table does not include if all zero
    return p_imp

def loe_leak_category(df,leakfz_file='leakfz', extrapolation=False):
    """Input: A dataframe (df) with impairment probabilites for each leak rate. Example:
        utility        Area  Leak rate   Impprob
        0        Cellar_JET          5  0.000000
        1        Cellar_JET         15  0.042300
        2        Cellar_JET         30  0.058192
        3        Cellar_JET        100  0.098817
        4        Lower_POOL         21  0.000000

      Output: A new dataframe with impairment probabilities for S, Me, Ma, L. Example:
                 Area Leak category   Impprob  Impprob escalated  Freq weighted
        0  Cellar_JET             S  0.000000           0.070379           True
        1  Cellar_JET            Me  0.002277           0.072115           True
        2  Cellar_JET            Ma  0.050238           0.083392           True
        3  Cellar_JET             L  0.072881           0.096715           True
        0  Lower_POOL             S  0.000000           0.000000           True
        1  Lower_POOL            Me  0.000000           0.000000           True
        2  Lower_POOL            Ma  0.000000           0.000000           True
        3  Lower_POOL             L  0.000000           0.000000           True

    IMPORTANT: The function extrapolate has not yet been properly implemented,
               hence the argument extrapolation should be set to False
    """
    all_results=[]
    for area in df['Area'].unique():
        sim_leak_rates=np.array(df.loc[df['Area']==area]['Leak rate'])
        impprob=np.array(df.loc[df['Area']==area]['Impprob'])
        if extrapolation: #Not yet implemented
            sim_leak_rates, impprob = extrapolate(sim_leak_rates, impprob, area)
        intp_impprob=interpolate_values(sim_leak_rates,impprob)
        leakfz=read_leakfz(area,leakfz_file)
        if leakfz is not None:
            intp_leakfz=interpolate_values(leakfz[0],leakfz[1])
            f_weighted=True
        else:
            intp_leakfz=None
            f_weighted=False
        impprob_categories=[leak_categorize(intp_impprob,category,
            intp_leakfz) for category in ['S','Me','Ma','L']]
        intp_impprob_esc=escalate(intp_impprob, 50) # <-- Hardcoded, why?
        impprob_categories_esc=[leak_categorize(intp_impprob_esc,category,
            intp_leakfz) for category in ['S','Me','Ma','L']]
        d = {'Area': area, 'Leak category': ['S','Me','Ma','L'], 
            'Impprob': impprob_categories, 'Impprob escalated': impprob_categories_esc, 
            'Freq weighted': f_weighted}
        all_results.append(d)
    df_all=pd.concat([pd.DataFrame(data=d) for d in all_results])
    df_all = df_all[['Area','Leak category','Impprob','Impprob escalated','Freq weighted']] #reorganise
    return df_all

def summarise_leak_rates(df, start_nodes='start_nodes.csv'):
    """Input: A dataframe (df) with all results, and a list of start nodes
       Output: Loss of escape probabilites from each area, per fire area and leak rate
    """
    p_imp_tot=df.groupby(['Area','Leak rate']).size().reset_index().drop([0],axis=1) #Improve?
    for area in pd.read_csv(start_nodes).area_name.unique():
        p_imp = loe_leak_rate(df, loe_area=area)
        p_imp = p_imp.rename(columns={'Impprob': '{}'.format(area)})
        p_imp_tot = pd.merge(p_imp_tot,p_imp)
    return p_imp_tot

def summarise_leak_categories(p_imp_tot, leakfz_file='leakfz', esc_rate=50, extrapolation=False):
    """Input:  p_imp_tot (created from summarise_leak_rates)
       Output: Probabilites from each area, per fire area and leak category
               Escalated fires are also included
    """
    loe_areas=p_imp_tot.drop(columns=['Area','Leak rate']).columns
    dummy2=p_imp_tot.iloc[:,:3]
    dummy2.columns=['Area','Leak rate', 'Impprob']
    p_imp_cat_tot=loe_leak_category(dummy2)
    p_imp_cat_tot=p_imp_cat_tot.loc[:,'Area':'Leak category'].reset_index(drop=True)
    #print(p_imp_cat_tot) #Can be done better; can it be avoided with better merging somehow?
    
    for loe_area in loe_areas:  
        foo=pd.DataFrame()
        for area in p_imp_tot['Area'].unique():
            df=p_imp_tot.loc[p_imp_tot['Area']==area,['Area','Leak rate',loe_area]]
            sim_leak_rates=np.array(df['Leak rate'])
            impprob=np.array(df[loe_area])
            if extrapolation: #Not yet implemented
                sim_leak_rates, impprob = extrapolate(sim_leak_rates, impprob, area)
            intp_impprob=interpolate_values(sim_leak_rates,impprob)
            leakfz=read_leakfz(area,leakfz_file)
            if leakfz is not None:
                intp_leakfz=interpolate_values(leakfz[0],leakfz[1])
                f_weighted=True
            else:
                intp_leakfz=None
                f_weighted=False
            impprob_categories=[leak_categorize(intp_impprob,category,
                intp_leakfz) for category in ['S','Me','L']] #Modified for Gudrun
            intp_impprob_esc=escalate(intp_impprob, esc_rate)
            impprob_categories_esc=[leak_categorize(intp_impprob_esc,category,
                intp_leakfz) for category in ['S','Me','L']] #Modified for Gudrun
            d = {'Area': area, 'Leak category': ['S','Me','L'], #Modified for Gudrun
                 str(loe_area): impprob_categories, str(loe_area)+'_esc': impprob_categories_esc,
                str(loe_area)+'_leakfz': f_weighted}
            foo=foo.append(pd.DataFrame(d))
        p_imp_cat_tot = pd.merge(p_imp_cat_tot,foo)
    return p_imp_cat_tot

#**********************************************************************************************
#**********************************************************************************************

def main():
    """Example calculations may be added here"""

if __name__ == '__main__':
    sys.exit(main())
