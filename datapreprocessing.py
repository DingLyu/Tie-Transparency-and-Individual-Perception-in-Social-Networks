#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ding lyu
"""

# this file includes:
# build standard global network(SGN)
# build the directed standard global network(DSGN)
# Fig.~S3: build the evolution from the initial global network to the standard network, and refactored global network
# Tab.~1
# Tab.~S1, Tab.~S2, Tab.~S3

import datetime as dt
import networkx as nx

# Tab.~S2
LINK_STATUS = {-3: 'Both Reject', -21: 'Target Reject Source Confirm', -12: 'Source Reject Target Confirm',
               -2: 'Target Reject', -1: 'Source Reject', 0: 'Both Unconfirm',
               1: 'Source Confirm', 2: 'Target Confirm', 3: 'Both Confirm'}

starttime = dt.datetime.strptime("2016-09-29 10:0:0", "%Y-%m-%d %H:%M:%S")  #2016-09-29 10:0:0
T1 = dt.datetime.strptime("2016-10-06 10:0:0", "%Y-%m-%d %H:%M:%S")
T2 = dt.datetime.strptime("2016-10-13 10:0:0", "%Y-%m-%d %H:%M:%S")
T3 = dt.datetime.strptime("2016-10-20 10:0:0", "%Y-%m-%d %H:%M:%S")
separatetime = dt.datetime.strptime("2016-10-27 10:0:0", "%Y-%m-%d %H:%M:%S")  #2016-10-27 10:0:0

# Tab.~S1
# read data from 10001.csv, format: source node, target node, link creater, status, creating time, confirmation time
def Read():
    f = open('data/10001.csv', 'r')
    Data = []
    l = f.readline()
    while l:
        Data.append(l.strip('\n').split(','))
        l = f.readline()
    del Data[0]
    return Data

# build the standard global network organized with 155 nodes and 1996 undirected unweighted edges
def StandardGlobalNetwork():
    Data = Read()
    G = nx.Graph()
    for data in Data:
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - separatetime).total_seconds() < 0:
            if data[0] == data[2] or data[1] == data[2]:
                if data[3] == '3':
                    G.add_edge(data[0], data[1])
    print('Standard Global Network(SGN):')
    print('Num of nodes:', nx.number_of_nodes(G))
    print('Num of nodes:', nx.number_of_edges(G))
    return G

# build the directed version of the standard global network of 155 nodes and 2824 directed edges
def DirectedGlobalNetwork():
    Data = Read()
    DG = nx.DiGraph()
    for data in Data:
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - separatetime).total_seconds() < 0:
            if data[0] == data[2] or data[1] == data[2]:
                if data[3] == '3':
                    DG.add_edge(data[0], data[1])
    return DG

# Tab.~S3
# statistics and classification
def Statistics():
    Data = Read()
    ALL = [[0 for j in range(9)]for i in range(5)]
    state = ['R_R', 'C_R', 'R_C', 'U_R', 'R_U', 'U_U', 'C_U', 'U_C', 'C_C']
    classify = ['-3', '-21', '-12', '-2', '-1', '0', '1', '2', '3']
    for data in Data:
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - starttime).total_seconds() <= 0:
            if data[3] in classify:
                ALL[0][classify.index(data[3])] += 1
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - starttime).total_seconds() > 0 and (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - T1).total_seconds() <= 0:
            if data[3] in classify:
                ALL[1][classify.index(data[3])] += 1
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - T1).total_seconds() > 0 and (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - T2).total_seconds() <= 0:
            if data[3] in classify:
                ALL[2][classify.index(data[3])] += 1
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - T2).total_seconds() > 0 and (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - T3).total_seconds() <= 0:
            if data[3] in classify:
                ALL[3][classify.index(data[3])] += 1
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - T3).total_seconds() > 0 and (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - separatetime).total_seconds() <= 0:
            if data[3] in classify:
                ALL[4][classify.index(data[3])] += 1
    for i in range(len(ALL)):
        if i != 0:
            for j in range(len(ALL[i])):
                ALL[i][j] += ALL[i-1][j]
        print("Phase%d:"%i)
        print(state)
        print(ALL[i])
        print("All:", sum(ALL[i]))

# Fig.~S3(a)
# snapshots at the five moments of the fist phase
def FiveSnapshots():
    Data = Read()
    L1, L2, L3, L4, L5 = [], [], [], [], []
    G1, G2, G3, G4, G5 = nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()
    for data in Data:
        if data[3] == '3':
            if data[0] == data[2] or data[1] == data[2]:
                if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - starttime).total_seconds() <= 0:
                    L1.append([data[0], data[1]])
                if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - T1).total_seconds() <= 0:
                    L2.append([data[0], data[1]])
                if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - T2).total_seconds() <= 0:
                    L3.append([data[0], data[1]])
                if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - T3).total_seconds() <= 0:
                    L4.append([data[0], data[1]])
                if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - separatetime).total_seconds() <= 0:
                    L5.append([data[0], data[1]])
    G1.add_edges_from(L1)
    G2.add_edges_from(L2)
    G3.add_edges_from(L3)
    G4.add_edges_from(L4)
    G5.add_edges_from(L5)
    return G1, G2, G3, G4, G5

# Fig.~S3(b)
# refactoring of the global network of 155 nodes and 2999 edges
def RefactoredGlobalNetwork():
    RG = nx.Graph()
    Data = Read()
    for data in Data:
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - separatetime).total_seconds() > 0:
            if data[0] != data[2] and data[1] != data[2]:
                RG.add_edge(data[0], data[1])
    return RG

# Tab.~1
# the table to verify Two Degrees of Perception with statistics of ties in the perception phase
def Table_TwoDegreesPerception():
    Data = Read()
    G = StandardGlobalNetwork()
    Count = 0
    C = [0, 0, 0, 0, 0, 0]
    label = ['1-1', '1-2', '2-2', '2-3', '3-3', '3-4']
    for data in Data:
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - separatetime).total_seconds() > 0:
            if G.has_edge(data[0], data[1]) and data[0] != data[2] and data[1] != data[2]:
                Count += 1
                x = nx.shortest_path_length(G, data[0], data[2]) + nx.shortest_path_length(G, data[1], data[2])
                C[x-2] += 1
    for i in range(len(C)):
        print(label[i], C[i], '%.2f %%'%(100*C[i]/Count))
    print('Total', Count, '100%')

