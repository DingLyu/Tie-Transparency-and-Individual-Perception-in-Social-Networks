#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ding lyu
"""
# this file includes:
# compute tie transparency by knowable degree (KD)
# find out 192 covert ties (KD=0)
# find out perceived ties and misperceived tis of 155 individuals
# compute individual perceived capability
# Generate the gexf file of Fig.~1(d), Fig.~1(e), Fig.~2(b), Fig.~S6(b)(c), Fig.~S7(a-c)

import math as mt
import numpy as np
import scipy as sp
import pandas as pd
import datetime as dt
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats
from datapreprocessing import *

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return h

# tie transparency is characterized by the knowable degree
# compute the knowable degree (KD)
def KnowableDegree():
    G = StandardGlobalNetwork()
    Data = Read()
    KD = dict()
    Perception = []
    for data in Data:
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - separatetime).total_seconds() > 0:
            if data[0] != data[2] and data[1] != data[2]:
                if int(data[0]) < int(data[1]):
                    Perception.append((data[0], data[1]))
                else:
                    Perception.append((data[1], data[0]))
    for edge in G.edges():
        if edge in Perception or (edge[1], edge[0]) in Perception:
            KD[edge] = Perception.count(edge) + Perception.count((edge[1], edge[0]))
        else:
            KD[edge] = 0
    for c in Perception:
        if not G.has_edge(c[0], c[1]):
            KD[(c[0], c[1])] = 0-Perception.count(c)
    return KD

# find out all cover ties that can not be perceived by any other third-party
def CovertTies():
    G = StandardGlobalNetwork()
    Data = Read()
    Perception = []
    CT = []
    for data in Data:
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - separatetime).total_seconds() > 0:
            if data[0] != data[2] and data[1] != data[2]:
                Perception.append((data[0], data[1]))
    for edge in G.edges():
        if edge not in Perception and (edge[1], edge[0]) not in Perception:
            CT.append(edge)
    return CT

# compute perception of 155 individuals
def IndividualPerception():
    G = StandardGlobalNetwork()
    Data = Read()
    Perception = {node: [] for node in G.nodes()}
    Misperception = {node: [] for node in G.nodes()}
    for data in Data:
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - separatetime).total_seconds() > 0:
            if data[0] != data[2] and data[1] != data[2]:
                if G.has_edge(data[0], data[1]):
                    Perception[data[2]].append((data[0], data[1]))
                else:
                    Misperception[data[2]].append((data[0], data[1]))
    return Perception, Misperception

# quantify the individual perceived capability of all participants
def Contribution(flag):
    G = StandardGlobalNetwork()
    P, M = IndividualPerception()
    KD = KnowableDegree()
    Cont = dict()

    if flag == 1:                 # accurate number
        for node in G.nodes():
            Cont[node] = len(P[node])
        return Cont

    if flag == 2:                          # accuracy
        for node in G.nodes():
            if len(P[node])+len(M[node]) == 0:
                Cont[node] = 0
            else:
                Cont[node] = len(P[node])/(len(P[node])+len(M[node]))
        return Cont

    if flag == 3:                    # our model
        for node in G.nodes():
            Cont[node] = 0
            if P[node] != []:
                for i in P[node]:
                    if (i[0], i[1]) in KD:
                        t = KD[(i[0], i[1])]
                    elif (i[1], i[0]) in KD:
                        t = KD[(i[1], i[0])]
                    Cont[node] += 1 / t
            if M[node] != []:
                for i in M[node]:
                    if (i[0], i[1]) in KD:
                        t = KD[(i[0], i[1])]
                    elif (i[1], i[0]) in KD:
                        t = KD[(i[1], i[0])]
                    Cont[node] += 1 / t
        return Cont

    if flag == 4:                     # log-improved model
        maximum = max(KD.values())
        minimum = min(KD.values())
        for node in G.nodes():
            Cont[node] = 0
            if P[node] != []:
                for i in P[node]:
                    if (i[0], i[1]) in KD:
                        t = KD[(i[0], i[1])]
                    elif (i[1], i[0]) in KD:
                        t = KD[(i[1], i[0])]
                    Cont[node] += 1 - mt.log(t) / mt.log(maximum)
            if M[node] != []:
                for i in M[node]:
                    if (i[0], i[1]) in KD:
                        t = KD[(i[0], i[1])]
                    elif (i[1], i[0]) in KD:
                        t = KD[(i[1], i[0])]
                    Cont[node] -= 0.5 * (1 - mt.log(-1 * t) / mt.log(-1 * minimum))
        return Cont

# Generate the gexf file of Fig.~1(d)
def Example_Embeddedness_KD():
    G = StandardGlobalNetwork()
    Data = Read()
    Perceivers = {edge: [] for edge in G.edges()}
    for data in Data:
        if (dt.datetime.strptime(data[4][:19], "%Y-%m-%d %H:%M:%S") - separatetime).total_seconds() > 0:
            if data[0] != data[2] and data[1] != data[2]:
                if (data[0], data[1]) in Perceivers:
                    Perceivers[(data[0], data[1])].append(data[2])
                if (data[1], data[0]) in Perceivers:
                    Perceivers[(data[1], data[0])].append(data[2])
    example = ('10043', '10062')
    Example = nx.Graph()
    for node in G.nodes():
        if node == example[0] or node == example[1]:
            Example.add_node(node, core=0)
        elif node in Perceivers[example]:
            Example.add_node(node, core=-1)
        elif node in nx.common_neighbors(G, example[0], example[1]):
            Example.add_node(node, core=1)
        else:
            Example.add_node(node, core=2)
    for edge in G.edges():
        Example.add_edge(edge[0], edge[1], weight=1)
    Example.get_edge_data(example[0], example[1])['weight'] = 10
    nx.write_gexf(Example, 'ExampleKD.gexf')

# Generate the gexf file of Fig.~1(e)
def CommunitytoNewcommunity():
    KD = KnowableDegree()
    Community = nx.Graph()
    Newcommunity = nx.Graph()
    for tie in KD:
        if KD[tie] >= 6:
            Community.add_edge(tie[0], tie[1])
        if KD[tie] >= 7:
            Newcommunity.add_edge(tie[0], tie[1])
    nx.write_gexf(Community, 'Community.gexf')
    nx.write_gexf(Newcommunity, 'Newcommunity.gexf')

# Generate the gexf file of Fig.~2(b)
# visualization of an ego network and its two degrees of perception
def ExampleofTDP():
    P, M = IndividualPerception()
    G = StandardGlobalNetwork()
    ego = '10041'
    Example = nx.Graph()
    perception = P[ego]
    for node in G.nodes():
        if node == ego:
            Example.add_node(ego, core=0)
        else:
            Example.add_node(node, core=nx.shortest_path_length(G, ego, node))
    for tie in perception:
        Example.add_edge(tie[0], tie[1], weight=8)
    for edge in G.edges():
        if not Example.has_edge(edge[0], edge[1]):
            Example.add_edge(edge[0], edge[1], weight=1)
    nx.write_gexf(Example, 'ExampleTDP.gexf')

# Generate the gexf file of Fig.~S6(b)(c)
def CovertTie_GlobalNetworks():
    G = StandardGlobalNetwork()
    DG = DirectedGlobalNetwork()
    KD = KnowableDegree()
    CTinBG = nx.Graph()
    CTinDG = nx.Graph()
    for edge in G.edges():
        if (DG.has_edge(edge[0], edge[1]) and DG.has_edge(edge[1], edge[0])):
            if KD[edge] == 0:
                CTinBG.add_edge(edge[0], edge[1], weight=8)
            else:
                CTinBG.add_edge(edge[0], edge[1], weight= 1)
        else:
            if KD[edge] == 0:
                CTinDG.add_edge(edge[0], edge[1], weight=8)
            else:
                CTinDG.add_edge(edge[0], edge[1], weight= 1)
    nx.write_gexf(CTinBG, 'Bi.gexf')
    nx.write_gexf(CTinDG, 'Di.gexf')

# Generate the gexf file of Fig.~S7(a-c)
# visualization of top 20 perceivers quantified by three models
def Top20Peceivers():
    P, M = IndividualPerception()
    Cont1 = Contribution(1)
    Cont2 = Contribution(2)
    Cont3 = Contribution(3)
    G = StandardGlobalNetwork()
    G1 = nx.Graph()
    G2 = nx.Graph()
    G3 = nx.Graph()
    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist1 = sorted(nodelist, key=lambda x: (x[1], x[2]), reverse=True)
    nodelist2 = sorted(nodelist, key=lambda x: (x[2], x[1]), reverse=True)
    nodelist3 = sorted(nodelist, key=lambda x: (x[3], x[1]), reverse=True)

    for i in range(155):
        if i < 20:
            G1.add_node(nodelist1[i][0], top=1)
            G2.add_node(nodelist2[i][0], top=1)
            G3.add_node(nodelist3[i][0], top=1)
        else:
            G1.add_node(nodelist1[i][0], top=0)
            G2.add_node(nodelist2[i][0], top=0)
            G3.add_node(nodelist3[i][0], top=0)

    for edge in G.edges():
        G1.add_edge(edge[0], edge[1], perceived=1)
        G2.add_edge(edge[0], edge[1], perceived=1)
        G3.add_edge(edge[0], edge[1], perceived=1)
    for i in range(155):
        if i < 20:
            PN = P[nodelist1[i][0]]
            for tie in PN:
                G1[tie[0]][tie[1]]['perceived'] = 8
        if i < 20:
            PN = P[nodelist2[i][0]]
            for tie in PN:
                G2[tie[0]][tie[1]]['perceived'] = 8
        if i < 20:
            PN = P[nodelist3[i][0]]
            for tie in PN:
                G3[tie[0]][tie[1]]['perceived'] = 8

    nx.write_gexf(G1, 'G1.gexf')
    nx.write_gexf(G2, 'G2.gexf')
    nx.write_gexf(G3, 'G3.gexf')