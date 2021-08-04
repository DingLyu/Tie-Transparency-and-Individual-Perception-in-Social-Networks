#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ding lyu
"""
# this file includes:
# plot Fig.~1, Fig.~2, Fig.~3, Fig.~4, Fig.~S4, Fig.~S5, Fig.~S6 except the visualization of networks

from utils import *
from datapreprocessing import *

# Fig.~1(a)
# correlation between edge embeddedness and tie transparency
def Correlation_EE_KD():
    G = StandardGlobalNetwork()
    KD = KnowableDegree()
    Embeddedness = {}
    for edge in G.edges():
        a = G.degree(edge[0])
        b = G.degree(edge[1])
        c = len(list(nx.common_neighbors(G, edge[0], edge[1])))
        Embeddedness[edge] = c / (a + b - c)
    E = max(Embeddedness.values())
    X = []
    Y = []
    Err = []
    H = [[]for i in range(27)]
    for edge in G.edges():
        e = Embeddedness[edge]/E
        if edge in KD:
            if KD[edge] >= 0:
                H[KD[edge]].append(e)
        elif (edge[1], edge[0]) in KD:
            if KD[(edge[1], edge[0])] >= 0:
                H[KD[(edge[1], edge[0])]].append(e)
    print(np.std(H[0]))
    print(len(H[0]))
    print(sp.stats.sem(H[0])*sp.stats.t._ppf((1+0.95)/2., 191))
    for i in range(27):
        if H[i] != []:
            X.append(i)
            Y.append(np.mean(H[i]))
            if i < 25:
                Err.append(mean_confidence_interval(H[i], confidence=0.99))
            else:
                Err.append(np.std(H[i]))
    plt.figure(figsize=(8,5))
    plt.errorbar(x=X, y=Y, yerr=Err, fmt="o",color="steelblue",ecolor='grey',elinewidth=.5,capsize=4)
    plt.xlabel('Tie Transparency', fontsize=20)
    plt.ylabel('Edge Embeddedness', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~1(b)
# correlation between edge betweenness and tie transparency
def Correlation_EB_KD():
    G = StandardGlobalNetwork()
    KD = KnowableDegree()
    Data = []
    edge_betweeness = nx.edge_betweenness_centrality(G)
    h = max(edge_betweeness.values())
    X = []
    Y = []
    Err = []
    H = [[]for i in range(27)]
    for e in G.edges():
        a = edge_betweeness[e]/h
        if e in KD:
            if KD[e] >= 0:
                b = KD[e]
        elif (e[1], e[0]) in KD:
            if KD[(e[1], e[0])] >= 0:
                b = KD[(e[1], e[0])]
        H[b].append(a)
    for i in range(27):
        if H[i] != []:
            X.append(i)
            Y.append(np.mean(H[i]))
            if i < 25:
                Err.append(mean_confidence_interval(H[i], confidence=0.99))
            else:
                Err.append(np.std(H[i]))
    plt.figure(figsize=(8,5))
    plt.errorbar(x=X, y=Y, yerr=Err, fmt="o",color="steelblue",ecolor='grey',elinewidth=.5,capsize=4)
    plt.xlabel('Tie Transparency', fontsize=20)
    plt.ylabel('Edge Betweenness', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~1(c)
# correlation between attribute similarity and tie transparency
# to protect the privacy, we hide the attributes of nodes (name, degree, grade, department, class, and gender)
def Correlation_AS_KD():
    from data import attribute as A
    KD = KnowableDegree()
    Err = []
    H = [[]for i in range(27)]
    X = []
    Y = []
    for tie in KD:
        if KD[tie] >= 0:
            vec1 = np.array(A.A[tie[0]])
            vec2 = np.array(A.A[tie[1]])
            distance = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * (np.linalg.norm(vec2)))
            H[KD[tie]].append(distance)
    for i in range(27):
        if H[i] != []:
            X.append(i)
            Y.append(np.mean(H[i]))
            if i < 25:
                Err.append(mean_confidence_interval(H[i], confidence=0.99))
            else:
                Err.append(np.std(H[i]))
    plt.figure(figsize=(8,5))
    plt.errorbar(x=X, y=Y, yerr=Err, fmt="o",color="steelblue",ecolor='grey',elinewidth=.5,capsize=4)
    plt.xlabel('Tie Transparency', fontsize=20)
    plt.ylabel('Attribute Similarity', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~1(e)
# Make communities emerge with the removal of edges in the KD ascending order and the KD descending order
def CommunityEmerge(): #KD从小到大和从大到小去除，看最大连通片规模
    G = StandardGlobalNetwork()
    Up = list()
    up = list()
    Down = list()
    down = list()
    KD = KnowableDegree()
    for i in range(68):
        for edge in G.edges():
            if KD[edge] == i:
                down.append(edge)
            if KD[edge] == 67 - i:
                up.append(edge)
    G = StandardGlobalNetwork()
    for e in up:
        G.remove_edge(e[0], e[1])
        Gc = len(list(max(nx.connected_components(G), key=len)))
        Up.append(Gc)
    G = StandardGlobalNetwork()
    for e in down:
        G.remove_edge(e[0], e[1])
        Gc = len(list(max(nx.connected_components(G), key=len)))
        Down.append(Gc)

    X = [i for i in range(101)]
    x = [i / 100 for i in range(101)]
    UP = []
    DOWN = []
    for i in X:
        UP.append(Up[round(i * 1995 / 100)] / 155)
        DOWN.append(Down[round(i * 1995 / 100)] / 155)

    fig = plt.figure(figsize=(10, 5))
    left, bottom, width, height = 0.085, 0.13, 0.865, 0.82
    ax1 = fig.add_axes([left, bottom, width, height])
    plt.tick_params(labelsize=15)
    ax1.set_xlabel("$f$", fontsize=20)
    ax1.set_ylabel("$S(f)$", fontsize=20)
    ax1.scatter(x, UP, marker='v', color='red', alpha=1, label="KD Descending")
    ax1.scatter(x, DOWN, marker='^', color='steelblue', alpha=1, label="KD Ascending")
    plt.legend(fontsize=15)

    community = mpimg.imread('community.png')
    newcommunity = mpimg.imread('newcommunity.png')

    left, bottom, width, height = 0.1, 0.3, 0.3, 0.3
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.axis('off')
    ax2.imshow(community)
    left, bottom, width, height = 0.4, 0.3, 0.3, 0.3
    ax3 = fig.add_axes([left, bottom, width, height])
    ax3.axis('off')
    ax3.imshow(newcommunity)
    plt.show()

# Fig.~1(f)
# distributions of tie transparency distinguished by same and different separate attributes
# including degree, grade, department, class, and gender.
# to protect the privacy, we hide the attributes of nodes (name, degree, grade, department, class, and gender)
def KD_Attributes():
    from data import attribute as A
    G = StandardGlobalNetwork()
    KD = KnowableDegree()
    Data = []
    for edge in G.edges():
        if A.A[edge[0]][:3] == A.A[edge[1]][:3]: # degree
            Data.append([KD[edge], 'Degree', 'Same'])
        else:
            Data.append([KD[edge], 'Degree', 'Different'])
        if A.A[edge[0]][3:7] == A.A[edge[1]][3:7]: # grade
            Data.append([KD[edge], 'Grade', 'Same'])
        else:
            Data.append([KD[edge], 'Grade', 'Different'])
        if A.A[edge[0]][7:13] == A.A[edge[1]][7:13]: # department
            Data.append([KD[edge], 'Department', 'Same'])
        else:
            Data.append([KD[edge], 'Department', 'Different'])
        if A.A[edge[0]][13:35] == A.A[edge[1]][13:35]: # class
            Data.append([KD[edge], 'Class', 'Same'])
        else:
            Data.append([KD[edge], 'Class', 'Different'])
        if A.A[edge[0]][35:] == A.A[edge[1]][35:]: # sex
            Data.append([KD[edge], 'Gender', 'Same'])
        else:
            Data.append([KD[edge], 'Gender', 'Different'])
    plt.figure(figsize=(8,5))
    df = pd.DataFrame(data=Data, columns=['Tie Transparency', 'Attributes', 'Label'])
    sns.violinplot(data=df, x='Attributes', y='Tie Transparency', hue='Label', palette=['#2980b9','#bdc3c7']
                   ,split=True, inner="quart", linewidth=1)
    plt.xlabel('Attributes', fontsize=20)
    plt.ylabel('Tie Transparency', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    legend = plt.legend(loc='upper right', title='Label', fontsize=15)
    legend.get_title().set_fontsize(fontsize=15)
    plt.show()

# Fig.~2(a)
# Explanation of the phenominon of two degrees of perception with the distribution of perceptual proportion
def TwoDegreesofPerception():
    A = np.load('PerceptualProportion.npy')
    X = np.arange(11)
    B = [[] for x in range(7)]
    for i in range(len(A)):
        count0, count1, count2, count3, count4, count5, count6, count7, count8, count9, count10 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for j in range(len(A[i])):
            if A[i][j] == 0:
                count0 += 1
            if A[i][j] <= 0.1 and A[i][j] > 0:
                count1 += 1
            if A[i][j] <= 0.2 and A[i][j] > 0.1:
                count2 += 1
            if A[i][j] <= 0.3 and A[i][j] > 0.2:
                count3 += 1
            if A[i][j] <= 0.4 and A[i][j] > 0.3:
                count4 += 1
            if A[i][j] <= 0.5 and A[i][j] > 0.4:
                count5 += 1
            if A[i][j] <= 0.6 and A[i][j] > 0.5:
                count6 += 1
            if A[i][j] <= 0.7 and A[i][j] > 0.6:
                count7 += 1
            if A[i][j] <= 0.8 and A[i][j] > 0.7:
                count8 += 1
            if A[i][j] <= 0.9 and A[i][j] > 0.8:
                count9 += 1
            if A[i][j] <= 1 and A[i][j] > 0.9:
                count10 += 1
        B[i] = [count0/len(A[i]), count1/len(A[i]), count2/len(A[i]), count3/len(A[i]), count4/len(A[i]), count5/len(A[i]), count6/len(A[i]), count7/len(A[i]), count8/len(A[i]), count9/len(A[i]), count10/len(A[i])]
    fig = plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.set_xticklabels(
        ["0", "(0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]", "(0.5,0.6]", "(0.6,0.7]", "(0.7,0.8]",
         "(0.8,0.9]", "(0.9,1]"])
    plt.xticks(np.linspace(0, 10, 11))
    plt.tick_params(labelsize=12)
    plt.yticks(fontsize=15)
    plt.xlabel("Perceptual Proportion", fontsize=25)
    plt.ylabel("f", fontsize=25)
    plt.plot(X, B[0], linestyle='-', color='#e74c3c', label='$1-1$ edge', alpha=1, marker='o')
    plt.plot(X, B[1], linestyle='-', color='#e67e22', label='$1-2$ edge', alpha=1, marker='8')
    plt.plot(X, B[2], linestyle='-', color='#f1c40f', label='$2-2$ edge', alpha=1, marker='h')
    plt.plot(X, B[3], linestyle='--', color='#1abc9c', label='$2-3$ edge', alpha=1, marker='v')
    plt.plot(X, B[4], linestyle='--', color='#2ecc71', label='$3-3$ edge', alpha=1, marker='^')
    plt.plot(X, B[5], linestyle='--', color='#3498db', label='$3-4$ edge', alpha=1, marker='<')
    plt.plot(X, B[6], linestyle='--', color='#9b59b6', label='$4-4$ edge', alpha=1, marker='>')
    plt.ylim([0, 1])
    plt.legend(fontsize=15)
    plt.show()

# Fig.~3(a)
# Observation of Recovery Coverage when more participants contribute to the reconstruction of the global network
def RecoveryCoverage():
    P, M = IndividualPerception()
    G = StandardGlobalNetwork()
    X = [i/G.number_of_nodes() for i in range(G.number_of_nodes()+1)]
    Cont1 = Contribution(1)
    Cont2 = Contribution(2)
    Cont3 = Contribution(3)

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[1], x[2]), reverse=True)
    Y = [0]
    edgelist = {edge:0 for edge in G.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values())/G.number_of_edges())
    plt.plot(X, Y, 'red', linestyle='--', label='Count only')

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[2], x[1]), reverse=True)
    Y = [0]
    edgelist = {edge: 0 for edge in G.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values()) / G.number_of_edges())
    plt.plot(X, Y, 'black', linestyle='-.', label='Accuracy only')

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[3],x[1]), reverse=True)
    Y = [0]
    edgelist = {edge: 0 for edge in G.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values()) / G.number_of_edges())
    plt.plot(X, Y, 'steelblue', linestyle='-', label='Our model')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('$f$', fontsize=20)
    plt.ylabel('Recovery Coverage', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    legend = plt.legend(loc='lower right', title='Model', fontsize=15)
    legend.get_title().set_fontsize(fontsize=15)
    plt.show()

# Fig.~3(b)
# Observation of Recovery Accuracy when more participants contribute to the reconstruction of the global network
def RecoveryAccuracy():
    P, M = IndividualPerception()
    G = StandardGlobalNetwork()
    g = RefactoredGlobalNetwork()
    X = [i / G.number_of_nodes() for i in range(G.number_of_nodes() + 1)]
    Cont1 = Contribution(1)
    Cont2 = Contribution(2)
    Cont3 = Contribution(3)

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[1], x[2]), reverse=True)

    Y = [1]
    edgelist = {edge: 0 for edge in G.edges()}
    Edgelist = {edge: 0 for edge in g.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        if M[N] != []:
            for l in M[N]:
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values()) / sum(Edgelist.values()))
    plt.plot(X, Y, 'red', linestyle='--', label='Count only')

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[2], x[1]), reverse=True)
    Y = [1]
    edgelist = {edge: 0 for edge in G.edges()}
    Edgelist = {edge: 0 for edge in g.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        if M[N] != []:
            for l in M[N]:
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values()) / sum(Edgelist.values()))
    plt.plot(X, Y, 'black', linestyle='-.', label='Accuracy only')

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[3],x[1]), reverse=True)
    Y = [1]
    edgelist = {edge: 0 for edge in G.edges()}
    Edgelist = {edge: 0 for edge in g.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        if M[N] != []:
            for l in M[N]:
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values()) / sum(Edgelist.values()))

    plt.plot(X, Y, 'steelblue', linestyle='-', label='Our model')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('$f$', fontsize=20)
    plt.ylabel('Recovery Accuracy', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    legend = plt.legend(loc='lower right', title='Model', fontsize=15)
    legend.get_title().set_fontsize(fontsize=15)
    plt.show()

# Fig.~4(a)
# correlation between node degree and individual perceived capability (IPC)
# IPC is calculated by utils.contribution(3)
def Correlation_Degree_IPC():
    G = StandardGlobalNetwork()
    X, Y = [], []
    Cont3 = Contribution(3)
    for node in G.nodes():
        X.append(G.degree(node))
        Y.append(Cont3[node])
    x1, y1 = [], []
    for i in range(len(Y)):
        if Y[i]>0 and X[i]>0:
            x1.append(np.log(X[i]))
            y1.append(np.log(Y[i]))
    p = np.polyfit(x1, y1, 1)
    plt.figure(figsize=(8,6))
    x2 = np.arange(2, 70,1)
    y2 = np.exp(np.polyval(p, np.log(x2)))
    plt.loglog(X, Y, '*', x2, y2, '-')
    plt.xlabel('Degree', fontsize=20)
    plt.ylabel('Individual Perceptual Capability', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~4(b)
# correlation between node betweenness and individual perceived capability (IPC)
# IPC is calculated by utils.contribution(3)
def Correlation_Betweenness_IPC():
    G = StandardGlobalNetwork()
    X, Y = [], []
    Cont3 = Contribution(3)
    B = nx.centrality.betweenness_centrality(G)
    for node in G.nodes():
        X.append(B[node])
        Y.append(Cont3[node])
    x1, y1 = [], []
    for i in range(len(Y)):
        if Y[i]>0 and X[i]>0:
            x1.append(np.log(X[i]))
            y1.append(np.log(Y[i]))
    p = np.polyfit(x1, y1, 1)
    x2 = np.arange(0.0001, 0.2, 0.001)
    y2 = np.exp(np.polyval(p, np.log(x2)))
    plt.figure(figsize=(8,6))
    plt.loglog(X, Y, '*', x2, y2, '-')
    plt.xlabel('Betweenness', fontsize=20)
    plt.ylabel('Individual Perceptual Capability', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~S4(a)
# distribution of knowable degree (perceived ties and misperceived ties)
def KDdis():
    KD = KnowableDegree()
    P = []
    M = []
    for tie in KD:
        if KD[tie] < 0:
            M.append(KD[tie])
        else:
            P.append(KD[tie])
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    for i in range(27):
        X1.append(i)
        Y1.append(P.count(i))

    for i in range(11):
        X2.append(-i-1)
        Y2.append(M.count(-i-1))
    plt.xlabel('Tie Transparency', fontsize=20)
    plt.ylabel('Count',fontsize=20)
    plt.bar(X1, Y1, color='#2980b9', label='Perceived Ties')
    plt.bar(X2, Y2, color='#bdc3c7', label='Misperceived Ties')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    legend = plt.legend(loc='upper right', title='Label', fontsize=15)
    legend.get_title().set_fontsize(fontsize=15)
    plt.show()

# Fig.~S4(b)
# distribution of individuals who have several (x-axis) covert ties
def CTdis():
    H = CovertTies()
    G = StandardGlobalNetwork()
    CD = {node:0 for node in G.nodes()}
    for h in H:
        CD[h[0]]+= 1
        CD[h[1]] += 1
    L = list(CD.values())
    X = []
    Y = []
    for i in range(max(L)+1):
        X.append(i)
        Y.append(L.count(i))
    plt.bar(X, Y, color='grey')
    plt.xlabel('Covert Ties', fontsize=20)
    plt.ylabel('$N(p)$', fontsize=20)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
               ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~S5(a-c)
# Distributions of edge embeddedness and edge betweenness of covert ties, transparent ties, and the ratio of covert ties in all ties
def TvsC(flag):

    G = StandardGlobalNetwork()
    KD = KnowableDegree()
    Data = []

    Embeddedness = {}
    for edge in G.edges():
        a = G.degree(edge[0])
        b = G.degree(edge[1])
        c = len(list(nx.common_neighbors(G, edge[0], edge[1])))
        Embeddedness[edge] = c / (a + b - c)
    e = max(Embeddedness.values())
    edge_betweeness = nx.edge_betweenness_centrality(G)
    b = max(edge_betweeness.values())
    H1 = []
    H2 = []
    for edge in G.edges():
        if KD[edge] == 0:
            H1.append([Embeddedness[edge]/e, edge_betweeness[edge]/b]) #covert
        else:
            H2.append([Embeddedness[edge]/e, edge_betweeness[edge]/b]) #transparent
    Hi1 = [[0 for i in range(20)] for j in range(20)]
    Hi2 = [[0 for i in range(20)] for j in range(20)]
    for h in H1:
        if h[0] == 1:
            if h[1] == 1:
                Hi1[19][19] += 1
            else:
                Hi1[19][int(h[1]*20)] += 1
        else:
            if h[1] == 1:
                Hi1[int(h[0]*20)][19] += 1
            else:
                Hi1[int(h[0]*20)][int(h[1]*20)] += 1
    for h in H2:
        if h[0] == 1:
            if h[1] == 1:
                Hi2[19][19] += 1
            else:
                Hi2[19][int(h[1]*20)] += 1
        else:
            if h[1] == 1:
                Hi2[int(h[0]*20)][19] += 1
            else:
                Hi2[int(h[0]*20)][int(h[1]*20)] += 1
    for i in range(20):
        for j in range(20):
            if flag == 1:
                Data.append([i,j,Hi1[i][j]])
            if flag == 2:
                Data.append([i,j,Hi2[i][j]])
            if flag == 3:
                if Hi1[i][j] + Hi2[i][j] == 0:
                    Data.append([i,j,0])
                else:
                    Data.append([i,j, Hi1[i][j]/(Hi1[i][j] + Hi2[i][j])])
    df = pd.DataFrame(Data, columns=['A', 'B', 'num'])
    flights = df.pivot("B", "A", "num")
    plt.figure(figsize=(6,5))
    g = sns.heatmap(flights,linewidths=.1)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
               ['0', ' ', '0.1', ' ', '0.2', ' ', '0.3', ' ', '0.4', ' ', '0.5', ' ', '0.6', ' ', '0.7', ' ', '0.8', ' ', '0.9', ' ', '1'])
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
               ['0', ' ', '0.1', ' ', '0.2', ' ', '0.3', ' ', '0.4', ' ', '0.5', ' ', '0.6', ' ', '0.7', ' ', '0.8', ' ', '0.9', ' ', '1'])
    g.set_xlabel('Edge Embeddedness', fontsize=20)
    g.set_ylabel('Edge Betweenness', fontsize=20)
    plt.show()


# Fig.~S6(a)
# distribution of covert ties with each same/different attribute
def CovertTies_Attributes():
    from data import attribute as A
    KD = KnowableDegree()
    G = StandardGlobalNetwork()
    DG = DirectedGlobalNetwork()
    Data = []
    plt.figure(figsize=(6,5))
    Total = {'Degree':0, 'Grade':0, 'Department':0, 'Class':0, 'Gender':0}
    Covert = {'Degree':0, 'Grade':0, 'Department':0, 'Class':0, 'Gender':0}
    for edge in G.edges():
        if A.A[edge[0]][:3] != A.A[edge[1]][:3]:  # degree
            Total['Degree'] += 1
            if KD[edge] == 0:
                Covert['Degree'] += 1
        if A.A[edge[0]][3:7] != A.A[edge[1]][3:7]:  # grade
            Total['Grade'] += 1
            if KD[edge] == 0:
                Covert['Grade'] += 1
        if A.A[edge[0]][7:13] != A.A[edge[1]][7:13]:  # department
            Total['Department'] += 1
            if KD[edge] == 0:
                Covert['Department'] += 1
        if A.A[edge[0]][13:35] != A.A[edge[1]][13:35]:  # class
            Total['Class'] += 1
            if KD[edge] == 0:
                Covert['Class'] += 1
        if A.A[edge[0]][35:] != A.A[edge[1]][35:]:  # gender
            Total['Gender'] += 1
            if KD[edge] == 0:
                Covert['Gender'] += 1
    for key in Total:
        Data.append([key, Total[key], Covert[key]])
    df = pd.DataFrame(Data, columns=['Attributes', 'Different', 'Covert'])
    df = df.sort_values("Different", ascending=False)
    plt.subplot(211)
    sns.barplot(x="Different", y="Attributes", data=df,
                label="Different", color="#ecf0f1")
    sns.barplot(x="Covert", y="Attributes", data=df,
                label="Covert", color='#bdc3c7')
    plt.legend(fontsize=10)
    plt.xlim([0,2000])
    plt.xlabel(' ')
    plt.tick_params(labelsize=10)
    plt.ylabel(' ')
    Data = []
    Total = {'Degree':0, 'Grade':0, 'Department':0, 'Class':0, 'Gender':0}
    Covert = {'Degree':0, 'Grade':0, 'Department':0, 'Class':0, 'Gender':0}
    for edge in G.edges():
        if A.A[edge[0]][:3] == A.A[edge[1]][:3]:  # degree
            Total['Degree'] += 1
            if KD[edge] == 0:
                Covert['Degree'] += 1
        if A.A[edge[0]][3:7] == A.A[edge[1]][3:7]:  # grade
            Total['Grade'] += 1
            if KD[edge] == 0:
                Covert['Grade'] += 1
        if A.A[edge[0]][7:13] == A.A[edge[1]][7:13]:  # department
            Total['Department'] += 1
            if KD[edge] == 0:
                Covert['Department'] += 1
        if A.A[edge[0]][13:35] == A.A[edge[1]][13:35]:  # class
            Total['Class'] += 1
            if KD[edge] == 0:
                Covert['Class'] += 1
        if A.A[edge[0]][35:] == A.A[edge[1]][35:]:  # gender
            Total['Gender'] += 1
            if KD[edge] == 0:
                Covert['Gender'] += 1
    for key in Total:
        Data.append([key, Total[key], Covert[key]])
    df = pd.DataFrame(Data, columns=['Attributes', 'Same', 'Covert'])
    df = df.sort_values("Same", ascending=False)
    plt.subplot(212)
    sns.barplot(x="Same", y="Attributes", data=df,
                label="Same", color='#3498db')
    sns.barplot(x="Covert", y="Attributes", data=df,
                label="Covert", color='#2980b9')
    plt.legend(fontsize=10)
    plt.xlim([0,2000])
    plt.ylabel(' ')
    plt.xlabel('Count',fontsize=15)
    plt.tick_params(labelsize=10)
    plt.show()

